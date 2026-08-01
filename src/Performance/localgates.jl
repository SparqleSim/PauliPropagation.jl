###
##
# Local gate masks: make a one- or two-qubit Pauli rotation cost O(1) per term instead of O(nqubits).
#
# A wide Pauli string is one `BitIntegers` integer hundreds of bytes long, but a one- or two-qubit gate
# is nonzero in at most two of those bytes. The commutation check and the rotation sign therefore only
# ever need those two bytes, and reading just them removes the whole-string work from the gate loop.
#
# This also sidesteps a compiler cliff. Operations that need the whole value -- notably shifting it,
# which is what `commutes` and the weight counters are built on -- stop being expanded inline above a
# width that depends on the LLVM the Julia build ships with (about 1216 bits on LLVM 15, about 2112 on
# LLVM 18), and cost several times more from there on. Elementwise operations are unaffected. Working
# a byte at a time never forms a whole-value shift, so the gate loop has no threshold to fall off.
##
###

"""
    _masksetbits(symbols, qinds)

The bit positions a gate's Pauli mask sets, ascending. Qubit `q` occupies bits `2(q-1)` and
`2(q-1)+1`, holding its Pauli (I=0, X=1, Y=2, Z=3) low bit first. Reading these off the gate rather
than scanning the mask keeps the cost independent of how wide a Pauli string is, and asks nothing of
the Pauli string type -- which for the padded `BitIntegers` types cannot be reinterpreted to bytes
at all.
"""
function _masksetbits(symbols, qinds)
    bits = Int[]
    for (symbol, qind) in zip(_astuple(symbols), _astuple(qinds))
        pauli = PauliPropagation.symboltoint(symbol)
        !iszero(pauli & 1) && push!(bits, 2 * (qind - 1))
        !iszero(pauli & 2) && push!(bits, 2 * (qind - 1) + 1)
    end
    return sort!(bits)
end

_astuple(x::Union{Symbol,Integer}) = (x,)
_astuple(x) = x

"""
    ByteMask{TT}

A gate mask paired with the (at most two, not necessarily adjacent) bytes of a Pauli string it is
nonzero in. Used in place of the plain gate mask, it makes the commutation check and the rotation
sign cost O(1) instead of O(nqubits).

A qubit's two Pauli bits sit at an even bit position and the one above it, so they never straddle a
byte boundary and two bytes cover any one- or two-qubit gate whatever qubits it acts on, adjacent or
not. Byte indices are plain fields rather than type parameters, so all gates share one compiled
specialization no matter which bytes they touch. A gate confined to a single byte repeats that byte
with a zero mask, which contributes to neither the commutation flags nor the sign counts and so needs
no branch.
"""
struct ByteMask{TT}
    mask::TT
    inds::NTuple{2,Int}
    bytes::NTuple{2,UInt8}
end

# Reading two bytes only pays off once a Pauli string is wider than this; below it the string is a
# register or two, whole-string operations vectorize, and the generic path is cheaper.
const _MIN_LOCAL_BYTES = 24

"""
    _bytemask(gate_mask, bits, terms)

Wrap `gate_mask` (whose set bits are at `bits`) for the local gate path, or hand it back unchanged
when that path does not apply: too narrow a Pauli string, a gate spanning more than two bytes, or a
terms array with no pointer to read the bytes through. Called once per gate, so the type instability
of the result is paid per gate and never per Pauli string.
"""
function _bytemask(gate_mask::TT, bits, terms) where {TT}
    (isempty(bits) || sizeof(TT) < _MIN_LOCAL_BYTES || !(terms isa Vector)) && return gate_mask

    # `bits` is ascending, so a third byte can only show up strictly between the outer two
    i1 = bits[1] >> 3
    i2 = bits[end] >> 3
    any(b -> i1 < b >> 3 < i2, bits) && return gate_mask

    b1 = b2 = 0x00
    for b in bits
        b >> 3 == i1 ? (b1 |= 0x01 << (b & 7)) : (b2 |= 0x01 << (b & 7))
    end
    return ByteMask{TT}(gate_mask, (i1 + 1, i2 + 1), (b1, b2))
end

# The plain gate mask, whether or not it was wrapped for the local path.
_plainmask(m::ByteMask) = m.mask
_plainmask(gate_mask) = gate_mask

# ...0101: the low bit of every Pauli pair.
const _ALTBYTE = 0x55

# Per-byte pieces of `_bitcommutes` and `_calculatesignexponent`. Bytes outside the mask's support
# contribute nothing to either, because every term in them is gated by a zero mask bit.
@inline function _byteflags(a::UInt8, b::UInt8)
    return ((a & _ALTBYTE) & ((b >> 1) & _ALTBYTE)) ⊻ (((a >> 1) & _ALTBYTE) & (b & _ALTBYTE))
end

@inline function _bytesigncounts(a::UInt8, b::UInt8)
    a1 = (a >> 1) & _ALTBYTE
    a2 = a & _ALTBYTE
    b1 = (b >> 1) & _ALTBYTE
    b2 = b & _ALTBYTE
    notcommuting = (a1 | a2) & (b1 | b2) & ((a1 ⊻ b1) | (a2 ⊻ b2))
    negative = notcommuting & ((a1 ⊻ b2) | (~a2 & ~b1))
    return count_ones(notcommuting), count_ones(negative)
end

### Hooks used by the fused gate loop

# The bytes a gate touches are read straight out of the terms array. Going through the Pauli string
# itself would need a runtime index into it, which forces the whole thing onto the stack just to pick
# out two bytes.
#
# Strings lie `aligned_sizeof` apart, which for a `BitIntegers` type whose width is not a multiple of
# its alignment leaves a few bytes of padding between them (`UInt1800` is 225 bytes in a 240-byte
# slot). `reinterpret` refuses those padded types altogether, so the bytes are read through a pointer
# instead -- single-byte loads, so alignment never enters. The pointer is valid only while `terms` is
# alive, which the `GC.@preserve` in the gate loop guarantees, and only for a CPU array, which
# `_bytemask` checks before taking this path.
_bytesof(terms::Vector, ::ByteMask) = Ptr{UInt8}(pointer(terms))
_bytesof(terms, gate_mask) = terms

@inline _byteat(bytes::Ptr{UInt8}, ii::Int, ind::Int, ::ByteMask{TT}) where {TT} =
    unsafe_load(bytes + (ii - 1) * Base.aligned_sizeof(TT) + ind - 1)

"""
    _gatecommutes(gate_mask, pstr, bytes, ii)
    _gateproduct(productfunc, gate_mask, pstr, bytes, ii)

Commutation check and rotation product for the Pauli string at index `ii`, where `bytes` is
`_bytesof(terms, gate_mask)`. Both fall back to the generic full-width implementations for any gate
mask that is not a `ByteMask`.
"""
@inline _gatecommutes(gate_mask, pstr, bytes, ii) = PauliPropagation.commutes(gate_mask, pstr)
@inline _gateproduct(productfunc::PF, gate_mask, pstr, bytes, ii) where {PF} = productfunc(gate_mask, pstr)

@inline function _gatecommutes(m::ByteMask, pstr, bytes, ii)
    flags = _byteflags(m.bytes[1], _byteat(bytes, ii, m.inds[1], m)) ⊻
            _byteflags(m.bytes[2], _byteat(bytes, ii, m.inds[2], m))
    return iseven(count_ones(flags))
end

@inline function _gateproduct(::PF, m::ByteMask, pstr, bytes, ii) where {PF}
    n1, g1 = _bytesigncounts(m.bytes[1], _byteat(bytes, ii, m.inds[1], m))
    n2, g2 = _bytesigncounts(m.bytes[2], _byteat(bytes, ii, m.inds[2], m))
    exponent = (2 * (g1 + g2) + n1 + n2) & 3

    # same trick as `paulirotationproduct`: sign == real(im * im^exponent)
    return pstr ⊻ m.mask, (exponent & 2) - 1
end
