###
##
# A one- or two-qubit rotation only ever changes two bytes of a Pauli string, however long that
# string is. Reading just those two bytes makes the commutation check and the rotation sign cost the
# same at any number of qubits, instead of growing with it.
#
# It also avoids shifting the whole string, which gets several times slower above a certain length
# (around 600 qubits on Julia 1.10, around 1050 on Julia 1.12).
##
###

"""
    _masksetbits(symbols, qinds)

The bit positions a gate's Pauli mask sets, ascending. Read off the gate rather than by scanning the
mask, so the cost is the same for any Pauli string.
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

A gate mask together with the at most two bytes it touches. A qubit's two bits never straddle a byte
boundary, so two bytes cover any one- or two-qubit gate. A gate inside a single byte repeats that
byte with an empty mask, which changes nothing and saves a branch.
"""
struct ByteMask{TT}
    mask::TT
    inds::NTuple{2,Int}
    bytes::NTuple{2,UInt8}
end

# set to false to work on the whole string at once, for timing comparisons
const USE_LOCAL_GATES = Ref(true)

# shorter than this, working on the whole string at once is cheaper
const _MIN_LOCAL_BYTES = 24

"""
    _bytemask(gate_mask, bits, terms)

Wrap `gate_mask` for the two-byte path, or return it unchanged when that path does not apply. Called
once per gate, so the cost of the wrapping is paid per gate and never per Pauli string.
"""
function _bytemask(gate_mask::TT, bits, terms) where {TT}
    (!USE_LOCAL_GATES[] || isempty(bits) || sizeof(TT) < _MIN_LOCAL_BYTES || !(terms isa Vector)) && return gate_mask

    # bits is ascending, so a third byte can only lie strictly between the outer two
    i1 = bits[1] >> 3
    i2 = bits[end] >> 3
    any(b -> i1 < b >> 3 < i2, bits) && return gate_mask

    b1 = b2 = 0x00
    for b in bits
        b >> 3 == i1 ? (b1 |= 0x01 << (b & 7)) : (b2 |= 0x01 << (b & 7))
    end
    return ByteMask{TT}(gate_mask, (i1 + 1, i2 + 1), (b1, b2))
end

_plainmask(m::ByteMask) = m.mask
_plainmask(gate_mask) = gate_mask

# ...0101: the low bit of every Pauli pair
const _ALTBYTE = 0x55

# per-byte pieces of `_bitcommutes` and `_calculatesignexponent`
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

# Read by address out of the terms array: picking a byte out of a Pauli string itself would copy the
# whole string first. Strings sit `aligned_sizeof` apart, which padding can make wider than `sizeof`.
# Only valid inside the `GC.@preserve` in the gate loop, and only for a CPU array (`_bytemask` checks).
_bytesof(terms::Vector, ::ByteMask) = Ptr{UInt8}(pointer(terms))
_bytesof(terms, gate_mask) = terms

@inline _byteat(bytes::Ptr{UInt8}, ii::Int, ind::Int, ::ByteMask{TT}) where {TT} =
    unsafe_load(bytes + (ii - 1) * Base.aligned_sizeof(TT) + ind - 1)

"""
    _gatecommutes(gate_mask, pstr, bytes, ii)
    _gateproduct(productfunc, gate_mask, pstr, bytes, ii)

Commutation check and rotation product for the Pauli string at index `ii`, where `bytes` is
`_bytesof(terms, gate_mask)`. Any mask that is not a `ByteMask` falls back to the whole-string
versions.
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

    # as in `paulirotationproduct`: sign == real(im * im^exponent)
    return pstr ⊻ m.mask, (exponent & 2) - 1
end
