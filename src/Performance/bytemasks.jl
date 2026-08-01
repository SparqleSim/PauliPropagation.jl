###
##
# A one- or two-qubit rotation only ever changes two bytes of a Pauli string, however long that
# string is. Reading just those two bytes makes the commutation check and the rotation sign cost the
# same at any number of qubits, and avoids shifting the whole string, which gets several times slower
# above a certain length (around 600 qubits on Julia 1.10, around 1050 on Julia 1.12).
##
###

# shorter than this, working on the whole string at once is cheaper
const _MIN_LOCAL_BYTES = 24

# the positions of the set bits of `mask`, ascending
function _masksetbits(mask::TT) where {TT}
    bits = Int[]
    while !iszero(mask)
        push!(bits, trailing_zeros(mask))
        mask &= mask - one(TT)
    end
    return bits
end

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

_plainmask(m::ByteMask) = m.mask
_plainmask(gate_mask) = gate_mask

"""
    _bytemask(gate_mask, terms)

Wrap `gate_mask` for the two-byte path, or return it unchanged when that path does not apply. Called
once per gate, never per Pauli string.
"""
function _bytemask(gate_mask::TT, terms) where {TT}
    (sizeof(TT) < _MIN_LOCAL_BYTES || !(terms isa Vector)) && return gate_mask

    bits = _masksetbits(gate_mask)
    isempty(bits) && return gate_mask

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

### Hooks used by the fused gate loop

# Read by address out of the terms array: picking a byte out of a Pauli string itself would copy the
# whole string first. Strings sit `aligned_sizeof` apart, and byte k holds bits 8k and up, so this
# wants a little-endian machine. Only valid inside the `GC.@preserve` in the gate loop.
_bytesof(terms::Vector, ::ByteMask) = Ptr{UInt8}(pointer(terms))
_bytesof(terms, gate_mask) = terms

@inline _byteat(bytes::Ptr{UInt8}, ii::Int, m::ByteMask{TT}, k::Int) where {TT} =
    unsafe_load(bytes + (ii - 1) * Base.aligned_sizeof(TT) + m.inds[k] - 1)

"""
    _gatecommutes(gate_mask, pstr, bytes, ii)
    _gateproduct(gate_mask, pstr, bytes, ii)

Commutation check and rotation product for the Pauli string at index `ii`, where `bytes` is
`_bytesof(terms, gate_mask)`. Any mask that is not a `ByteMask` falls back to the whole-string
versions.
"""
@inline _gatecommutes(gate_mask, pstr, bytes, ii) = PauliPropagation.commutes(gate_mask, pstr)
@inline _gateproduct(gate_mask, pstr, bytes, ii) = PauliPropagation.paulirotationproduct(gate_mask, pstr)

# two bytes commute overall when they anticommute in the same number of places, so when they agree
@inline function _gatecommutes(m::ByteMask, pstr, bytes, ii)
    return PauliPropagation._bitcommutes(m.bytes[1], _byteat(bytes, ii, m, 1)) ==
           PauliPropagation._bitcommutes(m.bytes[2], _byteat(bytes, ii, m, 2))
end

# the two bytes contribute independent factors of im, so their exponents add
@inline function _gateproduct(m::ByteMask, pstr, bytes, ii)
    exponent = PauliPropagation._calculatesignexponent(m.bytes[1], _byteat(bytes, ii, m, 1)) +
               PauliPropagation._calculatesignexponent(m.bytes[2], _byteat(bytes, ii, m, 2))

    # as in `paulirotationproduct`: sign == real(im * im^exponent)
    return pstr ⊻ m.mask, (exponent & 2) - 1
end
