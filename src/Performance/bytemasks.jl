###
##
# A one- or two-qubit rotation only ever touches two bytes, or two 64-bit words, of a Pauli string,
# however long that string is. Reading just those makes the commutation check and the rotation sign
# cost the same at any number of qubits, and avoids shifting the whole string, which gets several
# times slower above a certain length (around 600 qubits on Julia 1.10, around 1050 on Julia 1.12).
# Both also answer the commutation question without loading the string at all.
##
###

# shorter than this, working on the whole string at once is cheaper
const _MIN_LOCAL_BYTES = 24

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

"""
    WordMask{TT,N}

A gate mask together with the `N` 64-bit words it touches, at most two for a one- or two-qubit gate,
and per word the masks `ma`, `mb` that make the commutation test a parity:
`isodd(count_ones(((w >> 1) & ma) ⊻ (w & mb)))`, from `(a0 & b1) ⊻ (a1 & b0)` per qubit with the
gate's bit pair `a` fixed.
"""
struct WordMask{TT,N}
    mask::TT
    inds::NTuple{N,Int}
    words::NTuple{N,UInt64}
    ma::NTuple{N,UInt64}
    mb::NTuple{N,UInt64}
end

_plainmask(m::ByteMask) = m.mask
_plainmask(m::WordMask) = m.mask
_plainmask(gate_mask) = gate_mask

"""
    _gatemask(gate_mask, terms)

Wrap `gate_mask` for whichever local path applies, or return it unchanged when none does. Called once
per gate, never per Pauli string.
"""
function _gatemask(gate_mask::TT, terms) where {TT}
    local_mask = _bytemask(gate_mask, terms)
    local_mask isa ByteMask && return local_mask
    word_mask = _wordmask(gate_mask, terms)
    return word_mask === nothing ? gate_mask : word_mask
end

# `nothing` when the words cannot be read directly, or the gate spreads over more than two of them
function _wordmask(gate_mask::TT, terms) where {TT}
    little_endian = Base.ENDIAN_BOM == 0x04030201
    (!little_endian || !(terms isa Vector) || sizeof(TT) % 8 != 0) && return nothing

    bits = PropagationBase._masksetbits(gate_mask)
    isempty(bits) && return nothing

    inds = unique(b >> 6 for b in bits)
    length(inds) > 2 && return nothing

    return length(inds) == 1 ? _wordmask(gate_mask, inds, Val(1)) : _wordmask(gate_mask, inds, Val(2))
end

function _wordmask(gate_mask::TT, inds, ::Val{N}) where {TT,N}
    low = PauliPropagation.alternatingmask(zero(UInt64))
    words = ntuple(k -> (gate_mask >> (64 * inds[k])) % UInt64, Val(N))
    return WordMask{TT,N}(gate_mask, ntuple(k -> inds[k] + 1, Val(N)), words,
        ntuple(k -> words[k] & low, Val(N)), ntuple(k -> (words[k] >> 1) & low, Val(N)))
end

"""
    _bytemask(gate_mask, terms)

Wrap `gate_mask` for the two-byte path, or return it unchanged when that path does not apply.
"""
function _bytemask(gate_mask::TT, terms) where {TT}
    # the byte reads assume little-endian layout; ENDIAN_BOM is a constant, so the test folds away
    little_endian = Base.ENDIAN_BOM == 0x04030201
    (!little_endian || sizeof(TT) < _MIN_LOCAL_BYTES || !(terms isa Vector)) && return gate_mask

    bits = PropagationBase._masksetbits(gate_mask)
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

_bytesof(terms::Vector, ::ByteMask) = Ptr{UInt8}(pointer(terms))
_bytesof(terms::Vector, ::WordMask) = Ptr{UInt64}(pointer(terms))
_bytesof(terms, gate_mask) = terms

@inline _byteat(bytes::Ptr{UInt8}, ii::Int, m::ByteMask{TT}, k::Int) where {TT} =
    unsafe_load(bytes + (ii - 1) * Base.aligned_sizeof(TT) + m.inds[k] - 1)

@inline _wordat(words::Ptr{UInt64}, ii::Int, m::WordMask{TT}, k::Int) where {TT} =
    unsafe_load(words + (ii - 1) * Base.aligned_sizeof(TT) + 8 * (m.inds[k] - 1))

"""
    _gatecommutes(gate_mask, pstr, bytes, ii)
    _gatecommutesat(gate_mask, terms, bytes, ii)
    _gateproduct(gate_mask, pstr, bytes, ii)

Commutation check and rotation product for the Pauli string at index `ii`, where `bytes` is
`_bytesof(terms, gate_mask)`. Any mask that is not a `ByteMask` falls back to the whole-string
versions. `_gatecommutesat` answers without the string, where the mask can.
"""
@inline _gatecommutes(gate_mask, pstr, bytes, ii) = PauliPropagation.commutes(gate_mask, pstr)
@inline _gateproduct(gate_mask, pstr, bytes, ii) = PauliPropagation.paulirotationproduct(gate_mask, pstr)

@inline _gatecommutesat(gate_mask, terms, bytes, ii) = _gatecommutes(gate_mask, (@inbounds terms[ii]), bytes, ii)

# the words anticommute independently, so the string commutes when they do so an even number of times
@inline function _gatecommutesat(m::WordMask{TT,N}, terms, words, ii) where {TT,N}
    flags = ntuple(Val(N)) do k
        w = _wordat(words, ii, m, k)
        count_ones(((w >> 1) & m.ma[k]) ⊻ (w & m.mb[k]))
    end
    return iseven(sum(flags))
end

# the words contribute independent factors of im, so their exponents add
@inline function _gateproduct(m::WordMask{TT,N}, pstr, words, ii) where {TT,N}
    exponent = sum(ntuple(k -> PauliPropagation._calculatesignexponent(m.words[k], _wordat(words, ii, m, k)), Val(N)))

    # as in `paulirotationproduct`: sign == real(im * im^exponent)
    return pstr ⊻ m.mask, (exponent & 2) - 1
end

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
