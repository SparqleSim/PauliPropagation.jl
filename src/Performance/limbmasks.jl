###
##
# A one- or two-qubit rotation acts on at most two 64-bit limbs of a Pauli string, however many limbs the string has.
# Commutation and the rotation sign add up over the limbs, so the gate's limbs and the Pauli string's limbs in the same
# places decide both exactly as the whole strings do, at a cost that does not grow with the number of qubits.
##
###

"""
    LimbMask{TT}

A gate mask of term type `TT` together with the at most two 64-bit limbs it acts on: their indices `inds`, and the
gate's own limbs there as an `NTupleInteger{2}`. A gate within one limb names that limb twice, the second time with
an empty mask, which changes nothing and keeps one type for every gate.
"""
struct LimbMask{TT}
    mask::TT
    inds::NTuple{2,Int}
    limbs::NTupleInteger{2}
end

# the 64-bit limbs of the term types wider than one limb, in the order they lie in memory,
# which is the order `_gateandterm` reads a term's limbs in
_limbs(pstr::NTupleInteger) = pstr.limbs
_limbs(pstr::UInt128) = reinterpret(NTuple{2,UInt64}, pstr)

"""
    _limbmask(gate_mask)

Wrap `gate_mask` in a `LimbMask` when its term type is wider than one limb and it acts on at most two limbs,
or return it unchanged. Called once per gate, never per Pauli string.
"""
function _limbmask(gate_mask::Union{UInt128,NTupleInteger})
    gate_limbs = _limbs(gate_mask)
    acted_on = findall(!iszero, gate_limbs)
    if isempty(acted_on) || length(acted_on) > 2
        return gate_mask
    end

    first_ind, last_ind = first(acted_on), last(acted_on)
    second_limb = first_ind == last_ind ? zero(UInt64) : gate_limbs[last_ind]
    return LimbMask(gate_mask, (first_ind, last_ind), NTupleInteger{2}((gate_limbs[first_ind], second_limb)))
end

_limbmask(gate_mask) = gate_mask

_plainmask(m::LimbMask) = m.mask
_plainmask(gate_mask) = gate_mask

"""
    _gateandterm(gate_mask, terms, ii)

The gate and the Pauli string at index `ii` as far as a rule has to read them: both cut down to the limbs of a
`LimbMask`, and both whole for any other mask. The limbs are read through a view of `terms` as a matrix of 64-bit
words, one term per column, which loads only those words.
"""
@inline _gateandterm(gate_mask, terms, ii::Int) = gate_mask, (@inbounds terms[ii])

@inline function _gateandterm(m::LimbMask{TT}, terms::AbstractVector{TT}, ii::Int) where {TT}
    words = reinterpret(reshape, UInt64, terms)
    term_limbs = (@inbounds(words[m.inds[1], ii]), @inbounds(words[m.inds[2], ii]))
    return m.limbs, NTupleInteger{2}(term_limbs)
end
