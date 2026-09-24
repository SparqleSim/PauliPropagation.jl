###
##
# A one- or two-qubit gate acts on at most two 64-bit limbs of a Pauli string, however many limbs the string has.
# Commutation and the rotation sign add up over the limbs, so a rule built for the gate's own limbs decides from the
# same limbs of a Pauli string exactly as it does from the whole strings, at a cost that does not grow with the number of qubits.
##
###

"""
    OnLimbs(rule, inds)

`rule`, asked about the two 64-bit limbs `inds` of every Pauli string, as an `NTupleInteger{2}`, instead of the whole string.
`_onlimbs` builds `rule` for the same two limbs of the gate.
"""
struct OnLimbs{R}
    rule::R
    inds::NTuple{2,Int}
end

# the array kernels come with an index, and a view of `terms` as a matrix of 64-bit words, one term per column, loads only the two limbs
@inline function PropagationBase.ruleat(r::OnLimbs, terms, coefficients, ii::Int)
    words = reinterpret(reshape, UInt64, terms)
    term_limbs = NTupleInteger{2}((@inbounds(words[r.inds[1], ii]), @inbounds(words[r.inds[2], ii])))
    return @inline r.rule(term_limbs, (@inbounds coefficients[ii]))
end

# every other storage comes with the whole term
@inline function (r::OnLimbs)(pstr, coeff)
    pstr_limbs = _limbs(pstr)
    return @inline r.rule(NTupleInteger{2}((pstr_limbs[r.inds[1]], pstr_limbs[r.inds[2]])), coeff)
end

"""
    _onlimbs(makerule, gate_mask)

The rule `makerule(gate_mask)`, or, when the Pauli strings span several 64-bit limbs and the gate acts on at most two,
`makerule` of those two limbs of `gate_mask`, asked about the same two limbs of every Pauli string by an `OnLimbs`.
Called once per gate, never per Pauli string.
"""
function _onlimbs(makerule::F, gate_mask::Union{UInt128,NTupleInteger}) where {F}
    gate_limbs = _limbs(gate_mask)
    acted_on = findall(!iszero, gate_limbs)
    if isempty(acted_on) || length(acted_on) > 2
        return makerule(gate_mask)
    end

    # a gate within one limb names that limb twice, the second time with an empty mask, which changes nothing
    first_ind, last_ind = first(acted_on), last(acted_on)
    second_limb = first_ind == last_ind ? zero(UInt64) : gate_limbs[last_ind]
    return OnLimbs(makerule(NTupleInteger{2}((gate_limbs[first_ind], second_limb))), (first_ind, last_ind))
end

_onlimbs(makerule::F, gate_mask) where {F} = makerule(gate_mask)

# the 64-bit limbs of the term types wider than one limb, in the order they lie in memory
_limbs(pstr::NTupleInteger) = pstr.limbs
_limbs(pstr::UInt128) = reinterpret(NTuple{2,UInt64}, pstr)
