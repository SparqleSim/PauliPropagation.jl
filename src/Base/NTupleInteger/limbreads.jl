###
##
# A rule of `xorbranch!` that reads a term only through the mask it branches by decides from the 64-bit limbs where the
# mask is not zero, however many limbs the terms have. A gate on one or two qubits acts on at most two limbs of a Pauli
# string, commutation and the rotation sign add up over the limbs, and the Paulis under the mask lie in those limbs,
# so the rule built for the gate's own limbs decides from the same limbs of every Pauli string as it does from the
# whole strings, at a cost that does not grow with the number of qubits.
##
###

"""
    onlimbs(buildrule, mask, args...)

The rule `buildrule(mask, args...)` of `xorbranch!`, which has to read a term only through `mask`.
Where the terms span more 64-bit limbs than `mask` acts on, the rule is built for those limbs of `mask` instead, and an `OnLimbs` asks it about the same limbs of every term.
Called once per gate, never per term.
"""
function onlimbs(buildrule::F, mask, args::Vararg{Any,K}) where {F,K}
    inds = limbspan(mask)
    if isnothing(inds)
        return buildrule(mask, args...)
    end
    return OnLimbs(buildrule(limbwindow(mask, inds), args...), inds)
end

"""
    OnLimbs(rule, inds)

`rule`, asked about the two 64-bit limbs `inds` of every term, as an `NTupleInteger{2}`, instead of the whole term.
`onlimbs` builds it.
"""
struct OnLimbs{R}
    rule::R
    inds::NTuple{2,Int}
end

# the array kernels come with an index, and a view of `terms` as a matrix of 64-bit words, one term per column, loads only the two limbs
@inline function ruleat(r::OnLimbs, terms, coefficients, ii::Int)
    words = reinterpret(reshape, UInt64, terms)
    term_limbs = NTupleInteger{2}((@inbounds(words[r.inds[1], ii]), @inbounds(words[r.inds[2], ii])))
    return @inline r.rule(term_limbs, (@inbounds coefficients[ii]))
end

# every other storage comes with the whole term
@inline (r::OnLimbs)(term, coefficient) = @inline r.rule(limbwindow(term, r.inds), coefficient)

"""
    limbspan(mask)

The first and last 64-bit limb of `mask` that are not zero, or `nothing` when more than two are, 
when none is, or when the terms have at most two limbs, so that a whole term is as cheap to read.
"""
limbspan(mask) = nothing
function limbspan(mask::NTupleInteger)
    n_acted_on = count(!iszero, mask.limbs)
    if n_acted_on == 0 || n_acted_on > 2 || length(mask.limbs) <= 2
        return nothing
    end
    return (findfirst(!iszero, mask.limbs)::Int, findlast(!iszero, mask.limbs)::Int)
end

"""
    limbwindow(term, inds)

The limbs `inds` of `term` as one `NTupleInteger{2}`.
A window within one limb names it twice, and the second copy is left empty so that it changes nothing.
"""
@inline function limbwindow(term::NTupleInteger, inds::NTuple{2,Int})
    second = inds[1] == inds[2] ? zero(UInt64) : term.limbs[inds[2]]
    return NTupleInteger{2}((term.limbs[inds[1]], second))
end
