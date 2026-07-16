###
##
# Monte Carlo "path sampling": instead of deterministically splitting a Pauli term into a
# cosine- and a sine-branch at every non-commuting gate, randomly keep only one branch,
# reweighted so that the result stays unbiased in expectation. The size of the Pauli sum
# never changes, so no merging or truncation is needed.
##
###

"""
    mcsample(circuit, tsum::AbstractTermSum, params=nothing; squared=false, heisenberg=true, kwargs...)
    mcsample(circuit, prop_cache::AbstractPropagationCache, params=nothing; squared=false, heisenberg=true, kwargs...)

Monte Carlo "path sampling" counterpart to `propagate`. 
Each term term in the term sum randomly samples a branch that a gate applies.
For squared=false, that yields an unbiased sample of the propagated term sum, the the coefficients will likely grow expoenntially.
Average many independent calls (or pack many copies of the same term into one large `tsum`) to
converge the result.
Use `squared=true` to sample with probabilities proportional to squared coefficients instead of their
absolute value (useful for e.g. 2-norm/OTOC-type estimators).
"""
mcsample(circuit, psum, params=nothing; kwargs...) = mcsample!(circuit, deepcopy(psum), params; kwargs...)

"""
    mcsample!(circuit, tsum::AbstractTermSum, params=nothing; squared=false, heisenberg=true, kwargs...)
    mcsample!(circuit, prop_cache::AbstractPropagationCache, params=nothing; squared=false, heisenberg=true, kwargs...)

In-place version of `mcsample`. See `mcsample` for details.
"""
function mcsample!(circuit, prop_cache::AbstractPropagationCache, params=nothing; kwargs...)
    # manipulates the active view of the prop_cache in place
    mcsample!(circuit, activesum(prop_cache), params; kwargs...)
    return prop_cache
end

function mcsample!(circuit, tsum::AbstractTermSum, params=nothing; heisenberg=true, kwargs...)
    circuit, params = _preparecircuit(circuit, params, heisenberg)
    return PropagationBase._propagate!(mcapplytoall!, circuit, tsum, params; kwargs...)
end


function mcapplytoall!(gate, object, args...; kwargs...)
   @throw error("mcapplytoall! not implemented for gate type $(typeof(gate)) and object type $(typeof(object)).")
end