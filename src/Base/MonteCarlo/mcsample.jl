###
##
# Monte Carlo "path sampling": instead of deterministically splitting a Pauli term into a
# cosine- and a sine-branch at every non-commuting gate, randomly keep only one branch,
# reweighted so that the result stays unbiased in expectation. The size of the Pauli sum
# never changes, so no merging or truncation is needed.
##
###

"""
    mcsample(circuit, tsum::AbstractTermSum, params=nothing; squared=false, kwargs...)
    mcsample(circuit, prop_cache::AbstractPropagationCache, params=nothing; squared=false, kwargs...)

Monte Carlo "path sampling" counterpart to `propagate`.
Each term term in the term sum randomly samples a branch that a gate applies.
For squared=false, that yields an unbiased sample of the propagated term sum, though the coefficients will likely grow exponentially.
Average many independent calls (or pack many copies of the same term into one large `tsum`) to
converge the result.
Use `squared=true` to sample with probabilities proportional to squared coefficients instead of their
absolute value (useful for e.g. 2-norm/OTOC-type estimators).
For Pauli sums, `heisenberg=true` additionally selects the Heisenberg vs. Schrödinger picture (see `propagate`).
"""
mcsample(circuit, psum, params=nothing; kwargs...) = mcsample!(circuit, deepcopy(psum), params; kwargs...)

"""
    mcsample!(circuit, tsum::AbstractTermSum, params=nothing; squared=false, kwargs...)
    mcsample!(circuit, prop_cache::AbstractPropagationCache, params=nothing; squared=false, kwargs...)

In-place version of `mcsample`. See `mcsample` for details.
"""
function mcsample!(circuit, prop_cache::AbstractPropagationCache, params=nothing; kwargs...)
    # manipulates the active view of the prop_cache in place
    mcsample!(circuit, activesum(prop_cache), params; kwargs...)
    return prop_cache
end

function mcsample!(circuit, tsum::AbstractTermSum, params=nothing; kwargs...)
    return PropagationBase._propagate!(mcapplytoall!, circuit, tsum, params; kwargs...)
end


function mcapplytoall!(gate, psum, args...; kwargs...)
    _thrownotimplemented(gate, :mcapplytoall!)
end