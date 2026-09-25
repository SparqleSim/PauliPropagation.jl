###
##
# Monte Carlo propagation: propagate deterministically as usual, and whenever the ensemble
# grows past `max_size`, resample it back down via `resample!`. This reuses the exact same
# `applymergetruncate!` pipeline that `propagate!` uses, just with a resampling step tacked on.
##
###

"""
    mcpropagate(circuit, tsum::AbstractTermSum, thetas=nothing; max_size, resampling_size=round(Int, max_size/2), resample_func=nothing, squared=false, thread=true, kwargs...)
    mcpropagate(circuit, prop_cache::AbstractPropagationCache, thetas=nothing; max_size, resampling_size=round(Int, max_size/2), resample_func=nothing, squared=false, thread=true, kwargs...)

Monte Carlo variant of `propagate()`.
Once the number of terms exceeds `max_size`, the running term sum is resampling down (close) to resampling_size.
`resample_func` selects the resampling strategy (see `resample!`), and `squared` is passed to it.
Further options of a strategy are set by passing a closure as `resample_func`.
`kwargs` are passed to `applymergetruncate!` (e.g. `min_abs_coeff`, `max_weight`).
For Pauli sums, `heisenberg=true` additionally selects the Heisenberg vs. Schrödinger picture (see `propagate`).
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
mcpropagate(circuit, object, thetas=nothing; kwargs...) = mcpropagate!(circuit, deepcopy(object), thetas; kwargs...)

"""
    mcpropagate!(circuit, psum::AbstractTermSum, thetas=nothing; thread=true, kwargs...)
    mcpropagate!(circuit, prop_cache::AbstractPropagationCache, thetas=nothing; thread=true, kwargs...)

In-place version of `mcpropagate`. See `mcpropagate` for details.
"""
function mcpropagate!(circuit, psum::AbstractTermSum, thetas=nothing; kwargs...)
    prop_cache = mcpropagate!(circuit, PropagationCache(psum), thetas; kwargs...)
    return extractsum!(prop_cache, psum)
end

function mcpropagate!(circuit, prop_cache::AbstractPropagationCache, thetas=nothing; kwargs...)
    return PropagationBase._propagate!(applymergetruncateresample!, circuit, prop_cache, thetas; kwargs...)
end

"""
    applymergetruncateresample!(gate, prop_cache::AbstractPropagationCache, args...; max_size::Real, resampling_size::Integer=round(Int, max_size/2), resample_func=nothing, squared=false, thread=true, kwargs...)

Like `applymergetruncate!`, but afterwards resamples `prop_cache` down to `resampling_size` terms
(via `resample!`) whenever it exceeds `max_size`. This is the per-gate step function behind `mcpropagate!`.
`resample_func` and `squared` go to `resample!`, and `kwargs` to `applymergetruncate!`.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function applymergetruncateresample!(gate, prop_cache::AbstractPropagationCache, args...; max_size::Real, resampling_size::Integer=round(Int, max_size / 2), resample_func=nothing, squared::Bool=false, thread::Bool=true, kwargs...)
    applymergetruncate!(gate, prop_cache, args...; thread, kwargs...)

    if length(prop_cache) > max_size
        resample!(prop_cache, resampling_size; resample_func, squared, thread)
    end

    return prop_cache
end
