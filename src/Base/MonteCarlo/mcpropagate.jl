###
##
# Monte Carlo propagation: propagate deterministically as usual, and whenever the ensemble
# grows past `max_size`, resample it back down via `resample!`. This reuses the exact same
# `applymergetruncate!` pipeline that `propagate!` uses, just with a resampling step tacked on.
##
###

"""
    mcpropagate(circuit, psum::VectorPauliSum, thetas=nothing; max_size, resampling_size=round(Int, max_size/2), resample_func=nothing, heisenberg=true, kwargs...)

Propagate `psum` through `circuit` like `propagate`, except that whenever the number of terms
exceeds `max_size`, the ensemble is resampled down to `resampling_size` terms via `resample!`.
`resample_func` selects the resampling strategy (see `resample!`); `kwargs` are also passed to
`applymergetruncate!` (e.g. `min_abs_coeff`, `max_weight`) and to the resampling strategy (e.g. `power`).
"""
mcpropagate(circuit, psum::VectorPauliSum, thetas=nothing; kwargs...) = mcpropagate!(circuit, deepcopy(psum), thetas; kwargs...)

"""
    mcpropagate!(circuit, psum::VectorPauliSum, thetas=nothing; kwargs...)
    mcpropagate!(circuit, prop_cache::VectorPauliPropagationCache, thetas=nothing; heisenberg=true, kwargs...)

In-place version of `mcpropagate`. See `mcpropagate` for details.
"""
function mcpropagate!(circuit, psum::VectorPauliSum, thetas=nothing; kwargs...)
    prop_cache = mcpropagate!(circuit, VectorPauliPropagationCache(psum), thetas; kwargs...)
    return extractsum!(prop_cache, psum)
end

function mcpropagate!(circuit, prop_cache::VectorPauliPropagationCache, thetas=nothing; heisenberg=true, kwargs...)
    circuit, thetas = _preparecircuit(circuit, thetas, heisenberg)
    return PropagationBase._propagate!(applymergetruncateresample!, circuit, prop_cache, thetas; kwargs...)
end

"""
    applymergetruncateresample!(gate, prop_cache::VectorPauliPropagationCache, args...; max_size::Real, resampling_size::Integer=round(Int, max_size/2), resample_func=nothing, kwargs...)

Like `applymergetruncate!`, but afterwards resamples `prop_cache` down to `resampling_size` terms
(via `resample!`) whenever it exceeds `max_size`. This is the per-gate step function behind `mcpropagate!`.
"""
function applymergetruncateresample!(gate, prop_cache::VectorPauliPropagationCache, args...; max_size::Real, resampling_size::Integer=round(Int, max_size / 2), resample_func=nothing, kwargs...)
    applymergetruncate!(gate, prop_cache, args...; kwargs...)

    if activesize(prop_cache) > max_size
        resample!(prop_cache, resampling_size; resample_func, kwargs...)
    end

    return prop_cache
end
