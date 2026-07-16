function mcpropagate!(
    circuit, prop_cache::AbstractPropagationCache, thetas; max_size::Real, resampling_size::Real, kwargs...
)

    # TODO: exactly the same architecture as propagate!

    # for ease, freeze circuit with params
    circuit = freeze(circuit, thetas)

    rev_circuit = reverse(circuit)

    for gate in rev_circuit
        applymergetruncateresample!(gate, prop_cache; max_size=max_size, resampling_size=resampling_size, kwargs...)

    end

    return prop_cache
end

function applymergetruncateresample!(gate, prop_cache::AbstractPropagationCache, args...; max_size::Real, resampling_size=round(Int, max_size/2), kwargs...)
    
    # TODO: figure out how to perform relative coefficient truncation
    applymergetruncate!(gate, prop_cache, args...; kwargs...)

    if activesize(prop_cache) > max_size
        # TODO: default to semi-deterministic resampling
        # TODO: allow for resampling by 2-norm. This will dispatch to multinomial
        prop_cache = resampling_func!(prop_cache, resampling_size)
    end

    return prop_cache
end