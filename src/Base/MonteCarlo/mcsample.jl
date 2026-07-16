# TODO: make top-level more general and also adapt to exactly the same high-level code as propagate!()
# TODO: little code-duplication and same entrypoints for propagate, mcpropagate and mcsample.

function mcsample(circuit, psum::VectorPauliSum, params=nothing; kwargs...)
    psum = mcsample!(circuit, deepcopy(psum), params; kwargs...)
    return psum
end
function mcsample!(circuit, prop_cache::VectorPauliPropagationCache, params=nothing; kwargs...)
    # manupulates the active view of the prop_cache in place
    mcsample!(circuit, activesum(prop_cache), params; kwargs...)
    return prop_cache
end

function mcsample!(circuit, psum::VectorPauliSum, params; heisenberg=true, kwargs...)
    # if circuit is actually a single gate, promote it to a list [gate]
    # similarly the params if it is a single number
    circuit, params = PropagationBase._promotecircandparams(circuit, params)

    # if params is nothing, the circuit must contain only StaticGates
    # also check if the length of params equals the number of parametrized gates
    PropagationBase._checknumberofparams(circuit, params)

    if heisenberg
        # this usually just reverses circuit and parameter order
        circuit, params = toheisenberg(circuit, params)
    else
        # this usually entails a conversion of how gates act
        circuit, params = toschrodinger(circuit, params)
    end

    parameter_iterator = Iterators.Stateful(params)

    for gate in circuit
        if isa(gate, ParametrizedGate)
            param = popfirst!(parameter_iterator)
            mcapplytoall!(gate, psum, param; kwargs...)
        else
            mcapplytoall!(gate, psum; kwargs...)
        end
    end
    return psum
end
