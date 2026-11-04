
"""
Abstract type for gates. 
"""
abstract type Gate end

"""
Abstract type for parametrized gates.
"""
abstract type ParametrizedGate <: Gate end

"""
Abstract type for static gates that are not parametrized.
"""
abstract type StaticGate <: Gate end


"""
    countparameters(circuit)

Utility function to count the number of gates in a circuit that take a parameter, see `isparametrized`.
"""
function countparameters(circuit)
    nparams = 0
    for gate in circuit
        nparams += isparametrized(gate)
    end
    return nparams
end

"""
    isparametrized(gate)

Returns whether `gate` takes a parameter when a circuit is propagated.
By default these are the gates of type `ParametrizedGate`.
"""
isparametrized(gate) = gate isa ParametrizedGate

"""
    qinds(gate)

Returns the indices of the qubits that `gate` acts on.
By default these are the `qinds` field of the gate, or else its `qind` field.
Gates with neither field need to overload `qinds`.
"""
function qinds(gate::Gate)
    if hasfield(typeof(gate), :qinds)
        return gate.qinds
    elseif hasfield(typeof(gate), :qind)
        return gate.qind
    else
        throw(ArgumentError("`qinds` is not defined for `$(typeof(gate))`, which has no `qinds` or `qind` field. Overload `PropagationBase.qinds` for it."))
    end
end
