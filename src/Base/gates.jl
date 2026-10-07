
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

Utility function to count the number of gates of type `ParametrizedGate` in a circuit.
"""
function countparameters(circuit)
    nparams = 0
    for gate in circuit
        nparams += isa(gate, ParametrizedGate)
    end
    return nparams
end

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
