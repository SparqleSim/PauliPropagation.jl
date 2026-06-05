
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
