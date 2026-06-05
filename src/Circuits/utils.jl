### Circuits/utils.jl
##
# Utility functions for circuits.
##
###


"""
    getparameterindices(circuit, GateType<:ParametrizedGate)

Utility function to get the parameter indices of gates of type `GateType` in a circuit.
This naturally only works for gates that subtype `ParametrizedGate`.
"""
function getparameterindices(circuit, ::Type{GT}) where {GT<:ParametrizedGate}
    indices = Int[]

    param_idx = 1
    for gate in circuit
        if isa(gate, GT)
            push!(indices, param_idx)
        end
        if isa(gate, ParametrizedGate)
            param_idx += 1
        end
    end
    return indices
end

"""
    getparameterindices(circuit, ::PauliRotation, gate_symbols::Vector{Symbol}))

Utility function to get the parameter indices of `PauliRotation` gates with symbol `gate_symbols`.
For example, `getparameterindices(circuit, PauliRotation, [:X])`.
"""
function getparameterindices(circuit, ::Type{PauliRotation}, gate_symbols::Vector{Symbol})
    indices = Int[]

    param_idx = 1
    for gate in circuit
        if isa(gate, PauliRotation) && gate.symbols == gate_symbols
            push!(indices, param_idx)
        end
        if isa(gate, ParametrizedGate)
            param_idx += 1
        end
    end
    return indices
end

"""
    getparameterindices(circuit, ::PauliRotation, gate_symbols::Vector{Symbol}), qinds::Vector{Int})

Utility function to get the parameter indices of `PauliRotation` gates with symbol `gate_symbols` acting on the qubits `qinds`.
For example, `getparameterindices(circuit, PauliRotation, [:X], [1])`.
"""
function getparameterindices(circuit, ::Type{PauliRotation}, gate_symbols::Vector{Symbol}, qinds::Vector{Int})
    indices = Int[]

    param_idx = 1
    for gate in circuit
        if isa(gate, PauliRotation) && gate.symbols == gate_symbols && gate.qinds == qinds
            push!(indices, param_idx)
        end
        if isa(gate, ParametrizedGate)
            param_idx += 1
        end
    end
    return indices
end

