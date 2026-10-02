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


## Commuting layers of a circuit

# Split `circuit` into layers of gate indices: maximal sets of mutually commuting Pauli rotations
# (`PauliRotation` or frozen ones, see Gates/paulirotations.jl), or runs of other gates. A
# rotation moves to the earliest layer it reaches past rotations it commutes with, never past
# other gates, so the layers applied in order give the same operator as the circuit; a Trotter
# step falls apart into one layer per Pauli type.
function _commutinglayers(circuit)
    layers = Vector{Int}[]
    isempty(circuit) && return layers

    # a circuit has no qubit count; the widest rotation sets the integer type
    max_qind = maximum((maximum(_rotationof(gate).qinds) for gate in circuit if _isrotation(gate)); init=1)
    TT = getinttype(max_qind)
    # the Pauli strings of every layer; an empty entry marks a layer of other gates. A rotation
    # put into layer k+1 commutes with every rotation in layers k+1 and later, so the layers
    # applied in order reproduce the circuit.
    pstrs = Vector{TT}[]
    for (ii, gate) in enumerate(circuit)
        if _isrotation(gate)
            pstr = _paulistringof(TT, _rotationof(gate))
            k = _blockinglayer(pstrs, pstr)
            if k + 1 <= length(layers)
                push!(layers[k+1], ii)
                push!(pstrs[k+1], pstr)
            else
                push!(layers, [ii])
                push!(pstrs, [pstr])
            end
        elseif !isempty(layers) && isempty(pstrs[end])
            push!(layers[end], ii)
        else
            push!(layers, [ii])
            push!(pstrs, TT[])
        end
    end
    return layers
end

# the last layer that a rotation on `pstr` may not move past: a layer of other gates, or one with
# a rotation that does not commute with it; 0 if there is none
function _blockinglayer(pstrs, pstr)
    k = length(pstrs)
    while k >= 1 && !isempty(pstrs[k]) && all(commutes(pstr, other) for other in pstrs[k])
        k -= 1
    end
    return k
end
