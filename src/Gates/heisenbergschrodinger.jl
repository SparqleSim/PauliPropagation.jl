###
##
# A file to manage conversion of our gates to Heisenberg or Schrödinger picture propagation.
##
###

"""
    toheisenberg(circuit[, params])

Convert a circuit and its parameters to the Heisenberg picture.
Calls `_toheisenberg(gate [, param])` for each gate in the circuit
with an optional parameter if the gate is parametrized.
By default, gates are assumed be defined in the Heisenberg picture already,
but the circuit order is in Schrödinger picture.
Some gates may currently be disallowed in Heisenberg picture.
"""
function toheisenberg(circuit, params)
    # by default, gates are assumed be defined in the Heisenberg picture already
    # the circuit order is in Schrödinger though.
    # some gates may be disallowed in Heisenberg picture

    PropagationBase._checknumberofparams(circuit, params)

    circuit = reverse(circuit)
    params = reverse(params)

    params_iterator = Iterators.Stateful(params)
    heisenberg_circuit = eltype(circuit)[]
    heisenberg_params = eltype(params)[]

    for gate in circuit
        if gate isa ParametrizedGate
            param = popfirst!(params_iterator)

            gate_heisenberg, param_heisenberg = _toheisenberg(gate, param)

            push!(heisenberg_circuit, gate_heisenberg)
            push!(heisenberg_params, param_heisenberg)
        else
            gate_heisenberg = _toheisenberg(gate)

            push!(heisenberg_circuit, gate_heisenberg)
        end
    end
    return heisenberg_circuit, heisenberg_params
end

function toheisenberg(circuit)
    # this must be a non-parametrized circuit
    circuit, _ = PropagationBase._promotecircandparams(circuit, nothing)
    circuit, _ = toheisenberg(circuit, Float64[])
    return circuit
end

toheisenberg(gate::Gate) = only(toheisenberg([gate]))
function toheisenberg(gate::Gate, param)
    circ, params = toheisenberg([gate], [param])
    return only(circ), only(params)
end

# We already assume gates are defined in Heisenberg picture
function _toheisenberg(gate::StaticGate)
    return gate
end

# We already assume gates are defined in Heisenberg picture 
function _toheisenberg(gate::ParametrizedGate, param)
    return gate, param
end

function _toheisenberg(gate::FrozenGate)
    gate_heisenberg, param_heisenberg = _toheisenberg(gate.gate, gate.parameter)
    return freeze(gate_heisenberg, param_heisenberg)
end

# ImaginaryPauliRotation are currently actively disallowed in Heisenberg picture
# This has to do because we don't know how to handle the normalization
function _toheisenberg(gate::ImaginaryPauliRotation, τ)
    throw(error("ImaginaryPauliRotation gates are currently not defined in the Heisenberg picture."))
end


"""
    toschrodinger(circuit[, params])

Convert a circuit and its parameters to the Schrödinger picture.
Calls `_toschrodinger(gate [, param])` for each gate in the circuit
with an optional parameter if the gate is parametrized.
This function needs to be overloaded for custom gates.
Returns the converted circuit and parameters.
"""
function toschrodinger(circuit, params)

    PropagationBase._checknumberofparams(circuit, params)

    params_iterator = Iterators.Stateful(params)
    schrodinger_circuit = eltype(circuit)[]
    schrodinger_params = eltype(params)[]

    for gate in circuit
        if gate isa ParametrizedGate
            param = popfirst!(params_iterator)

            gate_schrodinger, param_schrodinger = _toschrodinger(gate, param)

            push!(schrodinger_circuit, gate_schrodinger)
            push!(schrodinger_params, param_schrodinger)
        else
            gate_schrodinger = _toschrodinger(gate)

            push!(schrodinger_circuit, gate_schrodinger)
        end
    end
    return schrodinger_circuit, schrodinger_params
end

function toschrodinger(circuit)
    # this must be a non-parametrized circuit
    circuit, _ = PropagationBase._promotecircandparams(circuit, nothing)
    circuit, _ = toschrodinger(circuit, Float64[])
    return circuit
end

toschrodinger(gate::Gate) = only(toschrodinger([gate]))
function toschrodinger(gate::Gate, param)
    circ, params = toschrodinger([gate], [param])
    return only(circ), only(params)
end


function _toschrodinger(gate::G, args...) where G
    throw(error("Unkown how to define gate of type $G in the Schrodinger picture. 
    Please implement `toschrodinger(gate::G [, param])` for this gate type."))
end


# Method to transpose a `PauliRotation` gate for Schrödinger picture propagation.
# This inverts the sign of `θ`, which can be seen by the conjugation of exp(-i*θ*P/2).
function _toschrodinger(gate::PauliRotation, θ)
    return gate, -θ
end


# Method to transpose a `ImaginaryPauliRotation` gate for Schrödinger picture propagation.
# This does not change the parameter because exp(-τ*P/2) is the same under conjugation.
function _toschrodinger(gate::ImaginaryPauliRotation, τ)
    return gate, τ
end


# Method to transpose a `CliffordGate` for Schrödinger picture propagation.
# If not already registered, the transposed Clifford map is created via `transposecliffordmap()`
# and stored in the global `clifford_map`. 
# This Clifford gate is called `:(old_symbol)_transpose`, where `old_symbol` is the symbol of the original Clifford gate.
function _toschrodinger(gate::CliffordGate)
    transposed_symbol = Symbol(gate.symbol, :_transpose)

    if haskey(clifford_map, transposed_symbol)
        return gate
    end

    # register the transpose 
    lookup_map = clifford_map[gate.symbol]
    transposed_map = transposecliffordmap(lookup_map)
    clifford_map[transposed_symbol] = transposed_map

    return CliffordGate(transposed_symbol, gate.qinds)
end


# Method to transpose a `PauliNoise` gate for Schrödinger picture propagation.
# PauliNoise gates are symmetric under transposition and are thus unaffected.
function _toschrodinger(gate::PauliNoise, p)
    return gate, p
end


function _toschrodinger(frozen_gate::FrozenGate)
    gate, param = frozen_gate.gate, frozen_gate.parameter
    gate_schrodinger, param_schrodinger = _toschrodinger(gate, param)
    return freeze(gate_schrodinger, param_schrodinger)
end