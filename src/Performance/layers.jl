###
##
# Layers of commuting gates, gathered by `tolayers` inside `propagate`. A layer costs a pass over the sum, so a gathered
# layer with too few gates for the type of sum is split back into its gates. Layers already in the circuit stay as they are.
##
###

function _tolayers(circuit, params, thing)
    circuit, params = PropagationBase._promotecircandparams(circuit, params)
    given_layers = IdSet{Gate}(gate for gate in circuit if gate isa GateLayer)
    layered_circuit, layered_params = tolayers(circuit, params)

    new_circuit = Gate[]
    new_params = []
    param_index = 0
    for gate in layered_circuit
        gate_params = if isparametrized(gate)
            param_index += 1
            layered_params[param_index]
        else
            Float64[]
        end

        if gate isa GateLayer && !(gate in given_layers) && !_payslayer(gate, thing)
            gates, gates_params = PropagationBase.togates(gate, gate_params)
            append!(new_circuit, gates)
            append!(new_params, gates_params)
        else
            push!(new_circuit, gate)
            if isparametrized(gate)
                push!(new_params, gate_params)
            end
        end
    end
    return new_circuit, [param for param in new_params]
end

# Whether a gathered layer is faster than its gates one by one on `thing`.
# One by one, a gate that changes Pauli strings looks each up in a dictionary, so layers pay much sooner on a `PauliSum`,
# except for Pauli noise, which only scales coefficients.
# On arrays, finding the rotations that act on a Pauli string costs more on more qubits, and other gates are applied to a
# Pauli string one after the other, which pays only if each acts on a small part of the Pauli strings, here for a typical
# weight of 8. Amplitude damping, and Pauli noise in the parallel zones of a multi sum, are faster one by one.
function _payslayer(layer::GateLayer, thing)
    gates = [_unfrozen(gate) for gate in layer.gates]
    n_gates = length(gates)
    nq = nqubits(thing)
    storage = thing isa PauliString ? PropagationBase.DictStorage() : StorageType(thing)
    rotations = PauliPropagation._isrotation(first(gates))
    noise = all(gate -> gate isa PauliNoise, gates)

    if !rotations && !all(PropagationBase._appliesbyclass, gates)
        # a gate that normalizes the whole sum, such as an imaginary-time rotation, makes the layer apply its gates one by one
        return false
    elseif storage isa PropagationBase.DictStorage
        min_gates = if rotations
            max(16, nq ÷ 4)
        elseif noise
            max(32, nq ÷ 2)
        else
            16
        end
        return n_gates >= min_gates
    elseif rotations
        return n_gates >= max(32, 3 * nq ÷ 4)
    elseif any(gate -> gate isa AmplitudeDampingNoise, gates) || (noise && storage isa PropagationBase.MultiSumStorage && nzones(thing) > 1)
        return false
    else
        return n_gates >= 16 && 3 * 8 * maximum(gate -> length(qinds(gate)), gates) <= nq
    end
end

_unfrozen(gate) = gate
_unfrozen(gate::FrozenGate) = gate.gate
