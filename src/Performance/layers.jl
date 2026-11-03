###
##
# Layers of commuting Pauli rotations, gathered by `torotationlayers` inside `propagate`. A layer costs a pass over the
# sum, so a gathered layer with too few rotations for the type of sum is split back into its rotations. Layers already in
# the circuit stay as they are.
##
###

function _torotationlayers(circuit, params, thing)
    circuit, params = PropagationBase._promotecircandparams(circuit, params)
    given_layers = IdSet{Gate}(gate for gate in circuit if gate isa GateLayer)
    layered_circuit, layered_params = torotationlayers(circuit, params)

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

# Whether a gathered layer of Pauli rotations is faster than its rotations one by one on `thing`.
# One by one, every rotation costs a pass over the sum, and that pass costs more for wider Pauli strings than the lookup
# of the acting rotations does in a layer, so the number of rotations needed does not grow with the number of qubits.
# On a `PauliSum`, the rotations one by one are faster than on arrays, so a layer needs more of them.
function _payslayer(layer::GateLayer, thing)
    if thing isa PauliString || StorageType(thing) isa PropagationBase.DictStorage
        return length(layer.gates) >= 32
    else
        return length(layer.gates) >= 16
    end
end
