### rotationlayers.jl
##
# A file for layers of Pauli rotations that commute with each other.
# For example PauliRotationLayer([:Z, :Z], staircasetopology(4)) holds the rotations RZZ_12, RZZ_23 and RZZ_34.
# A layer is propagated as a whole, which is faster than propagating its rotations one after the other: class by class,
# as PropagationBase applies a layer of commuting rotations (Base/Layers/classes.jl). A Pauli rotation acts on the Pauli
# strings that anticommute with its generator, that is whose Paulis anticommute with those of the generator on an odd
# number of qubits, and flips the bits of the generator on them. Below the rotations of a circuit are gathered into
# layers, a layer is routed and its plan prepared, with the lookup of the rotations that anticommute with a Pauli string
# and the signs of the two Pauli strings that a rotation mixes.
##
###

"""
    PauliRotationLayer(rotations::Vector{PauliRotation})
    PauliRotationLayer(symbols, all_qinds)

Builds a `GateLayer` from a sequence of `PauliRotation` gates. 
They must all mutually commute. 
With `symbols` and `all_qinds`, the layer is built from one `PauliRotation(symbols, qinds)` for all `qinds` in `all_qinds`.
For example PauliRotationLayer(:X, 1:4) or PauliRotationLayer([:Z, :Z], staircasetopology(4)).
Propagating through a layer instead of a sequence of rotations is often faster, especially for large layers.
"""
function PauliRotationLayer(symbols, all_qinds)
    rotations = [PauliRotation(symbols, qinds) for qinds in all_qinds]
    return PauliRotationLayer(rotations)
end

function PauliRotationLayer(rotations)
    rotations = PauliRotation[rotation for rotation in rotations]
    _commutationcheck(rotations)
    return GateLayer(rotations; guaranteed_commutes=true)
end

function _commutationcheck(rotations::Vector{PauliRotation})
    # the generators are compared as integers of the type for the highest qubit that the rotations act on
    nqubits = maximum(rotation -> maximum(rotation.qinds), rotations; init=0)
    _commutationcheck(rotations, getinttype(nqubits))
    return
end

function _commutationcheck(rotations::Vector{PauliRotation}, ::Type{TT}) where {TT}
    generators = [symboltoint(TT, rotation.symbols, rotation.qinds) for rotation in rotations]
    for index1 in eachindex(generators), index2 in index1+1:lastindex(generators)
        if !commutes(generators[index1], generators[index2])
            throw(ArgumentError(
                "The rotations of a `PauliRotationLayer` must commute with each other. " *
                "The rotations $index1 and $index2 of the layer, $(rotations[index1]) and $(rotations[index2]), do not."
            ))
        end
    end
end


### Layers of a circuit

"""
    tolayers(circuit[, params])

Returns the circuit with every run of two or more consecutive Pauli rotations that commute with each other, frozen ones
included, as one `GateLayer`, and its parameters if they are given.
A run ends at any other gate and at a rotation that does not commute with every rotation before it in the run.
The parameter of a layer is the vector of the parameters of its rotations, and a layer of only frozen rotations is frozen
itself, so that it takes no parameter.
The layered circuit propagates like the circuit, as every layer applies its rotations in their order.
"""
function tolayers(circuit, params)
    PropagationBase._checknumberofparams(circuit, params)

    # the generators are compared as integers of the type for the highest qubit of the circuit
    TT = getinttype(maxqind(circuit))

    layered_circuit = Gate[]
    layered_params = []
    # the rotations gathered for the next layer, their generators, and the parameters of those not frozen
    layer = Gate[]
    generators = TT[]
    layer_params = []
    param_index = 0
    for gate in circuit
        if gate isa ParametrizedGate
            param_index += 1
        end

        if gate isa _LayerRotation
            rotation = if gate isa FrozenGate
                gate.gate
            else
                gate
            end
            generator = symboltoint(TT, rotation.symbols, rotation.qinds)
            if !all(layer_generator -> commutes(generator, layer_generator), generators)
                _pushlayer!(layered_circuit, layered_params, layer, generators, layer_params)
            end
            push!(layer, gate)
            push!(generators, generator)
            if gate isa ParametrizedGate
                push!(layer_params, params[param_index])
            end
        else
            _pushlayer!(layered_circuit, layered_params, layer, generators, layer_params)
            push!(layered_circuit, gate)
            if gate isa ParametrizedGate
                push!(layered_params, params[param_index])
            end
        end
    end
    _pushlayer!(layered_circuit, layered_params, layer, generators, layer_params)
    return layered_circuit, layered_params
end

function tolayers(circuit)
    # the layers do not depend on the parameters
    layered_circuit, _ = tolayers(circuit, zeros(countparameters(circuit)))
    return layered_circuit
end

# a gate that a layer applies class by class: a Pauli rotation, frozen or not
const _LayerRotation = Union{PauliRotation,FrozenGate{PauliRotation}}

# Pushes the gathered rotations with their parameters, a single one as it is and several as a layer, which is frozen if
# all of them are. Then nothing is gathered.
function _pushlayer!(layered_circuit, layered_params, layer, generators, layer_params)
    if length(layer) == 1
        push!(layered_circuit, only(layer))
        append!(layered_params, layer_params)
    elseif length(layer) > 1 && isempty(layer_params)
        frozen_layer = GateLayer([gate.gate for gate in layer]; guaranteed_commutes=true)
        push!(layered_circuit, freeze(frozen_layer, [gate.parameter for gate in layer]))
    elseif length(layer) > 1
        push!(layered_circuit, GateLayer(layer; guaranteed_commutes=true))
        push!(layered_params, [param for param in layer_params])
    end
    empty!(layer)
    empty!(generators)
    empty!(layer_params)
    return
end


### Propagation of a layer

# Whether the layer has a fast path on the sum of the cache, on which its gates are applied together, class by class,
# rather than one by one: every gate is a Pauli rotation, frozen or not, and the classes can be rotated in the sum.
_haslayerfastpath(layer::GateLayer, prop_cache::AbstractPropagationCache) =
    all(gate -> gate isa _LayerRotation, layer.gates) && PropagationBase._canrotateclasses(prop_cache)

# the Pauli rotations of a layer and their angles, those of frozen rotations included
function togates(layer::GateLayer, params)
    PropagationBase._checklayerparameters(layer, params)

    rotations = PauliRotation[]
    angles = []
    param_index = 0
    for gate in layer.gates
        if gate isa FrozenGate
            push!(rotations, gate.gate)
            push!(angles, gate.parameter)
        else
            param_index += 1
            push!(rotations, gate)
            push!(angles, PropagationBase._gateparameter(params, param_index))
        end
    end
    return rotations, [angle for angle in angles]
end


### The plan of a layer

# A pair is a Pauli string that anticommutes with a rotation and its partner, the string with the mask flipped. 
# The rotation mixes the two by the cosine and sine of its angle, with opposite signs.
# The lower string of a pair is the one in which the lowest distinguishing bit that the rotation flips is clear.

# A class is a set of Pauli strings that the rotations of a layer only mix among themselves. 
# All its strings anticommute with the same rotations.

# A representative is a value that all strings of a class share and no other string has.
# The distinguishing bits of a class tell its strings apart, so a class holds at most 2^(number of distinguishing bits)
# strings. A pivot bit is a bit that only one mask has after the masks are reduced against each other with XOR. The
# distinguishing bits of a class are the pivot bits that its rotations flip.
# A plan holds what is computed once per layer: the masks, cosines, sines, pivot bits and the lookup.
# The lookup finds the rotations that anticommute with a Pauli string, the rotation `i` as the bit `i - 1` of a tuple of
# words, and the bits that those rotations flip together, from a chunk table of what the rotations anticommuting with each
# Pauli on each qubit contribute.

# The plan of rotating the classes of Pauli strings of the type `TT` on `nqubits` qubits by the rotations `rotations`
# with the `angles`, where coefficients below `min_abs_coeff` are truncated.
function _prepareclasses(rotations::Vector{PauliRotation}, angles, ::Type{TT}, nqubits::Int;
    min_abs_coeff::Real=0) where {TT}

    # every rotation acts on qubits of the Pauli strings
    for rotation in rotations
        _check_qind_range(nqubits, rotation.qinds)
    end

    # integer representations of the PauliRotation generators
    gate_masks = [symboltoint(TT, rotation.symbols, rotation.qinds) for rotation in rotations]

    # a Pauli string anticommutes with a rotation if it shares an odd number of bits with the swapped generator, qubit by qubit
    lookup = PropagationBase.PrecomputedLookup(_commutationmask.(gate_masks), gate_masks)

    # the plan adds the cosines and sines of the angles and the pivot bits
    return PropagationBase.ClassPlan(gate_masks, angles, lookup, _pairsigns; min_abs_coeff)
end

# the generator of a rotation with its two bits swapped on every qubit, with which a Pauli string shares an odd number of
# bits if it anticommutes with the generator
function _commutationmask(gate_mask::TT) where {TT}
    low_bits = alternatingmask(gate_mask)
    return ((gate_mask >> 1) & low_bits) | ((gate_mask & low_bits) << 1)
end

# The signs that a rotation with the gate mask `gate_mask` gives the two Pauli strings of a pair: that of the partner it
# makes of the lower string, as `paulirotationproduct` gives it, and the opposite sign for the lower string made of the
# partner.
@inline function _pairsigns(gate_mask, lower_pstr)
    _, sign = paulirotationproduct(gate_mask, lower_pstr)
    return sign, -sign
end


