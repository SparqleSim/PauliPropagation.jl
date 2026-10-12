### rotationlayers.jl
##
# A file for layers of Pauli rotations that commute with each other.
# For example PauliRotationLayer([:Z, :Z], staircasetopology(4)) holds the rotations RZZ_12, RZZ_23 and RZZ_34.
# A layer is propagated as a whole, which is faster than propagating its rotations one after the other: class by class,
# as PropagationBase applies a layer of commuting gates (Base/Layers/classes.jl). A Pauli rotation acts on the Pauli
# strings that anticommute with its generator, that is whose Paulis anticommute with those of the generator on an odd
# number of qubits, and flips the bits of the generator on them. Below a layer is routed and its plan prepared, with the
# lookup of the rotations that anticommute with a Pauli string and the rule of each rotation on the strings it acts on.
# Which gates of a circuit are gathered into a layer is decided in gatelayers.jl.
##
###

"""
    PauliRotationLayer(symbols, all_qinds)
    PauliRotationLayer(rotations::Vector{PauliRotation})

A `GateLayer` of Pauli rotations that commute with each other.
With `symbols` and `all_qinds`, the layer holds one `PauliRotation(symbols, qinds)` for each `qinds` in `all_qinds`.
For example PauliRotationLayer(:X, 1:4) or PauliRotationLayer([:Z, :Z], staircasetopology(4)).
Rotations passed as a vector may act on the same qubits, but an error is thrown if two of them do not commute.
The parameter of the layer is a vector with one angle per rotation, in the order of the rotations.
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


### Propagation of a layer

# Whether the layer is applied as rotations, class by class, rather than its gates one by one: every gate is a Pauli
# rotation, frozen or not, and the classes can be applied to in the sum.
_hasrotationpath(layer::GateLayer, prop_cache::AbstractPropagationCache) =
    all(_isrotation, layer.gates) && PropagationBase._canapplylayer(prop_cache)

# Applies the rotations of the layer in one pass over the sum, class by class, truncating after every rotation.
function _applyrotationlayer!(layer::GateLayer, prop_cache::AbstractPropagationCache, params;
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, min_rel_coeff=nothing,
    customtruncfunc=nothing, thread::Bool=true, kwargs...)

    truncfunc = buildtruncfunc(prop_cache; min_abs_coeff, max_weight, max_freq, max_sins, min_rel_coeff, customtruncfunc, thread)
    rotations, angles = _rotationsandangles(layer, params)
    plan = _preparelayer(rotations, angles, prop_cache)
    PropagationBase._applypass!(prop_cache, plan, truncfunc; thread)
    return prop_cache
end

# The Pauli rotations of a layer and their angles, those of rotations without a parameter included. Every gate of the
# layer must be a Pauli rotation, as `_hasrotationpath` checks before a layer is applied as rotations.
function _rotationsandangles(layer::GateLayer, params)
    rotations = PauliRotation[]
    angles = []
    function pushrotation!(gate, angle)
        push!(rotations, _rotationof(gate))
        push!(angles, angle)
        return
    end
    pushrotation!(gate) = pushrotation!(gate, _angleof(gate))
    PropagationBase._foreachgate(pushrotation!, layer, params)
    return rotations, [angle for angle in angles]
end


### The plan of a layer

# The plan of applying the rotations `rotations` with the `angles` to the classes of the Pauli strings of the cache: the
# generators of the rotations as integers, which the rotations flip, the rules of the rotations, and the lookup of the
# rotations acting on a Pauli string.
function _preparelayer(rotations::Vector{PauliRotation}, angles, prop_cache)

    # every rotation acts on qubits of the Pauli strings
    for rotation in rotations
        _check_qind_range(nqubits(prop_cache), rotation.qinds)
    end

    # integer representations of the PauliRotation generators
    gate_masks = [_branchmask(rotation, prop_cache) for rotation in rotations]

    # a rotation acts on all Pauli strings of a class it is applied to, so its rule need not ask whether it does
    rules = [onlimbs(ActingRotationRule, gate_mask, cos(angle), sin(angle)) for (gate_mask, angle) in zip(gate_masks, angles)]

    # a Pauli string anticommutes with a rotation if it shares an odd number of bits with the swapped generator, qubit by qubit
    lookup = PropagationBase.PrecomputedLookup(_commutationmask.(gate_masks))
    return PropagationBase.LayerPlan(gate_masks, rules, lookup)
end

# the generator of a rotation with its two bits swapped on every qubit, with which a Pauli string shares an odd number of
# bits if it anticommutes with the generator
function _commutationmask(gate_mask::TT) where {TT}
    low_bits = alternatingmask(gate_mask)
    return ((gate_mask >> 1) & low_bits) | ((gate_mask & low_bits) << 1)
end

# The rule of a Pauli rotation on the Pauli strings it acts on: it keeps `cos_val` of a coefficient and creates `sin_val`
# of it with the sign that the rotation gives the partner. For a rotation on at most two qubits, that sign depends on the
# two Paulis of the Pauli string on those qubits alone, so the bit of `signs` for those two Paulis, read from the lowest
# bits `bit1` and `bit2` of the qubits, is set where the sign is -1.
struct ActingRotationRule{TT,RT}
    gate_mask::TT
    cos_val::RT
    sin_val::RT
    tabulated::Bool
    bit1::Int
    bit2::Int
    signs::UInt16
end

function ActingRotationRule(gate_mask::TT, cos_val, sin_val) where {TT}
    # the lowest bit of every qubit that the generator acts on
    qubit_bits = (gate_mask | (gate_mask >> 1)) & alternatingmask(gate_mask)
    if count_ones(qubit_bits) > 2
        return ActingRotationRule(gate_mask, cos_val, sin_val, false, 0, 0, zero(UInt16))
    end
    # a generator that is the identity acts on no Pauli string, and its table is not read
    bit1, bit2 = if iszero(qubit_bits)
        0, 0
    else
        trailing_zeros(qubit_bits), 8 * sizeof(TT) - 1 - leading_zeros(qubit_bits)
    end
    signs = zero(UInt16)
    for index in 0:15
        pstr = _withpauli(_withpauli(zero(TT), bit1, index & 3), bit2, ifelse(bit2 == bit1, 0, index >> 2))
        if !commutes(gate_mask, pstr) && last(paulirotationproduct(gate_mask, pstr)) == -1
            signs |= one(UInt16) << index
        end
    end
    return ActingRotationRule(gate_mask, cos_val, sin_val, true, bit1, bit2, signs)
end

# `pstr` with the two bits from `bit` on set to `pauli`
_withpauli(pstr::TT, bit::Int, pauli::Int) where {TT} =
    pstr | ifelse(isodd(pauli), one(TT) << bit, zero(TT)) | ifelse(pauli >= 2, one(TT) << (bit + 1), zero(TT))

# the sign that the rotation gives the partner of a Pauli string it acts on
@inline function _partnersign(rule::ActingRotationRule, pstr)
    if rule.tabulated
        pauli1 = (PropagationBase._wordat(pstr, rule.bit1) >> (rule.bit1 & 63)) & 3
        pauli2 = (PropagationBase._wordat(pstr, rule.bit2) >> (rule.bit2 & 63)) & 3
        return 1 - 2 * Int((rule.signs >> (pauli1 | (pauli2 << 2))) & 1)
    end
    return last(paulirotationproduct(rule.gate_mask, pstr))
end

_partnersign(rule::PropagationBase.OnLimbs, pstr) = _partnersign(rule.rule, PropagationBase.limbwindow(pstr, rule.inds))

@inline (rule::ActingRotationRule)(pstr, coeff) = Branch(coeff * rule.cos_val, coeff * rule.sin_val * _partnersign(rule, pstr))

# the partner of a Pauli string has the opposite sign, so the two need only one
@inline function PropagationBase.branchpair(rule::ActingRotationRule, pstr, coeff, partner, partner_coeff)
    sign = _partnersign(rule, pstr)
    new_coeff = mergefunc(coeff * rule.cos_val, partner_coeff * rule.sin_val * -sign)
    new_partner_coeff = mergefunc(partner_coeff * rule.cos_val, coeff * rule.sin_val * sign)
    return new_coeff, new_partner_coeff, true
end

# A coefficient creates `sin_val` of itself, so one below `cutoff / abs(sin_val)` creates nothing that the cutoff keeps;
# the margin covers the rounding of the quotient and of the product.
PropagationBase._creationfloor(rule::ActingRotationRule, cutoff) = (1 - 1e-12) * cutoff / abs(rule.sin_val)
