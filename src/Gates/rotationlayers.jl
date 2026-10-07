### rotationlayers.jl
##
# A file for layers of Pauli rotations that commute with each other.
# For example PauliRotationLayer([:Z, :Z], staircasetopology(4)) holds the rotations RZZ_12, RZZ_23 and RZZ_34.
# A layer is propagated as a whole, which is faster than propagating its rotations one after the other: class by class,
# as PropagationBase applies a layer of commuting rotations (Base/Layers/classes.jl). A Pauli rotation acts on the Pauli
# strings that anticommute with its generator, that is whose Paulis anticommute with those of the generator on an odd
# number of qubits, and flips the bits of the generator on them. Below a layer is routed and its plan prepared, followed
# by the reader of the rotations that anticommute with a Pauli string and the bits they flip, and the signs of the two
# Pauli strings that a rotation mixes.
##
###

"""
    PauliRotationLayer(rotations)
    PauliRotationLayer(symbols, qinds)

Returns a `GateLayer` of the Pauli rotations `rotations`.
With `symbols` and `qinds`, the layer holds one `PauliRotation(symbols, qinds[i])` for every entry of `qinds`.
For example PauliRotationLayer(:X, 1:4) or PauliRotationLayer([:Z, :Z], staircasetopology(4)).
The rotations act on any qubits and in any order, and need to commute with each other.
The parameter of the layer is a vector with one angle per rotation, or a number for a layer of one rotation.
In the Schrödinger picture, the rotations are applied in the order of `rotations` or `qinds`, with truncation after each.
"""
function PauliRotationLayer(symbols, qinds)
    rotations = [PauliRotation(symbols, rotation_qinds) for rotation_qinds in qinds]
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
                "The rotations on the qubits $(rotations[index1].qinds) and $(rotations[index2].qinds) do not."
            ))
        end
    end
end


### Propagation of a layer

# Whether the layer has a fast path on the sum of the cache, on which its gates are applied together, class by class,
# rather than one by one: every gate has it, and the classes can be rotated in the sum.
_haslayerfastpath(layer::GateLayer, prop_cache::AbstractPropagationCache) =
    all(_haslayerfastpath, layer.gates) && PropagationBase._canrotateclasses(prop_cache)

# whether a gate is applied on the fast path of a layer: a Pauli rotation, frozen or not
_haslayerfastpath(gate::Gate) = false
_haslayerfastpath(gate::PauliRotation) = true
_haslayerfastpath(gate::FrozenGate) = _haslayerfastpath(gate.gate)

# the Pauli rotations of a layer and their angles, those of frozen rotations included
function _rotationsandangles(layer::GateLayer, params)
    rotations = PauliRotation[]
    angles = Any[]
    function addrotation!(rotation::PauliRotation, angle)
        push!(rotations, rotation)
        push!(angles, angle)
        return
    end
    addrotation!(frozen_gate::FrozenGate) = addrotation!(frozen_gate.gate, frozen_gate.parameter)
    PropagationBase._foreachgate(addrotation!, layer, params)
    return rotations, [angle for angle in angles]
end


### The plan of a layer

# The plan of rotating the classes of Pauli strings of the type `TT` with coefficients of the type `CT` on `nqubits`
# qubits by the rotations `rotations` with the `angles`, where coefficients below `min_abs_coeff` are truncated.
function _prepareclasses(rotations::Vector{PauliRotation}, angles, ::Type{TT}, ::Type{CT}, nqubits::Int;
    min_abs_coeff::Real=0) where {TT,CT}

    for rotation in rotations
        _check_qind_range(nqubits, rotation.qinds)
    end
    masks = [symboltoint(TT, rotation.symbols, rotation.qinds) for rotation in rotations]
    signs = _rotationsigns(rotations, masks, _signtype(CT))
    reader = _classreader(rotations, masks, TT, nqubits)
    return PropagationBase.ClassPlan(masks, angles, signs, reader; min_abs_coeff)
end


### Readers: the rotations that anticommute with a string, and the bits they flip

# A layer whose rotations fall into more groups than this is read through byte tables instead.
const _MAX_ROTATION_GROUPS = 4

# whether the rotations are read in groups, from the whole Pauli string at once
_readsingroups(groups) = length(groups) <= _MAX_ROTATION_GROUPS

# The rotations whose qubits are the same distance apart with the same Paulis on them: the low bit of the lower qubit of
# every rotation, the distance from the lower to the upper qubit in bits (0 for rotations on one qubit), the Paulis of the
# generator on the lower and on the upper qubit, and for every qubit the index of the rotation of which it is the lower qubit.
struct _RotationGroup{TT}
    lower_mask::TT
    shift::Int
    lower_pauli::UInt8
    upper_pauli::UInt8
    rotation_at_lower::Vector{Int32}
end

# Finds the rotations that anticommute with a string, as the bits of their positions in a tuple of `W` words, and the
# bits that those rotations flip. A layer whose rotations act on one or two qubits each and fall into few groups is read
# group by group from the whole string at once. Any other layer is read through the byte tables of PropagationBase, from
# the generators with their two bits swapped on every qubit: a string anticommutes with a rotation if the bits where the
# string and the swapped generator are both set are odd in number.
struct _ClassReader{TT,W}
    reads_in_groups::Bool
    groups::Vector{_RotationGroup{TT}}

    # the byte tables, with the bits that every rotation flips
    tables::PropagationBase.TableReader{TT,W}
end

function _classreader(rotations::Vector{PauliRotation}, masks::Vector{TT}, ::Type{TT}, nqubits::Int) where {TT}
    n_words = cld(length(masks), 64)
    group_ids = Dict{Tuple{Int,UInt8,UInt8},Int}()
    groups = _RotationGroup{TT}[]
    shared_lower_qubit = false
    on_more_qubits = false

    for rotation in eachindex(rotations)
        qinds = rotations[rotation].qinds
        symbols = rotations[rotation].symbols
        if length(qinds) > 2
            on_more_qubits = true
            continue
        end
        first_qind, second_qind = qinds[1], get(qinds, 2, 0)
        first_pauli = UInt8(symboltoint(symbols[1]))
        second_pauli = UInt8(symboltoint(get(symbols, 2, :I)))

        shift, lower_pauli, upper_pauli, lower_qind = if second_qind == 0
            0, first_pauli, 0x00, first_qind
        elseif first_qind < second_qind
            2 * (second_qind - first_qind), first_pauli, second_pauli, first_qind
        else
            2 * (first_qind - second_qind), second_pauli, first_pauli, second_qind
        end

        group_id = get(group_ids, (shift, lower_pauli, upper_pauli), 0)
        if group_id == 0
            push!(groups, _RotationGroup{TT}(zero(TT), shift, lower_pauli, upper_pauli, zeros(Int32, nqubits)))
            group_id = length(groups)
            group_ids[(shift, lower_pauli, upper_pauli)] = group_id
        end
        group = groups[group_id]

        # a rotation that comes twice cannot be found through its lower qubit
        shared_lower_qubit |= group.rotation_at_lower[lower_qind] != 0
        group.rotation_at_lower[lower_qind] = rotation
        groups[group_id] = _RotationGroup{TT}(group.lower_mask | symboltoint(TT, :X, lower_qind), shift, lower_pauli, upper_pauli,
            group.rotation_at_lower)
    end

    reads_in_groups = !on_more_qubits && _readsingroups(groups) && !shared_lower_qubit

    tables = if reads_in_groups
        PropagationBase.TableReader{TT,n_words}(NTuple{n_words,UInt64}[], TT[])
    else
        PropagationBase.TableReader(_commutationmask.(masks), masks)
    end
    return _ClassReader{TT,n_words}(reads_in_groups, groups, tables)
end

# the generator of a rotation with its two bits swapped on every qubit, with which a Pauli string shares an odd number of
# bits if it anticommutes with the generator
function _commutationmask(mask::TT) where {TT}
    low_bits = alternatingmask(mask)
    return ((mask >> 1) & low_bits) | ((mask & low_bits) << 1)
end

# the positions of the rotations that anticommute with `pstr`, and the bits that those rotations flip
@inline function PropagationBase._branchinggates(reader::_ClassReader{TT,W}, pstr::TT) where {TT,W}
    if !reader.reads_in_groups
        return PropagationBase._branchinggates(reader.tables, pstr)
    end

    low, high = _qubitbits(pstr)
    positions = ntuple(_ -> zero(UInt64), Val(W))
    flipped = zero(TT)
    for group in reader.groups
        anticommuting = _groupanticommuting(group, low, high)
        flipped |= _groupflips(group, anticommuting)
        limbs = PropagationBase._limbs(anticommuting)
        for limb_index in eachindex(limbs)
            limb = limbs[limb_index]
            while limb != 0
                qind = 32 * (limb_index - 1) + (trailing_zeros(limb) >> 1) + 1
                limb &= limb - one(UInt64)
                positions = PropagationBase._withposition(positions, Int(group.rotation_at_lower[qind]) - 1)
            end
        end
    end
    return positions, flipped
end

# the bits that the rotations anticommuting with `pstr` flip, read group by group without their positions
@inline function PropagationBase._flippedbits(reader::_ClassReader{TT}, pstr::TT) where {TT}
    if !reader.reads_in_groups
        return PropagationBase._flippedbits(reader.tables, pstr)
    end

    low, high = _qubitbits(pstr)
    flipped = zero(TT)
    for group in reader.groups
        flipped |= _groupflips(group, _groupanticommuting(group, low, high))
    end
    return flipped
end

# the bits that the rotations of the group flip whose lower qubits have their low bits in `anticommuting`
@inline function _groupflips(group::_RotationGroup{TT}, anticommuting::TT) where {TT}
    flipped = _paulibits(group.lower_pauli, anticommuting)
    if group.shift != 0
        flipped |= _paulibits(group.upper_pauli, _shiftup(anticommuting, group.shift))
    end
    return flipped
end

# the bits of the Pauli `pauli` on every qubit whose low bit is in `low_bits`
@inline function _paulibits(pauli::UInt8, low_bits::TT) where {TT}
    low = ifelse(pauli & 0x01 != 0x00, low_bits, zero(TT))
    high = ifelse(pauli & 0x02 != 0x00, _shiftup(low_bits, 1), zero(TT))
    return low | high
end

# the low and the high bit of every qubit of `pstr`, both at the low bit
@inline function _qubitbits(pstr)
    low_bits = alternatingmask(pstr)
    return pstr & low_bits, _shiftdown(pstr, 1) & low_bits
end

# The low bit of every qubit on which the Pauli string of the bits `low` and `high` anticommutes with the Pauli `pauli`.
# Two Paulis anticommute where the low bit of one and the high bit of the other are set an odd number of times.
@inline function _anticommutingqubits(pauli::UInt8, low::TT, high::TT) where {TT}
    return ifelse(pauli & 0x02 != 0x00, low, zero(TT)) ⊻ ifelse(pauli & 0x01 != 0x00, high, zero(TT))
end

# the low bit of the lower qubit of every rotation of the group that anticommutes with the Pauli string of the bits
# `low` and `high`
@inline function _groupanticommuting(group::_RotationGroup, low, high)
    lower = _anticommutingqubits(group.lower_pauli, low, high)
    if group.shift == 0
        return lower & group.lower_mask
    end
    # a rotation on two qubits anticommutes if the Pauli string anticommutes with it on exactly one of them
    upper = _anticommutingqubits(group.upper_pauli, low, high)
    return (lower ⊻ _shiftdown(upper, group.shift)) & group.lower_mask
end

# Shifts of a whole Pauli string. Below 64 bits, a shift only moves bits between neighbouring limbs.
# The shifts are never negative, so an unsigned shift needs no code for the other direction.
@inline _shiftdown(pstr, shift::Int) = pstr >> (shift % UInt)
@inline _shiftup(pstr, shift::Int) = pstr << (shift % UInt)

@inline function _shiftdown(pstr::NTupleInteger{N}, shift::Int) where {N}
    if !(0 <= shift < 64)
        return pstr >> shift
    end
    limbs = pstr.limbs
    shifted_limb(i) = (limbs[i] >> shift) | ifelse(i < N, limbs[min(i + 1, N)] << (64 - shift), zero(UInt64))
    return NTupleInteger{N}(ntuple(shifted_limb, Val(N)))
end

@inline function _shiftup(pstr::NTupleInteger{N}, shift::Int) where {N}
    if !(0 <= shift < 64)
        return pstr << shift
    end
    limbs = pstr.limbs
    shifted_limb(i) = (limbs[i] << shift) | ifelse(i > 1, limbs[max(i - 1, 1)] >> (64 - shift), zero(UInt64))
    return NTupleInteger{N}(ntuple(shifted_limb, Val(N)))
end


### Signs

# The type that the signs take for coefficients of the type `CT`: the real type of floating-point coefficients, which
# they multiply as the integers do, so that they are not converted for every entry, and integers otherwise.
_signtype(::Type{CT}) where {CT<:Union{AbstractFloat,Complex{<:AbstractFloat}}} = real(CT)
_signtype(::Type) = Int8

# The signs of the pairs that a rotation on more than one qubit mixes, read from the lower Pauli string of a pair: the
# sign that the rotation gives the partner it makes of that string, as `paulirotationproduct` gives it, and the opposite
# sign, which it gives that string as made of the partner. The sign is read limb by limb for strings of any width.
struct _ProductSigns{N,ST}
    generator::NTupleInteger{N}
end

@inline function PropagationBase._pairsigns(product_signs::_ProductSigns{N,ST}, lower_pstr) where {N,ST}
    exponent = _calculatesignexponent(product_signs.generator, NTupleInteger{N}(PropagationBase._limbs(lower_pstr)))
    sign = (exponent & 2) - 1
    return (ST(sign), ST(-sign))
end

# The signs of every rotation, which `PropagationBase._pairsigns` reads: those of its products with the lower Pauli
# strings of its pairs, and for a rotation on one qubit the same two for every pair, since it leaves every Pauli string
# of its class with one of two Paulis there, the lower entry of a pair with the one whose key bit is clear. Every rotation
# acting on that qubit acts with the same Pauli, so the pivot of the echelon basis there is the lowest bit of that Pauli:
# the low bit of X and Z and the high bit of Y.
function _rotationsigns(rotations::Vector{PauliRotation}, masks::Vector{TT}, ::Type{ST}) where {TT,ST}
    n_limbs = length(PropagationBase._limbs(zero(TT)))
    signs = Vector{Union{Tuple{ST,ST},_ProductSigns{n_limbs,ST}}}(undef, length(rotations))
    for rotation in eachindex(rotations)
        signs[rotation] = if length(rotations[rotation].qinds) == 1
            pauli = UInt8(symboltoint(only(rotations[rotation].symbols)))
            lower_pauli = pauli == 0x02 ? 0x01 : 0x02
            _, sign = paulirotationproduct(pauli, lower_pauli)
            (ST(sign), ST(-sign))
        else
            _ProductSigns{n_limbs,ST}(NTupleInteger{n_limbs}(PropagationBase._limbs(masks[rotation])))
        end
    end
    return signs
end
