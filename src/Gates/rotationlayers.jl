### rotationlayers.jl
##
# A file for layers of Pauli rotations that commute with each other.
# For example PauliRotationLayer([:Z, :Z], staircasetopology(4)) holds the rotations RZZ_12, RZZ_23 and RZZ_34.
# A layer is propagated as a whole, which is faster than propagating its rotations one after the other: class by class,
# as PropagationBase applies a layer of commuting rotations (Base/Layers/classes.jl). A Pauli rotation acts on the Pauli
# strings that anticommute with its generator, that is whose Paulis anticommute with those of the generator on an odd
# number of qubits, and flips the bits of the generator on them. The Pauli strings that anticommute with the same
# rotations of the layer, and agree on what those rotations leave unchanged, form a class. Below a layer is routed, split
# into passes and the plan of a pass prepared, followed by how the classes are found: the reader of the rotations that
# anticommute with a Pauli string, the key of a class and its key bits, and the signs of the two Pauli strings that a
# rotation mixes.
##
###

"""
    PauliRotationLayer(rotations)
    PauliRotationLayer(symbols, qinds)

Returns a `GateLayer` of the Pauli rotations `rotations`.
With `symbols` and `qinds`, the layer holds one `PauliRotation(symbols, qinds[i])` for every entry of `qinds`.
For example PauliRotationLayer(:X, 1:4) or PauliRotationLayer([:Z, :Z], staircasetopology(4)).
The rotations act on one or two qubits each, on any qubits and in any order, and need to commute with each other.
The parameter of the layer is a vector with one angle per rotation, or a number for a layer of one rotation.
In the Schrödinger picture, the rotations are applied in the order of `rotations` or `qinds`, with truncation after each.
"""
function PauliRotationLayer(symbols, qinds)
    rotations = [PauliRotation(symbols, rotation_qinds) for rotation_qinds in qinds]
    return PauliRotationLayer(rotations)
end

function PauliRotationLayer(rotations)
    rotations = PauliRotation[rotation for rotation in rotations]

    for rotation in rotations
        _rotationlayerweightcheck(rotation)
    end

    _commutationcheck(rotations)

    return GateLayer(rotations; guaranteed_commutes=true)
end

function _rotationlayerweightcheck(rotation::PauliRotation)
    if !(1 <= length(rotation.qinds) <= 2)
        throw(ArgumentError("`PauliRotationLayer` is defined for rotations on one or two qubits. Got one on the qubits $(rotation.qinds)."))
    end
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

# whether a gate is applied on the fast path of a layer: a Pauli rotation on one or two qubits, frozen or not
_haslayerfastpath(gate::Gate) = false
_haslayerfastpath(gate::PauliRotation) = length(gate.qinds) <= 2
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


### The passes of a layer

# The rotations of the layer that each pass rotates class by class, as runs of the layer's order in which no qubit is
# acted on with two different Paulis: all of them in one pass for most layers. The key of a class keeps, on every qubit
# that its rotations act on with one Pauli, whether the string anticommutes with that Pauli there, and so tells apart
# strings that anticommute with different rotations. A qubit acted on with two Paulis keeps neither, so strings with
# different rotations could share a key there. In a layer of commuting rotations that happens for rotations on the same
# two qubits, as XZ on (1, 2) and on (2, 1), or XX and ZZ on (1, 2).
function _classpasses(rotations::Vector{PauliRotation})
    passes = UnitRange{Int}[]
    pauli_on_qubit = Dict{Int,Symbol}()
    first_rotation = 1
    for (index, rotation) in enumerate(rotations)
        if any(((symbol, qind),) -> get(pauli_on_qubit, qind, symbol) != symbol, zip(rotation.symbols, rotation.qinds))
            push!(passes, first_rotation:index-1)
            empty!(pauli_on_qubit)
            first_rotation = index
        end
        for (symbol, qind) in zip(rotation.symbols, rotation.qinds)
            pauli_on_qubit[qind] = symbol
        end
    end
    push!(passes, first_rotation:length(rotations))
    return passes
end


### The plan of a pass

# The plan of rotating the classes of Pauli strings of the type `TT` with coefficients of the type `CT` on `nqubits`
# qubits by the rotations `rotations[pass]` with the `angles`, by default all of them, where coefficients below
# `min_abs_coeff` are truncated.
function _prepareclasses(rotations::Vector{PauliRotation}, angles, ::Type{TT}, ::Type{CT}, nqubits::Int, pass=eachindex(rotations);
    min_abs_coeff::Real=0) where {TT,CT}

    n_rotations = length(rotations)

    masks = zeros(TT, n_rotations)
    qinds = fill((0, 0), n_rotations)
    # where `_gatherpaulis` reads the Paulis of a rotation: a rotation on one qubit names its qubit twice, which its signs do
    # not see, since its generator holds the identity in the upper half of the local Paulis
    shifts = fill((0, 0), n_rotations)
    # the Paulis of the generator of every rotation, that on its first qubit in the lower two bits
    local_masks = zeros(UInt8, n_rotations)
    for rotation in pass
        rotation_qinds = rotations[rotation].qinds
        rotation_symbols = rotations[rotation].symbols
        _check_qind_range(nqubits, rotation_qinds)
        masks[rotation] = symboltoint(TT, rotation_symbols, rotation_qinds)
        qinds[rotation] = (rotation_qinds[1], get(rotation_qinds, 2, 0))
        shifts[rotation] = (_bitshiftfromsiteindex(first(rotation_qinds)), _bitshiftfromsiteindex(last(rotation_qinds)))
        first_pauli = UInt8(symboltoint(rotation_symbols[1]))
        second_pauli = UInt8(symboltoint(get(rotation_symbols, 2, :I)))
        local_masks[rotation] = first_pauli | (second_pauli << 2)
    end

    # the signs of every generator on one or two qubits, by its local Paulis
    sign_tables = [_signsfor(CT, _localsigns(UInt8(local_mask))) for local_mask in 0:15]
    signs = _rotationsigns(sign_tables, local_masks, shifts)
    reader = _classreader(pass, masks, qinds, local_masks, TT, nqubits)
    return PropagationBase.ClassPlan(masks, angles, signs, reader; min_abs_coeff)
end


### Readers: the rotations that anticommute with a string, and the qubits they act on

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

# Finds the rotations that anticommute with a string, as the bits of their positions in a tuple of `W` words, and the low
# bits of the qubits that those rotations act on with X, with Y and with Z, which the key of a class is reduced on. A
# layer whose rotations fall into few groups is read group by group from the whole string at once. Any other layer is read
# through the byte tables of PropagationBase, from the generators with their two bits swapped on every qubit: a string
# anticommutes with a rotation if the bits where the string and the swapped generator are both set are odd in number.
struct _ClassReader{TT,W}
    reads_in_groups::Bool

    # the low and the high bit of the Pauli that the rotations act with on every qubit, both at the low bit, and the groups
    pauli_low_bits::TT
    pauli_high_bits::TT
    groups::Vector{_RotationGroup{TT}}

    # the byte tables, with the qubits that every rotation acts on
    tables::PropagationBase.TableReader{TT,W,NTuple{3,TT}}
end

function _classreader(pass, masks::Vector{TT}, qinds, local_masks::Vector{UInt8}, ::Type{TT}, nqubits::Int) where {TT}
    n_words = cld(length(masks), 64)
    # the generators of the rotations together, which within a pass act on every qubit with one Pauli
    paulis = zero(TT)
    group_ids = Dict{Tuple{Int,UInt8,UInt8},Int}()
    groups = _RotationGroup{TT}[]
    shared_lower_qubit = false

    for rotation in pass
        paulis |= masks[rotation]
        first_qind, second_qind = qinds[rotation]
        first_pauli = local_masks[rotation] & 0x03
        second_pauli = local_masks[rotation] >> 2

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

    reads_in_groups = _readsingroups(groups) && !shared_lower_qubit

    untouched = (zero(TT), zero(TT), zero(TT))
    tables = if reads_in_groups
        PropagationBase.TableReader{TT,n_words,NTuple{3,TT}}(NTuple{n_words,UInt64}[], NTuple{3,TT}[], untouched)
    else
        touched = fill(untouched, length(masks))
        for rotation in pass
            touched[rotation] = _rotationtouches(local_masks[rotation], qinds[rotation], TT)
        end
        PropagationBase.TableReader(pass, _commutationmask.(masks), touched, untouched)
    end
    low_bits = alternatingmask(paulis)
    return _ClassReader{TT,n_words}(reads_in_groups, paulis & low_bits, _shiftdown(paulis, 1) & low_bits, groups, tables)
end

# the generator of a rotation with its two bits swapped on every qubit, with which a Pauli string shares an odd number of
# bits if it anticommutes with the generator
function _commutationmask(mask::TT) where {TT}
    low_bits = alternatingmask(mask)
    return ((mask >> 1) & low_bits) | ((mask & low_bits) << 1)
end

# the low bits of the qubits that a rotation with the Paulis `local_mask` on the qubits `qinds` acts on, by Pauli
function _rotationtouches(local_mask::UInt8, qinds::Tuple{Int,Int}, ::Type{TT}) where {TT}
    first_qind, second_qind = qinds
    touched = _touch((zero(TT), zero(TT), zero(TT)), local_mask & 0x03, symboltoint(TT, :X, first_qind))
    if second_qind != 0
        touched = _touch(touched, local_mask >> 2, symboltoint(TT, :X, second_qind))
    end
    return touched
end

# the positions of the rotations that anticommute with `pstr`, and the low bits of the qubits that those rotations act on
# with X, with Y and with Z
@inline function PropagationBase._branchinggates(reader::_ClassReader{TT,W}, pstr::TT) where {TT,W}
    if !reader.reads_in_groups
        return PropagationBase._branchinggates(reader.tables, pstr)
    end

    candidates = _candidates(reader, pstr)
    positions = ntuple(_ -> zero(UInt64), Val(W))
    touched = (zero(TT), zero(TT), zero(TT))
    for group in reader.groups
        anticommuting = _groupanticommuting(group, candidates)
        touched = _touch(touched, group.lower_pauli, anticommuting)
        if group.shift != 0
            touched = _touch(touched, group.upper_pauli, _shiftup(anticommuting, group.shift))
        end
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
    return positions, touched
end

# the low bits of the qubits that the rotations anticommuting with `pstr` act on with X, with Y and with Z
@inline function PropagationBase._touchedbits(reader::_ClassReader{TT}, pstr::TT) where {TT}
    if !reader.reads_in_groups
        return PropagationBase._touchedbits(reader.tables, pstr)
    end

    candidates = _candidates(reader, pstr)
    touched = (zero(TT), zero(TT), zero(TT))
    for group in reader.groups
        anticommuting = _groupanticommuting(group, candidates)
        touched = _touch(touched, group.lower_pauli, anticommuting)
        if group.shift != 0
            touched = _touch(touched, group.upper_pauli, _shiftup(anticommuting, group.shift))
        end
    end
    return touched
end

# the low bit of every qubit in `low_bits`, added to the qubits that rotations act on with the Pauli `pauli`
@inline function _touch(touched::NTuple{3,TT}, pauli::UInt8, low_bits::TT) where {TT}
    touched_x, touched_y, touched_z = touched
    if pauli == 0x01
        touched_x |= low_bits
    elseif pauli == 0x02
        touched_y |= low_bits
    elseif pauli == 0x03
        touched_z |= low_bits
    end
    return (touched_x, touched_y, touched_z)
end

# The low bit of every qubit on which `pstr` anticommutes with the Pauli that the rotations act with there. Two Paulis
# anticommute where the low bit of one and the high bit of the other are set an odd number of times.
@inline function _candidates(reader::_ClassReader, pstr)
    return (pstr & reader.pauli_high_bits) ⊻ (_shiftdown(pstr, 1) & reader.pauli_low_bits)
end

# the low bit of the lower qubit of every rotation of the group that anticommutes with the string of `candidates`
@inline function _groupanticommuting(group::_RotationGroup, candidates)
    if group.shift == 0
        return candidates & group.lower_mask
    else
        # a rotation on two qubits anticommutes if the Pauli string anticommutes with it on exactly one of them
        return (candidates ⊻ _shiftdown(candidates, group.shift)) & group.lower_mask
    end
end

# Shifts of a whole Pauli string. Below 64 bits, a shift only moves bits between neighbouring limbs.
@inline _shiftdown(pstr, shift::Int) = pstr >> shift
@inline _shiftup(pstr, shift::Int) = pstr << shift

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


### Keys

# The Pauli string with every qubit that the rotations it anticommutes with act on reduced to what they leave unchanged:
# X leaves the high bit, Y the low bit and Z the parity of the two, and a qubit acted on with two different Paulis is cleared.
# All Pauli strings of a class have the same key. Within a pass no qubit is acted on with two different Paulis, so
# reducing a string on the qubits that another string's rotations touch only multiplies it by Paulis that commute with
# every rotation: two strings have the same key that way only if they anticommute with the same rotations.
@inline function PropagationBase._classkey(pstr::TT, touched::NTuple{3,TT}) where {TT}
    touched_x, touched_y, touched_z = touched
    mixed = (touched_x & touched_y) | (touched_x & touched_z) | (touched_y & touched_z)
    key = pstr & ~(mixed | _shiftup(mixed, 1))
    key &= ~(touched_x & ~mixed)
    key &= ~_shiftup(touched_y & ~mixed, 1)
    low_z = key & touched_z & ~mixed
    return key ⊻ (low_z | _shiftup(low_z, 1))
end

# The bits that tell the strings of a class apart, on the qubits that its rotations touch: the low bit where they touch
# with X or Z, the high bit where they touch with Y, and both where with two Paulis.
@inline function PropagationBase._keybits(touched::NTuple{3,TT}) where {TT}
    touched_x, touched_y, touched_z = touched
    mixed = (touched_x & touched_y) | (touched_x & touched_z) | (touched_y & touched_z)
    return touched_x | touched_z | mixed | _shiftup(touched_y | mixed, 1)
end


### Signs

# the sign a rotation gives the Pauli string it creates from each combination of Paulis on its qubits, 0 where it commutes with them
function _localsigns(local_mask::UInt8)
    function sign_of(index)
        paulis = UInt8(index - 1)
        if commutes(local_mask, paulis)
            return Int8(0)
        end
        _, sign = paulirotationproduct(local_mask, paulis)
        return Int8(sign)
    end
    return ntuple(sign_of, 16)
end

# The signs as the real type of floating-point coefficients, which they multiply as the integers do, so that they are not
# converted for every entry.
_signsfor(::Type{CT}, signs) where {CT<:Union{AbstractFloat,Complex{<:AbstractFloat}}} = map(sign -> convert(real(CT), sign), signs)
_signsfor(::Type, signs) = signs

# the signs that a generator on two qubits gives every combination of Paulis there, and where those are read
struct _GeneratorSigns{ST}
    signs::NTuple{16,ST}
    local_mask::Int
    shifts::Tuple{Int,Int}
end

# The signs of every rotation, which `PropagationBase._pairsigns` reads: those of its generator, read for every pair of a
# rotation on two qubits, and the same two for every pair of a rotation on one qubit, which leaves every Pauli string of
# its class with one of two Paulis there, the lower entry of a pair with the one whose key bit is clear.
function _rotationsigns(sign_tables::Vector{NTuple{16,R}}, local_masks::Vector{UInt8}, shifts) where {R}
    signs = Vector{Union{Tuple{R,R},_GeneratorSigns{R}}}(undef, length(local_masks))
    for rotation in eachindex(local_masks)
        local_mask = local_masks[rotation]
        rotation_signs = sign_tables[local_mask+1]
        signs[rotation] = if local_mask >> 2 == 0x00
            lower_paulis = (local_mask & 0x03) == 0x02 ? 0x01 : 0x02
            (rotation_signs[lower_paulis+1], rotation_signs[(lower_paulis⊻local_mask)+1])
        else
            _GeneratorSigns(rotation_signs, Int(local_mask), shifts[rotation])
        end
    end
    return signs
end

# the signs that the rotation gives the partner of the lower Pauli string of a pair and the lower string, from its Paulis
@inline function PropagationBase._pairsigns(generator_signs::_GeneratorSigns, lower_pstr)
    paulis = _gatherpaulis(lower_pstr, generator_signs.shifts)
    return (generator_signs.signs[paulis+1], generator_signs.signs[(paulis⊻generator_signs.local_mask)+1])
end
