###
##
# What a sublayer reads from a Pauli string: the rotations that anticommute with it, which span its orbit,
# the representative of the orbit, and the coordinate of the Pauli string in it.
# A Pauli string is the representative times the generators of some of those rotations, the selected ones,
# and its coordinate has a bit set for each of them, with the rotations counted by their lowest qubit.
# A rotation is selected if its pivot bit, which lies on its lowest qubit, is set once the generators of
# the selected rotations below it have been multiplied out of the Pauli string.
##
###

# a sublayer with more distances between the qubits of its rotations is read rotation by rotation
const _MAX_ROTATION_GROUPS = 4

# the rotations of a sublayer whose qubits are the same distance apart
struct RotationGroup{TT}
    # the low bit of the lower qubit of every rotation
    lower_mask::TT
    # from the lower to the upper qubit in bits, 0 for rotations on one qubit
    shift::Int
    # from the low bit of the lower qubit to the pivot bit
    pivot_offset::Int
    # the Paulis of the generator on the lower and on the upper qubit
    lower_pauli::UInt8
    upper_pauli::UInt8
end

"""
    SubLayerPlan(layer::RotationLayer, sublayer, theta, TT, nqubits)

The rotations `sublayer` of the `layer` with the parameter `theta`, prepared for Pauli strings of the type `TT` on `nqubits` qubits.
The rotations are kept in the order in which they are applied.
"""
struct SubLayerPlan{TT,A}
    # per rotation
    masks::Vector{TT}
    pivots::Vector{Int}
    qinds::Vector{NTuple{2,Int}}
    cosines::Vector{A}
    sines::Vector{A}

    # per qubit: the rotation whose lower qubit it is and the upper qubit of that rotation, 0 where there is none,
    # and the rotations whose upper qubit it is
    rotation_at_lower::Vector{Int32}
    upper_of_lower::Vector{Int32}
    rotations_at_upper::Vector{Vector{Int32}}

    # the low bit of every qubit a rotation acts on with its first and with its second symbol
    role_masks::NTuple{2,TT}
    symbol_codes::NTuple{2,UInt8}

    # per Pauli pair on the qubits of a rotation
    local_mask::UInt8
    signs::NTuple{16,Int8}
    weight_changes::NTuple{16,Int8}

    # empty if the rotations are read one by one
    groups::Vector{RotationGroup{TT}}

    # 1 if the rotations are applied in the order of their lowest qubits, -1 if in the reverse order, 0 otherwise
    order::Int

    # whether any two rotations share a qubit
    overlapping::Bool

    # per rotation and qubit of it, the other rotation on that qubit: 0 if there is none, -1 if there are several
    neighbors::Vector{NTuple{2,Int32}}

    # The signs and the weight change of a rotation whose qubits other rotations act on, per Pauli pair of the representative
    # and per way in which the rotations on the two qubits are selected: neither, the first, the second, or both.
    shared_signs_from_lower::NTuple{16,NTuple{4,Int8}}
    shared_signs_from_upper::NTuple{16,NTuple{4,Int8}}
    shared_weight_changes::NTuple{16,NTuple{4,Int8}}
end

function SubLayerPlan(layer::RotationLayer, sublayer::Vector{Int}, theta, ::Type{TT}, nqubits::Int) where {TT}
    symbols = layer.symbols
    n_rotations = length(sublayer)

    masks = Vector{TT}(undef, n_rotations)
    pivots = Vector{Int}(undef, n_rotations)
    qinds = Vector{NTuple{2,Int}}(undef, n_rotations)
    angles = [_rotationangle(theta, index) for index in sublayer]

    rotation_at_lower = zeros(Int32, nqubits)
    upper_of_lower = zeros(Int32, nqubits)
    rotations_at_upper = [Int32[] for _ in 1:nqubits]
    rotations_on_qubit = [Int32[] for _ in 1:nqubits]
    role_masks = [zero(TT), zero(TT)]

    for (position, index) in enumerate(sublayer)
        rotation_qinds = layer.qinds[index]
        _check_qind_range(nqubits, rotation_qinds)

        masks[position] = symboltoint(TT, symbols, rotation_qinds)
        qinds[position] = (rotation_qinds[1], get(rotation_qinds, 2, 0))

        lower_qind, lower_role = findmin(rotation_qinds)
        pivots[position] = 2 * (lower_qind - 1) + _pivotoffset(symbols[lower_role])
        if rotation_at_lower[lower_qind] != 0
            throw(ArgumentError("Two rotations of a sublayer have the lowest qubit $lower_qind."))
        end
        rotation_at_lower[lower_qind] = position

        if length(rotation_qinds) == 2
            upper_qind = maximum(rotation_qinds)
            upper_of_lower[lower_qind] = upper_qind
            push!(rotations_at_upper[upper_qind], position)
        end

        for (role, qind) in enumerate(rotation_qinds)
            push!(rotations_on_qubit[qind], position)
            role_masks[role] |= symboltoint(TT, :X, qind)
        end
    end

    neighbors = [(_neighbor(rotations_on_qubit, first_qind, position), _neighbor(rotations_on_qubit, second_qind, position))
                 for (position, (first_qind, second_qind)) in enumerate(qinds)]

    symbol_codes = (UInt8(symboltoint(symbols[1])), UInt8(symboltoint(get(symbols, 2, :I))))
    local_mask = symbol_codes[1] | (symbol_codes[2] << 2)

    groups = _rotationgroups(TT, symbols, qinds)
    if !_readsingroups(groups)
        empty!(groups)
    end

    signs = _localsigns(local_mask)
    weight_changes = _localweightchanges(local_mask)

    # a selected rotation multiplies the Pauli on the qubit it shares by the Pauli of the generator, which is the same for both rotations
    selections = ((0x00, 0x00), (symbol_codes[1], 0x00), (0x00, symbol_codes[2]), (symbol_codes[1], symbol_codes[2]))
    paulis_with(pair_index, selected) = (pair_index - 1) ⊻ Int(selections[selected][1]) ⊻ (Int(selections[selected][2]) << 2)
    signs_from_lower = ntuple(pair_index -> ntuple(selected -> signs[paulis_with(pair_index, selected)+1], 4), 16)
    signs_from_upper = ntuple(pair_index -> ntuple(selected -> signs[(paulis_with(pair_index, selected)⊻local_mask)+1], 4), 16)
    shared_weight_changes = ntuple(pair_index -> ntuple(selected -> weight_changes[paulis_with(pair_index, selected)+1], 4), 16)

    return SubLayerPlan(masks, pivots, qinds, cos.(angles), sin.(angles),
        rotation_at_lower, upper_of_lower, rotations_at_upper, (role_masks[1], role_masks[2]), symbol_codes,
        local_mask, signs, weight_changes, groups, _applicationorder(pivots),
        any(rotations -> length(rotations) > 1, rotations_on_qubit), neighbors,
        signs_from_lower, signs_from_upper, shared_weight_changes)
end

# the other rotation on the qubit `qind` of the rotation at `position`
function _neighbor(rotations_on_qubit::Vector{Vector{Int32}}, qind::Int, position::Int)
    if qind == 0
        return Int32(0)
    end

    others = filter(!=(position), rotations_on_qubit[qind])
    if isempty(others)
        return Int32(0)
    elseif length(others) == 1
        return only(others)
    else
        return Int32(-1)
    end
end

# whether the rotations are read from the whole Pauli string at once
_readsingroups(groups) = length(groups) <= _MAX_ROTATION_GROUPS

# X and Z flip the low bit of their qubit, Y only the high bit
function _pivotoffset(symbol::Symbol)
    if symbol == :Y
        return 1
    else
        return 0
    end
end

function _rotationgroups(::Type{TT}, symbols::Vector{Symbol}, qinds::Vector{NTuple{2,Int}}) where {TT}
    lower_masks = Dict{Tuple{Int,Symbol,Symbol},TT}()

    for (first_qind, second_qind) in qinds
        if second_qind == 0
            group = (0, symbols[1], :I)
            lower_qind = first_qind
        elseif first_qind < second_qind
            group = (2 * (second_qind - first_qind), symbols[1], symbols[2])
            lower_qind = first_qind
        else
            group = (2 * (first_qind - second_qind), symbols[2], symbols[1])
            lower_qind = second_qind
        end
        lower_masks[group] = get(lower_masks, group, zero(TT)) | symboltoint(TT, :X, lower_qind)
    end

    return [RotationGroup(lower_mask, shift, _pivotoffset(lower_symbol), UInt8(symboltoint(lower_symbol)), UInt8(symboltoint(upper_symbol)))
            for ((shift, lower_symbol, upper_symbol), lower_mask) in lower_masks]
end

# whether the rotations, in the order in which they are applied, go up or down in their lowest qubits
function _applicationorder(pivots::Vector{Int})
    if issorted(pivots)
        return 1
    elseif issorted(pivots; rev=true)
        return -1
    else
        return 0
    end
end

# the sign a rotation gives the Pauli string it creates from each Pauli pair, 0 where it commutes with the pair
function _localsigns(local_mask::UInt8)
    function sign_of(pair_index)
        paulis = UInt8(pair_index - 1)
        if commutes(local_mask, paulis)
            return Int8(0)
        end
        _, sign = paulirotationproduct(local_mask, paulis)
        return Int8(sign)
    end
    return ntuple(sign_of, 16)
end

# how the weight of a Pauli pair changes under the product with the generator
function _localweightchanges(local_mask::UInt8)
    weight_change_of(pair_index) = Int8(countweight(UInt8(pair_index - 1) ⊻ local_mask) - countweight(UInt8(pair_index - 1)))
    return ntuple(weight_change_of, 16)
end

# the Paulis of `pstr` on the qubits of the rotation at `position`, the first qubit in the low bits
@inline function _localpaulis(plan::SubLayerPlan, pstr, position::Integer)
    first_qind, second_qind = plan.qinds[position]
    limbs = _limbs(pstr)
    paulis = _pauliat(limbs, first_qind)
    if second_qind != 0
        paulis |= _pauliat(limbs, second_qind) << 2
    end
    return paulis
end

@inline function _pauliat(limbs::NTuple{L,UInt64}, qind::Int) where {L}
    bit = 2 * (qind - 1)
    return Int((limbs[(bit>>6)+1] >> (bit & 63)) & 0x3)
end


### Reading a Pauli string

"""
    _locateinorbit!(positions, plan::SubLayerPlan, pstr)

Returns the representative of the orbit of `pstr`, the coordinate of `pstr` in it, and the number of rotations that span the orbit.
`positions` is scratch memory.
"""
@inline function _locateinorbit!(positions::Vector{Int32}, plan::SubLayerPlan{TT}, pstr::TT) where {TT}
    if isempty(plan.groups)
        return _locatebyrotation!(positions, plan, pstr)
    else
        return _locatebygroup(plan, pstr)
    end
end

"""
    _orbitrotations!(positions, plan::SubLayerPlan, representative)

Writes the positions of the rotations that span the orbit of `representative` into `positions`, in the order of the bits of a coordinate, and returns their number.
"""
@inline function _orbitrotations!(positions::Vector{Int32}, plan::SubLayerPlan{TT}, representative::TT) where {TT}
    if isempty(plan.groups)
        return _anticommuting!(positions, plan, representative)
    end

    anticommuting = _limbs(_anticommutingmask(plan, representative))
    n_found = 0
    for limb_index in eachindex(anticommuting)
        limb = anticommuting[limb_index]
        while limb != 0
            qind = 32 * (limb_index - 1) + (trailing_zeros(limb) >> 1) + 1
            limb &= limb - 1
            n_found += 1
            positions[n_found] = plan.rotation_at_lower[qind]
        end
    end
    return n_found
end

# a Pauli string as 64-bit limbs, the lowest first
@inline _limbs(pstr::NTupleInteger) = pstr.limbs
@inline _limbs(pstr::UInt128) = (pstr % UInt64, (pstr >> 64) % UInt64)
@inline _limbs(pstr::Union{UInt8,UInt16,UInt32,UInt64}) = (pstr % UInt64,)

@inline _readbit(limbs::NTuple{L,UInt64}, bit::Int) where {L} = isodd(limbs[(bit>>6)+1] >> (bit & 63))

# the low bit of every qubit on which `pstr` anticommutes with the Pauli `symbol_code`
@inline function _locallyanticommuting(symbol_code::UInt8, pstr::TT) where {TT}
    low_bits = alternatingmask(pstr)
    if symbol_code == 0x01
        # X anticommutes with Y and Z, which have the high bit set
        return _shiftdown(pstr, 1) & low_bits
    elseif symbol_code == 0x02
        # Y anticommutes with X and Z, which have the low bit set
        return pstr & low_bits
    else
        # Z anticommutes with X and Y, whose bits differ
        return (pstr ⊻ _shiftdown(pstr, 1)) & low_bits
    end
end

# the low bit of every qubit on which `pstr` anticommutes with the rotation acting there
@inline function _candidates(plan::SubLayerPlan{TT}, pstr::TT) where {TT}
    candidates = _locallyanticommuting(plan.symbol_codes[1], pstr) & plan.role_masks[1]
    if plan.symbol_codes[2] != 0x00
        candidates |= _locallyanticommuting(plan.symbol_codes[2], pstr) & plan.role_masks[2]
    end
    return candidates
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


### Rotations read in groups, from the whole Pauli string at once

# the low bit of the lower qubit of every rotation of the group that anticommutes with `pstr`
@inline function _anticommutingmask(group::RotationGroup{TT}, candidates::TT) where {TT}
    if group.shift == 0
        return candidates & group.lower_mask
    else
        # a rotation on two qubits anticommutes if the Pauli string anticommutes with it on exactly one of them
        return (candidates ⊻ _shiftdown(candidates, group.shift)) & group.lower_mask
    end
end

@inline function _anticommutingmask(plan::SubLayerPlan{TT}, pstr::TT) where {TT}
    candidates = _candidates(plan, pstr)
    anticommuting = zero(TT)
    for group in plan.groups
        anticommuting |= _anticommutingmask(group, candidates)
    end
    return anticommuting
end

# the bits of the Pauli `pauli` on every qubit whose low bit is set in `low_bits`
@inline function _paulionqubits(pauli::UInt8, low_bits::TT) where {TT}
    if pauli == 0x01
        return low_bits
    elseif pauli == 0x02
        return _shiftup(low_bits, 1)
    elseif pauli == 0x03
        return low_bits | _shiftup(low_bits, 1)
    else
        return zero(TT)
    end
end

@inline function _locatebygroup(plan::SubLayerPlan{TT}, pstr::TT) where {TT}
    candidates = _candidates(plan, pstr)
    anticommuting = zero(TT)
    pivot_bits = zero(TT)

    for group in plan.groups
        anticommuting_here = _anticommutingmask(group, candidates)
        anticommuting |= anticommuting_here
        pivot_bits |= _shiftdown(pstr, group.pivot_offset) & anticommuting_here
    end

    selected = pivot_bits
    if plan.overlapping
        selected = _selectinturn(plan, pivot_bits, anticommuting)
    end

    flipped = zero(TT)
    for group in plan.groups
        selected_here = selected & group.lower_mask
        flipped ⊻= _paulionqubits(group.lower_pauli, selected_here)
        if group.shift != 0
            flipped ⊻= _shiftup(_paulionqubits(group.upper_pauli, selected_here), group.shift)
        end
    end

    coordinate, n_stages = _compressbits(selected, anticommuting)
    return pstr ⊻ flipped, coordinate, n_stages
end

# The generator of a selected rotation flips the pivot bit of the rotation on its upper qubit.
# Every round selects the rotations one qubit further up a run of rotations that share qubits, until nothing changes.
@inline function _selectinturn(plan::SubLayerPlan{TT}, pivot_bits::TT, anticommuting::TT) where {TT}
    selected = pivot_bits
    while true
        flipped_pivots = zero(TT)
        for group in plan.groups
            if group.shift != 0
                flipped_pivots ⊻= _shiftup(selected & group.lower_mask, group.shift)
            end
        end

        newly_selected = (pivot_bits ⊻ flipped_pivots) & anticommuting
        if newly_selected == selected
            return selected
        end
        selected = newly_selected
    end
end

# the bits of `bits` where `mask` is set, moved next to each other, and their number
@inline function _compressbits(bits::TT, mask::TT) where {TT}
    bit_limbs = _limbs(bits)
    mask_limbs = _limbs(mask)
    compressed = zero(UInt64)
    n_bits = 0

    for limb_index in eachindex(mask_limbs)
        mask_limb = mask_limbs[limb_index]
        bit_limb = bit_limbs[limb_index]
        while mask_limb != 0
            bit = trailing_zeros(mask_limb)
            compressed |= ((bit_limb >> (bit & 63)) & one(UInt64)) << (n_bits & 63)
            n_bits += 1
            mask_limb &= mask_limb - 1
        end
    end

    return compressed, n_bits
end


### Rotations read one by one, through the qubits on which the Pauli string anticommutes with them

"""
    _anticommuting!(positions, plan::SubLayerPlan, pstr)

Writes the positions of the rotations that anticommute with `pstr` into `positions`, in the order of their lowest qubits, and returns their number.
A rotation on two qubits anticommutes if `pstr` anticommutes with it on exactly one of them.
"""
function _anticommuting!(positions::Vector{Int32}, plan::SubLayerPlan{TT}, pstr::TT) where {TT}
    candidates = _limbs(_candidates(plan, pstr))
    n_found = 0

    for limb_index in eachindex(candidates)
        limb = candidates[limb_index]
        while limb != 0
            qind = 32 * (limb_index - 1) + (trailing_zeros(limb) >> 1) + 1
            limb &= limb - 1

            # the rotation of which this is the lower qubit
            position = plan.rotation_at_lower[qind]
            if position != 0
                upper_qind = plan.upper_of_lower[qind]
                if upper_qind == 0 || !_readbit(candidates, 2 * (upper_qind - 1))
                    n_found += 1
                    positions[n_found] = position
                end
            end

            # the rotations of which this is the upper qubit
            for upper_position in plan.rotations_at_upper[qind]
                if !_readbit(candidates, plan.pivots[upper_position] & ~1)
                    n_found += 1
                    positions[n_found] = upper_position
                end
            end
        end
    end

    _sortbypivot!(positions, n_found, plan.pivots)
    return n_found
end

# The rotations are found in the order of the qubits on which the Pauli string anticommutes, which is close to that of their lowest qubits.
function _sortbypivot!(positions::Vector{Int32}, n::Int, pivots::Vector{Int})
    for i in 2:n
        position = positions[i]
        j = i - 1
        while j >= 1 && pivots[positions[j]] > pivots[position]
            positions[j+1] = positions[j]
            j -= 1
        end
        positions[j+1] = position
    end
    return positions
end

@inline function _locatebyrotation!(positions::Vector{Int32}, plan::SubLayerPlan{TT}, pstr::TT) where {TT}
    n_stages = _anticommuting!(positions, plan, pstr)
    representative = pstr
    coordinate = zero(UInt64)

    # a coordinate holds `_MAX_STAGES` stages, and the caller gives up on more
    for bit in 1:min(n_stages, _MAX_STAGES)
        position = positions[bit]
        if _readbit(_limbs(representative), plan.pivots[position])
            representative ⊻= plan.masks[position]
            coordinate |= one(UInt64) << (bit - 1)
        end
    end

    return representative, coordinate, n_stages
end
