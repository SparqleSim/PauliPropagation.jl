###
##
# A second way to propagate a `RotationLayer`: class by class. The Pauli strings that anticommute with the same rotations
# of the layer form a class, and those rotations keep a string within its class, since they commute with each other.
# Every Pauli string becomes a record, the records are partitioned by the hash of the key of their class, and within a
# partition every class is rotated in a small table of its own: one rotation after the other in the order of the layer,
# each mixing every two strings that its generator turns into each other and truncating both, as the rotations one by one do.
# All rotations of the layer are applied in one pass, whether or not they share qubits, and a string may anticommute
# with any number of them.
##
###

# The tables of the classes of a partition hold what its records make as well, so a record counts this many times its
# size when the partitions are sized.
const _CLASS_BYTES_FACTOR = 4

# whether the plan rotates classes, rather than transforming orbits
_rotatesclasses(plan) = hasfield(typeof(plan), :positions)

_recordbytes(plan, record_bytes::Int) = _rotatesclasses(plan) ? _CLASS_BYTES_FACTOR * record_bytes : record_bytes

"""
    _applybyclass!(layer::RotationLayer, prop_cache, theta, truncation; thread=true)

Applies all rotations of the layer in one pass, class by class.
"""
function _applybyclass!(layer::RotationLayer, prop_cache::AbstractPauliPropagationCache, theta, truncation; thread::Bool=true)
    workspace = _takeworkspace(paulitype(prop_cache), coefftype(prop_cache))
    try
        for rotations in _classpasses(layer)
            plan = _prepareclasses(layer, theta, paulitype(prop_cache), nqubits(prop_cache), rotations)
            # every record has a class, so none lies in the partitions of long orbits
            _applysublayer!(StorageType(prop_cache), prop_cache, plan, plan, truncation, workspace; thread)
        end
    finally
        _putbackworkspace!(workspace)
    end
    return prop_cache
end

# The rotations of the layer that one pass rotates class by class: all of them, unless a qubit is acted on with two
# different Paulis. The key of a class keeps, on every qubit that its rotations act on with one Pauli, whether the string
# anticommutes with that Pauli there, and so tells apart strings that anticommute with different rotations. A qubit acted
# on with two Paulis keeps neither, so strings with different rotations could share a key there. In a layer of commuting
# rotations that happens only for a rotation and its swapped pair, as XZ on (1, 2) and on (2, 1), whose lowest qubit is the
# same, so that they lie in different sublayers: such a layer takes one pass per sublayer.
function _classpasses(layer::RotationLayer)
    if _actswithtwopaulis(layer)
        return layer.sublayers
    else
        return (eachindex(layer.qinds),)
    end
end

# whether a qubit is the first qubit of one rotation and the second of another, with a different Pauli
function _actswithtwopaulis(layer::RotationLayer)
    if length(layer.symbols) < 2 || layer.symbols[1] == layer.symbols[2]
        return false
    end
    first_qubits = Set(qinds[1] for qinds in layer.qinds)
    return any(qinds -> qinds[2] in first_qubits, layer.qinds)
end

# whether the layer is applied class by class, which by default a layer of more than one sublayer is
function _appliesbyclass(layer::RotationLayer, layer_method::Symbol)
    if layer_method == :auto
        return length(layer.sublayers) > 1
    else
        return layer_method == :classes
    end
end

"""
    _prepareclasses(layer::RotationLayer, theta, TT, nqubits, rotations=eachindex(layer.qinds))

The `rotations` of the `layer` with the parameter `theta`, by default all of them, prepared for rotating the classes of Pauli strings
of the type `TT` on `nqubits` qubits. Rotations with the same distance between their qubits and the same Paulis on them form a
group that is read from the whole string at once.
"""
function _prepareclasses(layer::RotationLayer, theta, ::Type{TT}, nqubits::Int, rotations=eachindex(layer.qinds)) where {TT}
    symbols = layer.symbols
    n_rotations = length(layer.qinds)

    masks = Vector{TT}(undef, n_rotations)
    qinds = Vector{NTuple{2,Int}}(undef, n_rotations)
    angles = [_rotationangle(theta, index) for index in 1:n_rotations]
    rotations_on_qubit = [Int32[] for _ in 1:nqubits]
    role_masks = [zero(TT), zero(TT)]

    group_ids = Dict{Tuple{Int,Symbol,Symbol},Int}()
    group_lower_masks = TT[]
    group_shifts = Int[]
    group_paulis = NTuple{2,UInt8}[]
    group_rotations = Vector{Int32}[]
    shared_lower_qubit = false

    for rotation in rotations
        rotation_qinds = layer.qinds[rotation]
        _check_qind_range(nqubits, rotation_qinds)
        masks[rotation] = symboltoint(TT, symbols, rotation_qinds)
        qinds[rotation] = (rotation_qinds[1], get(rotation_qinds, 2, 0))

        for (role, qind) in enumerate(rotation_qinds)
            push!(rotations_on_qubit[qind], rotation)
            role_masks[role] |= symboltoint(TT, :X, qind)
        end

        group, lower_qind = if length(rotation_qinds) == 1
            (0, symbols[1], :I), rotation_qinds[1]
        elseif rotation_qinds[1] < rotation_qinds[2]
            (2 * (rotation_qinds[2] - rotation_qinds[1]), symbols[1], symbols[2]), rotation_qinds[1]
        else
            (2 * (rotation_qinds[1] - rotation_qinds[2]), symbols[2], symbols[1]), rotation_qinds[2]
        end

        group_id = get(group_ids, group, 0)
        if group_id == 0
            push!(group_lower_masks, zero(TT))
            push!(group_shifts, group[1])
            push!(group_paulis, (UInt8(symboltoint(group[2])), UInt8(symboltoint(group[3]))))
            push!(group_rotations, zeros(Int32, nqubits))
            group_id = length(group_lower_masks)
            group_ids[group] = group_id
        end

        # a rotation that comes twice cannot be found through its lower qubit
        if group_rotations[group_id][lower_qind] != 0
            shared_lower_qubit = true
        end
        group_lower_masks[group_id] |= symboltoint(TT, :X, lower_qind)
        group_rotations[group_id][lower_qind] = rotation
    end

    groups = [(; lower_mask=group_lower_masks[id], shift=group_shifts[id], lower_pauli=group_paulis[id][1],
        upper_pauli=group_paulis[id][2], rotation_at_lower=group_rotations[id]) for id in eachindex(group_lower_masks)]

    symbol_codes = (UInt8(symboltoint(symbols[1])), UInt8(symboltoint(get(symbols, 2, :I))))
    local_mask = symbol_codes[1] | (symbol_codes[2] << 2)

    # A qubit that is the first qubit of one rotation and the second of another is read with either Pauli,
    # which the groups cannot tell apart when the two Paulis differ.
    mixed_roles = symbol_codes[2] != 0x00 && symbol_codes[1] != symbol_codes[2] && !iszero(role_masks[1] & role_masks[2])
    reads_in_groups = _readsingroups(groups) && !shared_lower_qubit && !mixed_roles

    # the place of every rotation in the order in which the layer applies them
    positions = zeros(Int, n_rotations)
    position = 0
    for sublayer in layer.sublayers, rotation in sublayer
        position += 1
        positions[rotation] = position
    end

    return (; masks, qinds, cosines=cos.(angles), sines=sin.(angles), positions, symbol_codes,
        role_masks=(role_masks[1], role_masks[2]), local_mask, signs=_localsigns(local_mask), groups, reads_in_groups, rotations_on_qubit)
end


### What a layer reads from a Pauli string

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

@inline _qubitlowbit(::Type{TT}, qind::Int) where {TT} = one(TT) << (2 * (qind - 1))

"""
    _touchedqubits(plan, pstr)

The low bits of the qubits that the rotations anticommuting with `pstr` act on with X, with Y and with Z.
"""
@inline function _touchedqubits(plan, pstr::TT) where {TT}
    candidates = _candidates(plan, pstr)
    touched = (zero(TT), zero(TT), zero(TT))

    if plan.reads_in_groups
        for group in plan.groups
            anticommuting = _groupanticommuting(group, candidates)
            touched = _touch(touched, group.lower_pauli, anticommuting)
            if group.shift != 0
                touched = _touch(touched, group.upper_pauli, _shiftup(anticommuting, group.shift))
            end
        end
        return touched
    end

    # otherwise through the qubits on which the Pauli string anticommutes with a rotation acting there
    candidate_limbs = _limbs(candidates)
    for limb_index in eachindex(candidate_limbs)
        limb = candidate_limbs[limb_index]
        while limb != 0
            qind = 32 * (limb_index - 1) + (trailing_zeros(limb) >> 1) + 1
            limb &= limb - 1

            for rotation in plan.rotations_on_qubit[qind]
                if plan.signs[_localpaulis(plan, pstr, rotation)+1] != 0
                    first_qind, second_qind = plan.qinds[rotation]
                    touched = _touch(touched, plan.symbol_codes[1], _qubitlowbit(TT, first_qind))
                    if second_qind != 0
                        touched = _touch(touched, plan.symbol_codes[2], _qubitlowbit(TT, second_qind))
                    end
                end
            end
        end
    end
    return touched
end

"""
    _classrotations!(rotations, plan, pstr)

Writes the rotations of the layer that anticommute with `pstr` into `rotations`, in the order in which the layer applies them, and returns their number.
"""
function _classrotations!(rotations::Vector{Int32}, plan, pstr)
    candidates = _candidates(plan, pstr)
    n_found = 0

    if plan.reads_in_groups
        for group in plan.groups
            anticommuting = _limbs(_groupanticommuting(group, candidates))
            for limb_index in eachindex(anticommuting)
                limb = anticommuting[limb_index]
                while limb != 0
                    qind = 32 * (limb_index - 1) + (trailing_zeros(limb) >> 1) + 1
                    limb &= limb - 1
                    n_found += 1
                    rotations[n_found] = group.rotation_at_lower[qind]
                end
            end
        end
    else
        candidate_limbs = _limbs(candidates)
        for limb_index in eachindex(candidate_limbs)
            limb = candidate_limbs[limb_index]
            while limb != 0
                qind = 32 * (limb_index - 1) + (trailing_zeros(limb) >> 1) + 1
                limb &= limb - 1

                for rotation in plan.rotations_on_qubit[qind]
                    if plan.signs[_localpaulis(plan, pstr, rotation)+1] == 0
                        continue
                    end
                    first_qind, second_qind = plan.qinds[rotation]
                    other_qind = first_qind == qind ? second_qind : first_qind

                    # a rotation with both qubits among the candidates is taken at the lower one
                    if other_qind != 0 && other_qind < qind && _readbit(candidates, 2 * (other_qind - 1))
                        continue
                    end
                    n_found += 1
                    rotations[n_found] = rotation
                end
            end
        end
    end

    _sortbyposition!(rotations, n_found, plan.positions)
    return n_found
end

# the rotations in the order in which the layer applies them
function _sortbyposition!(rotations::Vector{Int32}, n::Int, positions::Vector{Int})
    for i in 2:n
        rotation = rotations[i]
        j = i - 1
        while j >= 1 && positions[rotations[j]] > positions[rotation]
            rotations[j+1] = rotations[j]
            j -= 1
        end
        rotations[j+1] = rotation
    end
    return rotations
end

# The Pauli string with every qubit that the rotations it anticommutes with act on reduced to what they leave unchanged:
# X leaves the high bit, Y the low bit and Z the parity of the two, and a qubit acted on with two different Paulis is cleared.
# All Pauli strings of a class have the same key.
@inline function _orbitkey(pstr::TT, touched::NTuple{3,TT}) where {TT}
    touched_x, touched_y, touched_z = touched
    mixed = (touched_x & touched_y) | (touched_x & touched_z) | (touched_y & touched_z)
    key = pstr & ~(mixed | _shiftup(mixed, 1))
    key &= ~(touched_x & ~mixed)
    key &= ~_shiftup(touched_y & ~mixed, 1)
    low_z = key & touched_z & ~mixed
    return key ⊻ (low_z | _shiftup(low_z, 1))
end

# The label of the record of a Pauli string: a hash above `_HASH_SHIFT`, which picks the partition, and in the lowest bit
# whether the string anticommutes with any rotation. The hash is that of the key of its class, or else that of the string,
# so that the strings no rotation touches are spread over the partitions as well.
@inline function _classlabel(plan, pstr)
    touched = _touchedqubits(plan, pstr)
    if iszero(touched[1] | touched[2] | touched[3])
        return (_hashbits(pstr) << _HASH_SHIFT) % Int
    end
    return ((_hashbits(_orbitkey(pstr, touched)) << _HASH_SHIFT) | one(UInt64)) % Int
end

@inline _hasclass(label::Int) = isodd(label)


### Rotating the classes of a partition

"""
    _rotatepartitionbyclass!(output, task, plan, truncation, record_terms, record_coeffs, record_labels, lo, hi)

Groups the Pauli strings of the records `lo` to `hi` by class, rotates every class, and writes what the truncations keep to `output`.
A Pauli string that anticommutes with no rotation goes to `output` as it is.
"""
function _rotatepartitionbyclass!(output, task, plan, truncation, record_terms::Vector{TT}, record_coeffs::Vector{CT},
    record_labels::Vector{Int}, lo::Int, hi::Int) where {TT,CT}

    if !(1 <= lo && hi <= min(length(record_terms), length(record_coeffs), length(record_labels)))
        throw(ArgumentError("the records $lo to $hi are not among the records"))
    end
    n_records = hi - lo + 1
    class_of = _ensurelength!(task.class_of, n_records)
    class_keys = _ensurelength!(task.class_keys, n_records)
    class_hashes = _ensurelength!(task.class_hashes, n_records)
    class_starts = _ensurelength!(task.class_starts, n_records + 1)

    # the classes found so far, through the hash of their key
    table_length = max(16, nextpow(2, 2 * n_records))
    slot_mask = table_length - 1
    slots = _ensurelength!(task.slots, table_length)
    fill!(view(slots, 1:table_length), zero(Int32))
    n_classes = 0

    for i in lo:hi
        pstr = record_terms[i]
        label = record_labels[i]
        if !_hasclass(label)
            class_of[i-lo+1] = 0
            coeff = record_coeffs[i]
            if !_istruncated(truncation, pstr, coeff)
                _emit!(output, pstr, coeff)
            end
            continue
        end

        key = _orbitkey(pstr, _touchedqubits(plan, pstr))
        hashbits = _labelhash(label)
        slot = Int(hashbits & (slot_mask % UInt64)) + 1
        class = Int(slots[slot])
        while class != 0 && !(class_hashes[class] == hashbits && class_keys[class] == key)
            slot = (slot & slot_mask) + 1
            class = Int(slots[slot])
        end
        if class == 0
            n_classes += 1
            class = n_classes
            class_keys[class] = key
            class_hashes[class] = hashbits
            class_starts[class] = 0
            slots[slot] = class
        end
        class_starts[class] += 1
        class_of[i-lo+1] = class
    end

    # the records one class after the other
    next_start = 1
    for class in 1:n_classes
        n_here = class_starts[class]
        class_starts[class] = next_start
        next_start += n_here
    end
    class_starts[n_classes+1] = next_start
    class_records = _ensurelength!(task.class_records, n_records)
    for i in 1:n_records
        class = class_of[i]
        if class != 0
            class_records[class_starts[class]] = i
            class_starts[class] += 1
        end
    end
    for class in n_classes:-1:1
        class_starts[class+1] = class_starts[class]
    end
    class_starts[1] = 1

    for class in 1:n_classes
        _rotateclass!(output, task, plan, truncation, record_terms, record_coeffs, lo, class_starts[class], class_starts[class+1] - 1)
    end
    return output
end


### Rotating a class

# The Pauli strings of a class differ only in the bits that its rotations flip: the low bit of a qubit that they touch
# with X, the high bit of one touched with Y, and both bits of one touched with Z, which flip together. A qubit touched
# with two different Paulis has both of its bits flipped independently. The strings of a class are therefore told apart
# by the low bit of every qubit touched with X or Z, the high bit of every qubit touched with Y, and both bits of every
# qubit touched with two Paulis. These bits, gathered next to each other, are the key of a string in its class. A rotation flips a fixed set of bits of the key, so
# the key of the partner it makes is the key of the string with those bits flipped. The table of a class finds an entry
# by its key: through a slot of its own where the keys are few enough, and through a hash of the key otherwise.
# A class with more such bits than an integer holds is keyed by its Pauli strings themselves.

# the most bits that a key of its own slot can have
const _MAX_DIRECT_KEY_BITS = 20

# the most bits that a key gathered into an integer can have
const _MAX_KEY_BITS = 64

# A coefficient of at most this fraction below the smallest kept one is checked against the truncation when the partner
# it would make is decided, so that rounding cannot hide a partner that the truncation keeps.
const _MAKE_MARGIN = 1 - 1e-12

"""
    _rotateclass!(output, task, plan, truncation, record_terms, record_coeffs, lo, first, last)

Rotates the class of the records `class_records[first:last]`, counted from `lo`, one rotation after the other, and writes the entries that are left to `output`.
"""
function _rotateclass!(output, task, plan, truncation, record_terms::Vector{TT}, record_coeffs::Vector{CT}, lo::Int,
    first::Int, last::Int) where {TT,CT}

    first_pstr = record_terms[lo-1+task.class_records[first]]
    n_rotations = _classrotations!(task.orbit_rotations, plan, first_pstr)
    key_bits = _keybits(plan, first_pstr)
    n_key_bits = sum(count_ones, _limbs(key_bits))

    # the kernel is compiled for keys of each type, behind this barrier
    if n_key_bits <= _MAX_KEY_BITS
        _rotatekeyedclass!(output, task, plan, truncation, record_terms, record_coeffs, lo, first, last, n_rotations,
            task.entry_keys, key_bits, n_key_bits)
    else
        _rotatekeyedclass!(output, task, plan, truncation, record_terms, record_coeffs, lo, first, last, n_rotations,
            task.entry_terms, nothing, n_key_bits)
    end
    return output
end

function _rotatekeyedclass!(output, task, plan, truncation, record_terms::Vector{TT}, record_coeffs::Vector{CT}, lo::Int,
    first::Int, last::Int, n_rotations::Int, entry_keys::Vector{K}, key_bits, n_key_bits::Int) where {TT,CT,K}

    _openclass!(task, last - first + 1, K === UInt64 && n_key_bits <= _MAX_DIRECT_KEY_BITS, n_key_bits)
    _addmembers!(task, entry_keys, key_bits, record_terms, record_coeffs, lo, first, last)

    # without a limit on the weight, the weights are not counted
    limits_weight = _limitsweight(truncation)
    for step in 1:n_rotations
        rotation = Int(task.orbit_rotations[step])
        key_mask = _keyof(plan.masks[rotation], key_bits)
        if limits_weight
            _applytoclass!(task, plan, truncation, entry_keys, key_mask, rotation, Int32(step), CT, Val(true))
        else
            _applytoclass!(task, plan, truncation, entry_keys, key_mask, rotation, Int32(step), CT, Val(false))
        end
    end

    _emitclass!(output, task)
    _closeclass!(task, entry_keys)
    return output
end

# The bits that tell the strings of the class of `pstr` apart, on the qubits that the rotations anticommuting with `pstr`
# touch: the low bit where they touch with X or Z, the high bit where they touch with Y, and both where with two Paulis.
@inline function _keybits(plan, pstr)
    touched_x, touched_y, touched_z = _touchedqubits(plan, pstr)
    mixed = (touched_x & touched_y) | (touched_x & touched_z) | (touched_y & touched_z)
    return touched_x | touched_z | mixed | _shiftup(touched_y | mixed, 1)
end

# the key of `pstr` in its class: the key bits gathered into an integer, or the Pauli string itself
@inline _keyof(pstr, key_bits) = first(_compressbits(pstr, key_bits))
@inline _keyof(pstr, ::Nothing) = pstr

# The rotation mixes every two entries that its generator turns into each other, and every entry without a present
# partner keeps cos θ of its coefficient and makes the partner, if the truncation keeps that. Most entries do no more
# than keep cos θ, so the entries that do more are listed first, and those that only keep cos θ are scaled after them.
# A pair is listed at the entry whose key has the lowest bit of `key_mask` clear. Entries made by the rotation come after
# the others and are not visited.
function _applytoclass!(task, plan, truncation, entry_keys::Vector{K}, key_mask::K, rotation::Int, step::Int32, ::Type{CT},
    ::Val{LimitsWeight}) where {K,CT,LimitsWeight}

    mask = plan.masks[rotation]
    cos_val = plan.cosines[rotation]
    sin_val = plan.sines[rotation]
    signs = _signsfor(CT, plan.signs)
    local_mask = Int(plan.local_mask)
    lower_bit = _lowestbit(key_mask)

    min_coeff_to_make = _mincoefftomake(truncation, sin_val)

    # the Paulis on the qubits of the rotation are read from the words that hold them
    first_qind, second_qind = plan.qinds[rotation]
    first_bit = 2 * (first_qind - 1)
    second_bit = 2 * (max(second_qind, 1) - 1)
    reads_second = second_qind != 0

    # nothing grows while the rotation visits the entries
    n_before = task.n_entries
    _prepareentries!(task, entry_keys, n_before)
    entry_terms = task.entry_terms
    entry_coeffs = task.entry_coeffs
    entry_present = task.entry_present
    entry_steps = task.entry_steps
    events = task.events
    event_partners = task.event_partners
    slots = task.class_slots
    slot_mask = task.class_table_length - 1
    hash_shift = task.class_hash_shift

    # the entries that the rotation does more to than keeping cos θ, without branching on them
    n_events = 0
    @inbounds for entry in 1:n_before
        key = entry_keys[entry]
        partner, _ = _findentry(slots, entry_keys, slot_mask, hash_shift, key ⊻ key_mask)
        present = entry_present[entry]
        partner_present = (partner != 0) & entry_present[max(partner, 1)]
        is_lower_of_pair = present & partner_present & iszero(key & lower_bit)
        makes_partner = present & !partner_present & (abs(entry_coeffs[entry]) >= min_coeff_to_make)
        events[n_events+1] = entry % Int32
        event_partners[n_events+1] = partner % Int32
        n_events += is_lower_of_pair | makes_partner
    end

    n_entries = n_before
    @inbounds for event in 1:n_events
        entry = Int(events[event])
        partner = Int(event_partners[event])
        pstr = entry_terms[entry]
        partner_pstr = pstr ⊻ mask

        paulis = ((PropagationBase._wordat(pstr, first_bit) >> (first_bit & 63)) & 0x03) % Int
        if reads_second
            paulis |= (((PropagationBase._wordat(pstr, second_bit) >> (second_bit & 63)) & 0x03) % Int) << 2
        end
        sign_to_partner = signs[(paulis&15)+1]
        sign_from_partner = signs[((paulis⊻local_mask)&15)+1]

        # an entry that was truncated holds zero
        coeff = entry_coeffs[entry]
        partner_coeff = ifelse(partner == 0, zero(CT), entry_coeffs[max(partner, 1)])
        new_coeff = mergefunc(coeff * cos_val, partner_coeff * sin_val * sign_from_partner)
        new_partner_coeff = mergefunc(partner_coeff * cos_val, coeff * sin_val * sign_to_partner)

        keep = !_istruncatedin(truncation, pstr, new_coeff, Val(LimitsWeight))
        keep_partner = !_istruncatedin(truncation, partner_pstr, new_partner_coeff, Val(LimitsWeight))
        entry_coeffs[entry] = ifelse(keep, new_coeff, zero(CT))
        entry_present[entry] = keep
        entry_steps[entry] = step

        if partner != 0
            entry_coeffs[partner] = ifelse(keep_partner, new_partner_coeff, zero(CT))
            entry_present[partner] = keep_partner
            entry_steps[partner] = step
        elseif keep_partner
            partner_key = entry_keys[entry] ⊻ key_mask
            _, slot = _findentry(slots, entry_keys, slot_mask, hash_shift, partner_key)
            n_entries += 1
            entry_terms[n_entries] = partner_pstr
            entry_keys[n_entries] = partner_key
            entry_coeffs[n_entries] = new_partner_coeff
            entry_present[n_entries] = true
            entry_steps[n_entries] = step
            slots[slot+1] = n_entries % Int32
        end
    end

    # every other present entry keeps cos θ of its coefficient
    @inbounds for entry in 1:n_before
        coeff = entry_coeffs[entry]
        scales = entry_present[entry] & (entry_steps[entry] != step)
        new_coeff = coeff * cos_val
        keep = !_istruncatedin(truncation, entry_terms[entry], new_coeff, Val(LimitsWeight))
        entry_coeffs[entry] = ifelse(scales, ifelse(keep, new_coeff, zero(CT)), coeff)
        entry_present[entry] = ifelse(scales, keep, entry_present[entry])
    end

    task.n_entries = n_entries
    return task
end

# The signs as the real type of floating-point coefficients, which they multiply as the integers do, so that they are not
# converted for every entry.
_signsfor(::Type{CT}, signs) where {CT<:Union{AbstractFloat,Complex{<:AbstractFloat}}} = map(sign -> convert(real(CT), sign), signs)
_signsfor(::Type, signs) = signs

# the truncation of an entry, with the weight counted only where it is limited
@inline function _istruncatedin(truncation, pstr, coeff, ::Val{LimitsWeight}) where {LimitsWeight}
    if LimitsWeight && countweight(pstr) > truncation.max_weight
        return true
    end
    return @inline truncation.truncfunc(pstr, coeff)
end

# The smallest coefficient that can make a partner the truncation keeps. Without a smallest kept coefficient, any can.
function _mincoefftomake(truncation, sin_val)
    if iszero(truncation.min_abs_coeff)
        return zero(truncation.min_abs_coeff)
    end
    return _MAKE_MARGIN * truncation.min_abs_coeff / abs(sin_val)
end

# the lowest set bit of a key
@inline _lowestbit(key::UInt64) = key & (~key + one(UInt64))

@inline function _lowestbit(pstr::TT) where {TT}
    limbs = _limbs(pstr)
    for limb_index in eachindex(limbs)
        if limbs[limb_index] != 0
            return _shiftup(one(TT), 64 * (limb_index - 1) + trailing_zeros(limbs[limb_index]))
        end
    end
    return zero(TT)
end


### The table of a class

# A hash for the table of a class: the sum of the limbs, each times an odd factor, whose highest bits pick the slot.
@inline function _classhash(pstr)
    limbs = _limbs(pstr)
    folded = zero(UInt64)
    for i in eachindex(limbs)
        folded += limbs[i] * _foldfactor(i)
    end
    return folded
end

@inline _classhash(key::UInt64) = key * 0x9e3779b97f4a7c15

# An empty table for a class of `n_members` Pauli strings. Where every key has a slot of its own, the slot of a key is the
# key; otherwise the table is filled to at most an eighth through a hash of the keys. The slots are all zero outside a
# class: a class clears the slots it used when it closes, and a class that did not close leaves them to be cleared here.
function _openclass!(task, n_members::Int, has_direct_slots::Bool, n_key_bits::Int)
    if has_direct_slots
        table_length = 1 << n_key_bits
        task.class_hash_shift = 0
    else
        table_length = max(16, nextpow(2, 8 * n_members))
        task.class_hash_shift = 64 - trailing_zeros(table_length)
    end
    slots = task.class_slots
    n_zeroed = length(slots)
    if task.class_is_open
        fill!(slots, zero(Int32))
    end
    if table_length > n_zeroed
        resize!(slots, max(table_length, 2 * n_zeroed))
        fill!(view(slots, n_zeroed+1:length(slots)), zero(Int32))
    end
    task.class_is_open = true
    task.class_table_length = table_length
    task.n_entries = 0
    return task
end

# clears the slots that the class used
function _closeclass!(task, entry_keys::Vector)
    slots = task.class_slots
    if task.class_hash_shift == 0
        for entry in 1:task.n_entries
            slots[(entry_keys[entry]%Int)+1] = zero(Int32)
        end
    else
        fill!(view(slots, 1:task.class_table_length), zero(Int32))
    end
    task.class_is_open = false
    return task
end

# The entry with the key `key`, or 0, and the slot where it is or would be added.
@inline function _findentry(slots::Vector{Int32}, entry_keys::Vector{UInt64}, slot_mask::Int, hash_shift::Int, key::UInt64)
    if hash_shift == 0
        slot = key % Int
        return (@inbounds slots[slot+1]) % Int, slot
    end
    return _probeslots(slots, entry_keys, slot_mask, hash_shift, key)
end

@inline _findentry(slots::Vector{Int32}, entry_keys::Vector, slot_mask::Int, hash_shift::Int, key) =
    _probeslots(slots, entry_keys, slot_mask, hash_shift, key)

# The callers give slots at least the length of the table and keys for every entry in it.
@inline function _probeslots(slots::Vector{Int32}, entry_keys::Vector, slot_mask::Int, hash_shift::Int, key)
    slot = (_classhash(key) >> (hash_shift & 63)) % Int
    while true
        entry = (@inbounds slots[slot+1]) % Int
        if entry == 0 || @inbounds(entry_keys[entry]) == key
            return entry, slot
        end
        slot = (slot + 1) & slot_mask
    end
end

# The records `class_records[first:last]`, counted from `lo`, as the entries of the table of their class, where a Pauli
# string that comes twice is added up.
function _addmembers!(task, entry_keys::Vector, key_bits, record_terms::Vector{TT}, record_coeffs::Vector{CT}, lo::Int,
    first::Int, last::Int) where {TT,CT}

    class_records = task.class_records
    _prepareentries!(task, entry_keys, last - first + 1)
    entry_terms = task.entry_terms
    entry_coeffs = task.entry_coeffs
    entry_present = task.entry_present
    entry_steps = task.entry_steps
    slots = task.class_slots
    slot_mask = task.class_table_length - 1
    hash_shift = task.class_hash_shift

    n_entries = task.n_entries
    for index in first:last
        i = lo - 1 + class_records[index]
        pstr = record_terms[i]
        key = _keyof(pstr, key_bits)
        entry, slot = _findentry(slots, entry_keys, slot_mask, hash_shift, key)
        if entry == 0
            n_entries += 1
            entry_terms[n_entries] = pstr
            entry_keys[n_entries] = key
            entry_coeffs[n_entries] = record_coeffs[i]
            entry_present[n_entries] = true
            entry_steps[n_entries] = zero(Int32)
            slots[slot+1] = n_entries % Int32
        else
            entry_coeffs[entry] = mergefunc(entry_coeffs[entry], record_coeffs[i])
        end
    end
    task.n_entries = n_entries
    return task
end

# Room for twice `n_entries` entries and their events, and a hashed table that they fill to at most a quarter, so that
# a rotation can add an entry for every entry it visits without anything growing.
function _prepareentries!(task, entry_keys::Vector, n_entries::Int)
    n_room = 2 * n_entries + 1
    if n_room > min(length(task.entry_terms), length(entry_keys), length(task.entry_coeffs), length(task.entry_present),
        length(task.entry_steps), length(task.events), length(task.event_partners))
        _ensurelength!(task.entry_terms, n_room)
        _ensurelength!(entry_keys, n_room)
        _ensurelength!(task.entry_coeffs, n_room)
        _ensurelength!(task.entry_present, n_room)
        _ensurelength!(task.entry_steps, n_room)
        _ensurelength!(task.events, n_room)
        _ensurelength!(task.event_partners, n_room)
    end
    if task.class_hash_shift != 0 && task.class_table_length < 8 * n_entries
        _growclass!(task, entry_keys, 8 * n_entries)
    end
    _checkentries(task, entry_keys, n_room)
    return task
end

# a hashed table of at least `n_slots` slots with the entries that are there
function _growclass!(task, entry_keys::Vector, n_slots::Int)
    table_length = nextpow(2, n_slots)
    slots = task.class_slots
    n_zeroed = length(slots)
    if table_length > n_zeroed
        resize!(slots, max(table_length, 2 * n_zeroed))
    end
    fill!(view(slots, 1:length(slots)), zero(Int32))
    slot_mask = table_length - 1
    hash_shift = 64 - trailing_zeros(table_length)

    for entry in 1:task.n_entries
        slot = Int(_classhash(entry_keys[entry]) >> hash_shift)
        while slots[slot+1] != 0
            slot = (slot + 1) & slot_mask
        end
        slots[slot+1] = entry
    end
    task.class_table_length = table_length
    task.class_hash_shift = hash_shift
    return task
end

# the loops over the entries of a class index its arrays without bounds checks
function _checkentries(task, entry_keys::Vector, n_entries::Int)
    if n_entries > min(length(task.entry_terms), length(entry_keys), length(task.entry_coeffs), length(task.entry_present),
           length(task.entry_steps), length(task.events), length(task.event_partners)) ||
       task.class_table_length > length(task.class_slots) || !ispow2(task.class_table_length) ||
       (task.class_hash_shift != 0 && task.class_hash_shift != 64 - trailing_zeros(task.class_table_length))
        throw(ArgumentError("the $n_entries entries of a class do not fit the workspace"))
    end
    return
end

# the entries of the class that are present
function _emitclass!(output::ArrayOutputs, task)
    n_entries = task.n_entries
    if n_entries > min(length(task.entry_terms), length(task.entry_coeffs), length(task.entry_present))
        throw(ArgumentError("the $n_entries entries of a class do not fit the workspace"))
    end
    output_terms, output_coeffs = _roomtoemit!(output, n_entries)
    n_written = output.n_written

    # every entry is written, and the next one writes over it if it is not present
    @inbounds for entry in 1:n_entries
        output_terms[n_written+1] = task.entry_terms[entry]
        output_coeffs[n_written+1] = task.entry_coeffs[entry]
        n_written += task.entry_present[entry]
    end

    output.n_written = n_written
    return
end

function _emitclass!(output::Union{AbstractTermSum,ZoneOutputs}, task)
    for entry in 1:task.n_entries
        if task.entry_present[entry]
            _emit!(output, task.entry_terms[entry], task.entry_coeffs[entry])
        end
    end
    return
end
