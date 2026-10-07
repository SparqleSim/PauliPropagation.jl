###
##
# A class of many key bits rotated in a table of its own. The key bits of a class, as `_keybits` gives them, tell its
# terms apart, and gathered next to each other they are the key of a term in its class. A rotation flips a fixed set of
# bits of the key, so the key of the partner it makes is the key of the term with those bits flipped. The table of a
# class finds an entry by its key: through a slot of its own where the keys are few enough, and through a hash of the
# key otherwise. A class with more such bits than an integer holds is keyed by its terms themselves.
##
###

# the most bits that a key of its own slot can have
const _MAX_DIRECT_KEY_BITS = 20

# The table of a class: the terms of its entries, their keys and coefficients, whether they are present, and the last
# rotation that mixed them, found through the first `table_length` of `slots`: at their key where `hash_shift` is 0, and
# otherwise by the highest bits of the hash of their key from `hash_shift` on. The entries that a rotation mixes or that
# make a partner, and their partners.
mutable struct TableScratch{TT,CT}
    entry_terms::Vector{TT}
    entry_keys::Vector{UInt64}
    entry_coeffs::Vector{CT}
    entry_present::Vector{Bool}
    entry_steps::Vector{Int32}
    n_entries::Int
    slots::Vector{Int32}
    table_length::Int
    hash_shift::Int
    is_open::Bool
    events::Vector{Int32}
    event_partners::Vector{Int32}
end

TableScratch{TT,CT}() where {TT,CT} = TableScratch{TT,CT}(TT[], UInt64[], CT[], Bool[], Int32[], 0, Int32[], 0, 64, false, Int32[], Int32[])

"""
    _rotatetable!(sink, table, plan, truncfunc, class_terms, class_coeffs, rotations, entry_keys, key_bits, n_key_bits)

Rotates the class of the terms `class_terms` with the coefficients `class_coeffs` in `table`, by the `rotations` one
after the other, and writes what the truncations keep to `sink`.
The entries are keyed by `entry_keys`, the key bits of their terms gathered into an integer, or the terms themselves.
"""
function _rotatetable!(sink, table, plan, truncfunc::F, class_terms, class_coeffs::AbstractVector{CT}, rotations,
    entry_keys::Vector{K}, key_bits, n_key_bits::Int) where {F,CT,K}

    _openclass!(table, length(class_terms), K === UInt64 && n_key_bits <= _MAX_DIRECT_KEY_BITS, n_key_bits)
    _addmembers!(table, entry_keys, key_bits, class_terms, class_coeffs)

    for step in eachindex(rotations)
        rotation = Int(rotations[step])
        _applytoclass!(sink, table, plan, truncfunc, entry_keys, _keyof(plan.gate_masks[rotation], key_bits), rotation, Int32(step), CT)
    end

    _emitclass!(sink, table)
    _closeclass!(table, entry_keys)
    return sink
end

# The key of `term` in its class: its key bits gathered into an integer, every key bit that is set moved to the number of
# key bits below it. Without key bits, the term itself.
@inline function _keyof(term::TT, key_bits::TT) where {TT<:Integer}
    set_limbs = _limbs(term & key_bits)
    key_limbs = _limbs(key_bits)
    key = zero(UInt64)
    n_below = 0
    for limb_index in eachindex(key_limbs)
        set_bits = set_limbs[limb_index]
        while set_bits != 0
            lowest = set_bits & (~set_bits + one(UInt64))
            set_bits &= set_bits - one(UInt64)
            key |= one(UInt64) << ((n_below + count_ones(key_limbs[limb_index] & (lowest - one(UInt64)))) & 63)
        end
        n_below += count_ones(key_limbs[limb_index])
    end
    return key
end

@inline _keyof(term, ::Nothing) = term

# The rotation mixes every two entries that it turns into each other, and every entry without a present partner keeps
# cos θ of its coefficient and makes the partner, if the truncation keeps that. Most entries do no more than keep cos θ,
# so the entries that do more are listed first, and those that only keep cos θ are scaled after them. A pair is listed
# at the entry whose key has the lowest bit of `key_mask` clear. Entries made by the rotation come after the others and
# are not visited.
function _applytoclass!(sink, table, plan, truncfunc::F, entry_keys::Vector{K}, key_mask::K, rotation::Int, step::Int32, ::Type{CT}) where {F,K,CT}

    gate_mask = plan.gate_masks[rotation]
    cos_val = plan.cosines[rotation]
    sin_val = plan.sines[rotation]
    lower_bit = _lowestbit(key_mask)

    min_coeff_to_make = plan.min_coeffs_to_make[rotation]

    # nothing grows while the rotation visits the entries
    n_before = table.n_entries
    _prepareentries!(table, entry_keys, n_before)
    entry_terms = table.entry_terms
    entry_coeffs = table.entry_coeffs
    entry_present = table.entry_present
    entry_steps = table.entry_steps
    events = table.events
    event_partners = table.event_partners
    slots = table.slots
    slot_mask = table.table_length - 1
    hash_shift = table.hash_shift

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
        term = entry_terms[entry]
        partner_term = term ⊻ gate_mask

        # the signs are those of the lower term of the pair, which the entry is unless it makes its partner from above
        is_lower = iszero(entry_keys[entry] & lower_bit)
        lower_to_upper, upper_to_lower = plan.pairsigns(gate_mask, ifelse(is_lower, term, partner_term))
        sign_to_partner = ifelse(is_lower, lower_to_upper, upper_to_lower)
        sign_from_partner = ifelse(is_lower, upper_to_lower, lower_to_upper)

        # an entry that was truncated holds zero
        coeff = entry_coeffs[entry]
        partner_coeff = ifelse(partner == 0, zero(CT), entry_coeffs[max(partner, 1)])
        _addgradient!(sink, rotation, coeff, partner_coeff, sign_to_partner, sign_from_partner)
        new_coeff = mergefunc(coeff * cos_val, partner_coeff * sin_val * sign_from_partner)
        new_partner_coeff = mergefunc(partner_coeff * cos_val, coeff * sin_val * sign_to_partner)

        keep = !@inline(truncfunc(term, new_coeff))
        keep_partner = !@inline(truncfunc(partner_term, new_partner_coeff))
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
            entry_terms[n_entries] = partner_term
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
        keep = !@inline(truncfunc(entry_terms[entry], new_coeff))
        entry_coeffs[entry] = ifelse(scales, ifelse(keep, new_coeff, zero(CT)), coeff)
        entry_present[entry] = ifelse(scales, keep, entry_present[entry])
    end

    table.n_entries = n_entries
    return table
end

# the lowest set bit of a key
@inline _lowestbit(key) = key & (~key + one(key))


### The table of a class

# A hash for the table of a class, whose highest bits pick the slot: the hash of a term, or a key times an odd constant.
@inline _classhash(term) = _hashbits(term)

@inline _classhash(key::UInt64) = key * 0x9e3779b97f4a7c15

# An empty table for a class of `n_members` terms. Where every key has a slot of its own, the slot of a key is the key;
# otherwise the table is filled to at most an eighth through a hash of the keys. The slots are all zero outside a class:
# a class clears the slots it used when it closes, and a class that did not close leaves them to be cleared here.
function _openclass!(table, n_members::Int, has_direct_slots::Bool, n_key_bits::Int)
    if has_direct_slots
        table_length = 1 << n_key_bits
        table.hash_shift = 0
    else
        table_length = max(16, nextpow(2, 8 * n_members))
        table.hash_shift = 64 - trailing_zeros(table_length)
    end
    slots = table.slots
    n_zeroed = length(slots)
    if table.is_open
        fill!(slots, zero(Int32))
    end
    if table_length > n_zeroed
        _ensurecapacity!(slots, table_length)
        fill!(view(slots, n_zeroed+1:length(slots)), zero(Int32))
    end
    table.is_open = true
    table.table_length = table_length
    table.n_entries = 0
    return table
end

# clears the slots that the class used
function _closeclass!(table, entry_keys::Vector)
    slots = table.slots
    if table.hash_shift == 0
        for entry in 1:table.n_entries
            slots[(entry_keys[entry]%Int)+1] = zero(Int32)
        end
    else
        fill!(view(slots, 1:table.table_length), zero(Int32))
    end
    table.is_open = false
    return table
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

# The terms of the class as the entries of its table, where a term that comes twice is added up.
function _addmembers!(table, entry_keys::Vector, key_bits, class_terms, class_coeffs)
    _prepareentries!(table, entry_keys, length(class_terms))
    entry_terms = table.entry_terms
    entry_coeffs = table.entry_coeffs
    entry_present = table.entry_present
    entry_steps = table.entry_steps
    slots = table.slots
    slot_mask = table.table_length - 1
    hash_shift = table.hash_shift

    n_entries = table.n_entries
    for index in eachindex(class_terms, class_coeffs)
        term = class_terms[index]
        key = _keyof(term, key_bits)
        entry, slot = _findentry(slots, entry_keys, slot_mask, hash_shift, key)
        if entry == 0
            n_entries += 1
            entry_terms[n_entries] = term
            entry_keys[n_entries] = key
            entry_coeffs[n_entries] = class_coeffs[index]
            entry_present[n_entries] = true
            entry_steps[n_entries] = zero(Int32)
            slots[slot+1] = n_entries % Int32
        else
            entry_coeffs[entry] = mergefunc(entry_coeffs[entry], class_coeffs[index])
        end
    end
    table.n_entries = n_entries
    return table
end

# Room for twice `n_entries` entries and their events, and a hashed table that they fill to at most a quarter, so that
# a rotation can add an entry for every entry it visits without anything growing.
function _prepareentries!(table, entry_keys::Vector, n_entries::Int)
    n_room = 2 * n_entries + 1
    entry_arrays = _entryarrays(table, entry_keys)
    if n_room > minimum(length, entry_arrays)
        for array in entry_arrays
            _ensurecapacity!(array, n_room)
        end
    end
    if table.hash_shift != 0 && table.table_length < 8 * n_entries
        _growclass!(table, entry_keys, 8 * n_entries)
    end
    _checkentries(table, entry_keys, n_room)
    return table
end

# the arrays that hold the entries of a class and the events of a rotation, each with room for as many entries
_entryarrays(table, entry_keys::Vector) =
    (table.entry_terms, entry_keys, table.entry_coeffs, table.entry_present, table.entry_steps, table.events, table.event_partners)

# a hashed table of at least `n_slots` slots with the entries that are there
function _growclass!(table, entry_keys::Vector, n_slots::Int)
    table_length = nextpow(2, n_slots)
    slots = _ensurecapacity!(table.slots, table_length)
    fill!(view(slots, 1:table_length), zero(Int32))
    slot_mask = table_length - 1
    hash_shift = 64 - trailing_zeros(table_length)

    for entry in 1:table.n_entries
        slot = Int(_classhash(entry_keys[entry]) >> hash_shift)
        while slots[slot+1] != 0
            slot = (slot + 1) & slot_mask
        end
        slots[slot+1] = entry
    end
    table.table_length = table_length
    table.hash_shift = hash_shift
    return table
end

# the loops over the entries of a class index its arrays without bounds checks
function _checkentries(table, entry_keys::Vector, n_entries::Int)
    if n_entries > minimum(length, _entryarrays(table, entry_keys)) ||
       table.table_length > length(table.slots) || !ispow2(table.table_length) ||
       (table.hash_shift != 0 && table.hash_shift != 64 - trailing_zeros(table.table_length))
        throw(ArgumentError("the $n_entries entries of a class do not fit the workspace"))
    end
    return
end

# the entries of the class that are present
function _emitclass!(sink, table)
    for entry in 1:table.n_entries
        if table.entry_present[entry]
            _emit!(sink, table.entry_terms[entry], table.entry_coeffs[entry])
        end
    end
    return
end
