###
##
# The kernels that rotate one class, one rotation after the other, and write what the truncations keep to a sink: a
# class of few distinguishing bits densely, with an entry for every term it can hold, and any other sparsely, with an
# entry for each of its terms. What a task rotates the classes in comes last.
##
###


### Dense classes

# A class of few distinguishing bits rotated densely. A class of n distinguishing bits has at most 2^n terms, and the
# dense class holds an entry for each of those, at the coordinates of the term. A rotation changes the coordinates of a
# term by those of its gate mask, with XOR, so it pairs every entry with the entry whose index differs by them, and
# mixes the coefficients of every pair with at least one entry present. Which entries are present is kept as the bits of
# one word, so that a rotation visits the pairs with an entry present.

# for the bit b below 6 of the coordinates, the positions in a word of the entries without bit b
const _LOWER_POSITIONS = (0x5555555555555555, 0x3333333333333333, 0x0f0f0f0f0f0f0f0f, 0x00ff00ff00ff00ff, 0x0000ffff0000ffff, 0x00000000ffffffff)

# the entry of `term` in its dense class: its coordinates
@inline _denseentry(term, distinguishing_bits) = _classcoordinates(term, distinguishing_bits) % Int

# Moves the bit at every position p of a word to p ⊻ low_mask, for low_mask below 64, one set bit of low_mask at a time:
# a rotation flips the same bits of the coordinates of every entry, so a layer takes the same few steps for every word.
@inline function _xorpositions(word::UInt64, low_mask::Int)
    bits = low_mask & 63
    while bits != 0
        j = trailing_zeros(bits)
        bits &= bits - 1
        span = 1 << j
        lower = _LOWER_POSITIONS[j+1]
        word = ((word >> span) & lower) | ((word & lower) << span)
    end
    return word
end

# a dense class: a coefficient and a term for every entry
struct DenseScratch{TT,CT}
    coeffs::Vector{CT}
    terms::Vector{TT}
end

DenseScratch{TT,CT}() where {TT,CT} = DenseScratch{TT,CT}(CT[], TT[])

"""
    _rotatedense!(sink, dense, plan, truncfunc, class_terms, class_coeffs, rotations, distinguishing_bits, n_distinguishing_bits)

Rotates the class of the terms `class_terms` with the coefficients `class_coeffs` densely, with `2^n_distinguishing_bits` entries, by the `rotations` one after the other, and writes what the truncations keep to `sink`.
"""
function _rotatedense!(sink, dense, plan, truncfunc::F, class_terms, class_coeffs, rotations, distinguishing_bits::TT, n_distinguishing_bits::Int) where {F,TT}
    n_entries = 1 << n_distinguishing_bits
    coeffs = _ensurecapacity!(dense.coeffs, n_entries)
    dense_terms = _ensurecapacity!(dense.terms, n_entries)

    present = zero(UInt64)
    for index in eachindex(class_terms, class_coeffs)
        term = class_terms[index]
        entry = _denseentry(term, distinguishing_bits)
        bit = one(UInt64) << entry
        if present & bit != 0
            coeffs[entry+1] = mergefunc(coeffs[entry+1], class_coeffs[index])
        else
            coeffs[entry+1] = class_coeffs[index]
            dense_terms[entry+1] = term
            present |= bit
        end
    end
    for step in eachindex(rotations)
        rotation = Int(rotations[step])
        entry_mask = _denseentry(plan.gate_masks[rotation], distinguishing_bits)
        present = _rotatedenseby(sink, present, coeffs, dense_terms, n_entries, entry_mask, plan, rotation, truncfunc)
    end

    # the entries that are present
    while present != 0
        entry = trailing_zeros(present)
        present &= present - one(UInt64)
        _emit!(sink, dense_terms[entry+1], coeffs[entry+1])
    end
    return sink
end

# Rotates the pairs of the dense class: the entries at `lower_positions` and their partners, whose presence `upper_present`
# holds at the positions of the lower entries. Pairs with both entries present mix both
# coefficients. An entry whose partner is absent keeps cos θ of its coefficient and makes the partner only if its
# coefficient can reach the truncation threshold. Returns which lower entries and which upper entries the truncations
# keep, both at the positions of the lower entries.
@inline function _rotatepairs!(sink, coeffs::Vector{CT}, dense_terms::Vector{TT}, lower_present::UInt64, upper_present::UInt64,
    lower_positions::UInt64, entry_mask::Int, gate_mask::TT, cos_val, sin_val, min_coeff_to_make, plan, rotation::Int,
    truncfunc::F) where {F,CT,TT}

    both = lower_present & upper_present & lower_positions
    lower_alone = lower_present & ~upper_present & lower_positions
    upper_alone = upper_present & ~lower_present & lower_positions
    kept_lower = zero(UInt64)
    kept_upper = zero(UInt64)

    @inbounds while both != 0
        position = trailing_zeros(both)
        both &= both - one(UInt64)
        entry = position
        partner = entry ⊻ entry_mask
        term = dense_terms[entry+1]
        partner_term = term ⊻ gate_mask
        sign_to_partner, sign_from_partner = plan.pairsigns(gate_mask, term)
        coeff = coeffs[entry+1]
        partner_coeff = coeffs[partner+1]
        _addgradient!(sink, rotation, coeff, partner_coeff, sign_to_partner, sign_from_partner)
        new_coeff = mergefunc(coeff * cos_val, partner_coeff * sin_val * sign_from_partner)
        new_partner_coeff = mergefunc(partner_coeff * cos_val, coeff * sin_val * sign_to_partner)
        coeffs[entry+1] = new_coeff
        coeffs[partner+1] = new_partner_coeff
        kept_lower |= UInt64(!@inline(truncfunc(term, new_coeff))) << position
        kept_upper |= UInt64(!@inline(truncfunc(partner_term, new_partner_coeff))) << position
    end

    @inbounds while lower_alone != 0
        position = trailing_zeros(lower_alone)
        lower_alone &= lower_alone - one(UInt64)
        entry = position
        coeff = coeffs[entry+1]
        term = dense_terms[entry+1]
        new_coeff = coeff * cos_val
        coeffs[entry+1] = new_coeff
        kept_lower |= UInt64(!@inline(truncfunc(term, new_coeff))) << position
        if abs(coeff) >= min_coeff_to_make
            sign_to_partner, _ = plan.pairsigns(gate_mask, term)
            partner_term = term ⊻ gate_mask
            new_partner_coeff = coeff * sin_val * sign_to_partner
            if !@inline(truncfunc(partner_term, new_partner_coeff))
                partner = entry ⊻ entry_mask
                coeffs[partner+1] = new_partner_coeff
                dense_terms[partner+1] = partner_term
                kept_upper |= one(UInt64) << position
            end
        end
    end

    @inbounds while upper_alone != 0
        position = trailing_zeros(upper_alone)
        upper_alone &= upper_alone - one(UInt64)
        entry = position
        partner = entry ⊻ entry_mask
        partner_coeff = coeffs[partner+1]
        partner_term = dense_terms[partner+1]
        new_partner_coeff = partner_coeff * cos_val
        coeffs[partner+1] = new_partner_coeff
        kept_upper |= UInt64(!@inline(truncfunc(partner_term, new_partner_coeff))) << position
        if abs(partner_coeff) >= min_coeff_to_make
            term = partner_term ⊻ gate_mask
            _, sign_from_partner = plan.pairsigns(gate_mask, term)
            new_coeff = partner_coeff * sin_val * sign_from_partner
            if !@inline(truncfunc(term, new_coeff))
                coeffs[entry+1] = new_coeff
                dense_terms[entry+1] = term
                kept_lower |= one(UInt64) << position
            end
        end
    end

    return kept_lower, kept_upper
end

# One rotation on the dense class, whose presence is one word. Returns which entries are present after it.
@inline function _rotatedenseby(sink, present::UInt64, coeffs::Vector{CT}, dense_terms::Vector{TT}, n_entries::Int, entry_mask::Int, plan,
    rotation::Int, truncfunc::F) where {F,CT,TT}

    if !(0 < entry_mask < n_entries <= 64) || n_entries > min(length(coeffs), length(dense_terms))
        throw(ArgumentError("the dense class does not fit the workspace"))
    end
    sin_val = plan.sines[rotation]
    upper_present = _xorpositions(present, entry_mask)
    lower_positions = _LOWER_POSITIONS[trailing_zeros(entry_mask)+1]
    kept_lower, kept_upper = _rotatepairs!(sink, coeffs, dense_terms, present, upper_present, lower_positions, entry_mask, plan.gate_masks[rotation],
        plan.cosines[rotation], sin_val, plan.min_coeffs_to_make[rotation], plan, rotation, truncfunc)
    return kept_lower | _xorpositions(kept_upper, entry_mask)
end


### Sparse classes

# A class of many distinguishing bits rotated sparsely, with an entry for each of its terms. A rotation changes the
# coordinates of a term by those of its gate mask, with XOR, so the coordinates of the partner it makes are those of the
# term with these bits flipped. An entry is found by its coordinates: through a slot of its own where the distinguishing
# bits are few enough, and through a hash of the coordinates otherwise. The entries of a class with more distinguishing
# bits than an integer holds are found by their terms themselves.

# the most distinguishing bits of a class whose coordinates each have a slot of their own
const _MAX_DIRECT_DISTINGUISHING_BITS = 20

# A sparse class: the terms of its entries, their coordinates and coefficients, whether they are present, and the last
# rotation that mixed them, found through the first `n_slots` of `slots`: at their coordinates where `hash_shift` is 0,
# and otherwise by the highest bits of the hash of their coordinates from `hash_shift` on. The entries that a rotation
# mixes or that make a partner, and their partners.
mutable struct SparseScratch{TT,CT}
    entry_terms::Vector{TT}
    entry_coordinates::Vector{UInt64}
    entry_coeffs::Vector{CT}
    entry_present::Vector{Bool}
    entry_steps::Vector{Int32}
    n_entries::Int
    slots::Vector{Int32}
    n_slots::Int
    hash_shift::Int
    is_open::Bool
    events::Vector{Int32}
    event_partners::Vector{Int32}
end

SparseScratch{TT,CT}() where {TT,CT} = SparseScratch{TT,CT}(TT[], UInt64[], CT[], Bool[], Int32[], 0, Int32[], 0, 64, false, Int32[], Int32[])

"""
    _rotatesparse!(sink, sparse, plan, truncfunc, class_terms, class_coeffs, rotations, entry_coordinates, distinguishing_bits, n_distinguishing_bits)

Rotates the class of the terms `class_terms` with the coefficients `class_coeffs` sparsely, by the `rotations` one
after the other, and writes what the truncations keep to `sink`.
The entries are found by `entry_coordinates`, the coordinates of their terms, or the terms themselves.
"""
function _rotatesparse!(sink, sparse, plan, truncfunc::F, class_terms, class_coeffs::AbstractVector{CT}, rotations,
    entry_coordinates::Vector{K}, distinguishing_bits, n_distinguishing_bits::Int) where {F,CT,K}

    _openclass!(sparse, length(class_terms), K === UInt64 && n_distinguishing_bits <= _MAX_DIRECT_DISTINGUISHING_BITS, n_distinguishing_bits)
    _addmembers!(sparse, entry_coordinates, distinguishing_bits, class_terms, class_coeffs)

    for step in eachindex(rotations)
        rotation = Int(rotations[step])
        _applytoclass!(sink, sparse, plan, truncfunc, entry_coordinates, _classcoordinates(plan.gate_masks[rotation], distinguishing_bits), rotation, Int32(step), CT)
    end

    _emitclass!(sink, sparse)
    _closeclass!(sparse, entry_coordinates)
    return sink
end

# The coordinates of `term` in its class: its distinguishing bits gathered into an integer, every distinguishing bit
# that is set moved to the number of distinguishing bits below it. Without the distinguishing bits, for a class whose
# coordinates do not fit an integer, the term itself.
@inline function _classcoordinates(term::TT, distinguishing_bits::TT) where {TT<:Integer}
    set_limbs = _limbs(term & distinguishing_bits)
    distinguishing_limbs = _limbs(distinguishing_bits)
    coordinates = zero(UInt64)
    n_below = 0
    for limb_index in eachindex(distinguishing_limbs)
        set_bits = set_limbs[limb_index]
        while set_bits != 0
            lowest = set_bits & (~set_bits + one(UInt64))
            set_bits &= set_bits - one(UInt64)
            coordinates |= one(UInt64) << ((n_below + count_ones(distinguishing_limbs[limb_index] & (lowest - one(UInt64)))) & 63)
        end
        n_below += count_ones(distinguishing_limbs[limb_index])
    end
    return coordinates
end

@inline _classcoordinates(term, ::Nothing) = term

# The rotation mixes every two entries that it turns into each other, and every entry without a present partner keeps
# cos θ of its coefficient and makes the partner, if the truncation keeps that. Most entries do no more than keep cos θ,
# so the entries that do more are listed first, and those that only keep cos θ are scaled after them. A pair is listed
# at the entry whose coordinates have the lowest bit of `mask_coordinates` clear. Entries made by the rotation come
# after the others and are not visited.
function _applytoclass!(sink, sparse, plan, truncfunc::F, entry_coordinates::Vector{K}, mask_coordinates::K, rotation::Int, step::Int32, ::Type{CT}) where {F,K,CT}

    gate_mask = plan.gate_masks[rotation]
    cos_val = plan.cosines[rotation]
    sin_val = plan.sines[rotation]
    lower_bit = _lowestbit(mask_coordinates)

    min_coeff_to_make = plan.min_coeffs_to_make[rotation]

    # nothing grows while the rotation visits the entries
    n_before = sparse.n_entries
    _prepareentries!(sparse, entry_coordinates, n_before)
    entry_terms = sparse.entry_terms
    entry_coeffs = sparse.entry_coeffs
    entry_present = sparse.entry_present
    entry_steps = sparse.entry_steps
    events = sparse.events
    event_partners = sparse.event_partners
    slots = sparse.slots
    slot_mask = sparse.n_slots - 1
    hash_shift = sparse.hash_shift

    # the entries that the rotation does more to than keeping cos θ, without branching on them
    n_events = 0
    @inbounds for entry in 1:n_before
        coordinates = entry_coordinates[entry]
        partner, _ = _findentry(slots, entry_coordinates, slot_mask, hash_shift, coordinates ⊻ mask_coordinates)
        present = entry_present[entry]
        partner_present = (partner != 0) & entry_present[max(partner, 1)]
        is_lower_of_pair = present & partner_present & iszero(coordinates & lower_bit)
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
        is_lower = iszero(entry_coordinates[entry] & lower_bit)
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
            partner_coordinates = entry_coordinates[entry] ⊻ mask_coordinates
            _, slot = _findentry(slots, entry_coordinates, slot_mask, hash_shift, partner_coordinates)
            n_entries += 1
            entry_terms[n_entries] = partner_term
            entry_coordinates[n_entries] = partner_coordinates
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

    sparse.n_entries = n_entries
    return sparse
end

# the lowest set bit of coordinates
@inline _lowestbit(coordinates) = coordinates & (~coordinates + one(coordinates))


### The slots of a sparse class

# A hash for the slots of a sparse class, whose highest bits pick the slot: the hash of a term, or coordinates times an
# odd constant.
@inline _classhash(term) = _hashbits(term)

@inline _classhash(coordinates::UInt64) = coordinates * 0x9e3779b97f4a7c15

# An empty sparse class for `n_members` terms. Where all coordinates have a slot of their own, the slot of an entry is
# its coordinates; otherwise the slots are filled to at most an eighth through a hash of the coordinates. The slots are
# all zero outside a class: a class clears the slots it used when it closes, and a class that did not close leaves them
# to be cleared here.
function _openclass!(sparse, n_members::Int, has_direct_slots::Bool, n_distinguishing_bits::Int)
    if has_direct_slots
        n_slots = 1 << n_distinguishing_bits
        sparse.hash_shift = 0
    else
        n_slots = max(16, nextpow(2, 8 * n_members))
        sparse.hash_shift = 64 - trailing_zeros(n_slots)
    end
    slots = sparse.slots
    n_zeroed = length(slots)
    if sparse.is_open
        fill!(slots, zero(Int32))
    end
    if n_slots > n_zeroed
        _ensurecapacity!(slots, n_slots)
        fill!(view(slots, n_zeroed+1:length(slots)), zero(Int32))
    end
    sparse.is_open = true
    sparse.n_slots = n_slots
    sparse.n_entries = 0
    return sparse
end

# clears the slots that the class used
function _closeclass!(sparse, entry_coordinates::Vector)
    slots = sparse.slots
    if sparse.hash_shift == 0
        for entry in 1:sparse.n_entries
            slots[(entry_coordinates[entry]%Int)+1] = zero(Int32)
        end
    else
        fill!(view(slots, 1:sparse.n_slots), zero(Int32))
    end
    sparse.is_open = false
    return sparse
end

# The entry with the coordinates `coordinates`, or 0, and the slot where it is or would be added.
@inline function _findentry(slots::Vector{Int32}, entry_coordinates::Vector{UInt64}, slot_mask::Int, hash_shift::Int, coordinates::UInt64)
    if hash_shift == 0
        slot = coordinates % Int
        return (@inbounds slots[slot+1]) % Int, slot
    end
    return _probeslots(slots, entry_coordinates, slot_mask, hash_shift, coordinates)
end

@inline _findentry(slots::Vector{Int32}, entry_coordinates::Vector, slot_mask::Int, hash_shift::Int, coordinates) =
    _probeslots(slots, entry_coordinates, slot_mask, hash_shift, coordinates)

# The callers give at least `n_slots` slots and coordinates for every entry.
@inline function _probeslots(slots::Vector{Int32}, entry_coordinates::Vector, slot_mask::Int, hash_shift::Int, coordinates)
    slot = (_classhash(coordinates) >> (hash_shift & 63)) % Int
    while true
        entry = (@inbounds slots[slot+1]) % Int
        if entry == 0 || @inbounds(entry_coordinates[entry]) == coordinates
            return entry, slot
        end
        slot = (slot + 1) & slot_mask
    end
end

# The terms of the class as its entries, where a term that comes twice is added up.
function _addmembers!(sparse, entry_coordinates::Vector, distinguishing_bits, class_terms, class_coeffs)
    _prepareentries!(sparse, entry_coordinates, length(class_terms))
    entry_terms = sparse.entry_terms
    entry_coeffs = sparse.entry_coeffs
    entry_present = sparse.entry_present
    entry_steps = sparse.entry_steps
    slots = sparse.slots
    slot_mask = sparse.n_slots - 1
    hash_shift = sparse.hash_shift

    n_entries = sparse.n_entries
    for index in eachindex(class_terms, class_coeffs)
        term = class_terms[index]
        coordinates = _classcoordinates(term, distinguishing_bits)
        entry, slot = _findentry(slots, entry_coordinates, slot_mask, hash_shift, coordinates)
        if entry == 0
            n_entries += 1
            entry_terms[n_entries] = term
            entry_coordinates[n_entries] = coordinates
            entry_coeffs[n_entries] = class_coeffs[index]
            entry_present[n_entries] = true
            entry_steps[n_entries] = zero(Int32)
            slots[slot+1] = n_entries % Int32
        else
            entry_coeffs[entry] = mergefunc(entry_coeffs[entry], class_coeffs[index])
        end
    end
    sparse.n_entries = n_entries
    return sparse
end

# Room for twice `n_entries` entries and their events, and hashed slots that they fill to at most a quarter, so that
# a rotation can add an entry for every entry it visits without anything growing.
function _prepareentries!(sparse, entry_coordinates::Vector, n_entries::Int)
    n_room = 2 * n_entries + 1
    entry_arrays = _entryarrays(sparse, entry_coordinates)
    if n_room > minimum(length, entry_arrays)
        for array in entry_arrays
            _ensurecapacity!(array, n_room)
        end
    end
    if sparse.hash_shift != 0 && sparse.n_slots < 8 * n_entries
        _growclass!(sparse, entry_coordinates, 8 * n_entries)
    end
    _checkentries(sparse, entry_coordinates, n_room)
    return sparse
end

# the arrays that hold the entries of a class and the events of a rotation, each with room for as many entries
_entryarrays(sparse, entry_coordinates::Vector) =
    (sparse.entry_terms, entry_coordinates, sparse.entry_coeffs, sparse.entry_present, sparse.entry_steps, sparse.events, sparse.event_partners)

# hashed slots, at least `n_slots` of them, with the entries that are there
function _growclass!(sparse, entry_coordinates::Vector, n_slots::Int)
    n_slots = nextpow(2, n_slots)
    slots = _ensurecapacity!(sparse.slots, n_slots)
    fill!(view(slots, 1:n_slots), zero(Int32))
    slot_mask = n_slots - 1
    hash_shift = 64 - trailing_zeros(n_slots)

    for entry in 1:sparse.n_entries
        slot = Int(_classhash(entry_coordinates[entry]) >> hash_shift)
        while slots[slot+1] != 0
            slot = (slot + 1) & slot_mask
        end
        slots[slot+1] = entry
    end
    sparse.n_slots = n_slots
    sparse.hash_shift = hash_shift
    return sparse
end

# the loops over the entries of a class index its arrays without bounds checks
function _checkentries(sparse, entry_coordinates::Vector, n_entries::Int)
    if n_entries > minimum(length, _entryarrays(sparse, entry_coordinates)) ||
       sparse.n_slots > length(sparse.slots) || !ispow2(sparse.n_slots) ||
       (sparse.hash_shift != 0 && sparse.hash_shift != 64 - trailing_zeros(sparse.n_slots))
        throw(ArgumentError("the $n_entries entries of a class do not fit the workspace"))
    end
    return
end

# the entries of the class that are present
function _emitclass!(sink, sparse)
    for entry in 1:sparse.n_entries
        if sparse.entry_present[entry]
            _emit!(sink, sparse.entry_terms[entry], sparse.entry_coeffs[entry])
        end
    end
    return
end


### What a task rotates the classes in

# the scratch of the `TaskWorkspace` of every task of a pass of a layer of rotations (classes.jl)
struct ClassScratch{TT,CT}
    # the rotations that act on the terms of a class, in the order of the layer
    rotations::Vector{Int32}

    # what the task adds to the gradient of every rotation of the layer in a gradient pass
    gradient::Vector{Float64}

    # what the kernels rotate a class in
    dense::DenseScratch{TT,CT}
    sparse::SparseScratch{TT,CT}
end

ClassScratch{TT,CT}() where {TT,CT} = ClassScratch{TT,CT}(Int32[], Float64[], DenseScratch{TT,CT}(), SparseScratch{TT,CT}())
