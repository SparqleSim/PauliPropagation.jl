###
##
# The rotations of a sublayer on an orbit of many stages.
# Such an orbit has more Pauli strings than a block of coefficients should hold, and the truncations keep few of them.
# Only the Pauli strings that come up are kept, as entries that a small table finds by their coordinate.
##
###

"""
    _transformlongorbit!(sink, task, plan, truncation, record_terms, record_coeffs, record_labels, first_record)

Applies the rotations of the sublayer to the orbit of the records that are linked from `first_record` on, and writes its Pauli strings to `sink`.
"""
function _transformlongorbit!(sink, task::TaskWorkspace{TT,CT}, plan::SubLayerPlan{TT}, truncation::LayerTruncation,
    record_terms::Vector{TT}, record_coeffs::Vector{CT}, record_labels::Vector{Int}, first_record::Int) where {TT,CT}

    orbit = task.long_orbit
    positions = task.positions
    representative = record_terms[first_record]
    n_stages = _orbitrotations!(positions, plan, representative)
    if n_stages > _MAX_STAGES
        _throwwrongblock()
    end

    _emptyorbit!(orbit)
    record_index = first_record
    while record_index != 0
        coordinate = _coordinate(record_labels[record_index])
        entry = _findentry(orbit, coordinate)
        if entry == 0
            pstr = _termat(plan, positions, n_stages, representative, coordinate)
            entry = _addentry!(orbit, coordinate, pstr, _weightiflimited(truncation, pstr))
        end

        if orbit.present[entry]
            orbit.coeffs[entry] = mergefunc(orbit.coeffs[entry], record_coeffs[record_index])
        else
            orbit.coeffs[entry] = record_coeffs[record_index]
            orbit.present[entry] = true
        end
        record_index = task.next_records[record_index-task.first_partition_record+1]
    end

    for stage in 1:n_stages
        bit = _stagebit(positions, n_stages, stage, plan.order)
        position = positions[bit+1]
        _applystage!(orbit, plan, truncation, stage, bit, position)
    end

    for entry in 1:orbit.n_entries
        if orbit.present[entry]
            _emit!(sink, orbit.terms[entry], orbit.coeffs[entry])
        end
    end
    return sink
end

# One rotation on the entries of an orbit. The rotation pairs every entry with the one whose coordinate differs in `bit`.
function _applystage!(orbit::LongOrbit{TT,CT}, plan::SubLayerPlan{TT}, truncation::LayerTruncation,
    stage::Int, bit::Int, position::Integer) where {TT,CT}

    cos_val = plan.cosines[position]
    sin_val = plan.sines[position]
    mask = plan.masks[position]
    weight_changes = _weightchangesiflimited(truncation, plan)
    coordinate_bit = one(UInt64) << bit

    # the entries that this stage adds are changed along with the entry they pair up with
    for entry in 1:orbit.n_entries
        if orbit.stages[entry] == stage || !orbit.present[entry]
            continue
        end

        # the Paulis on the qubits of the rotation are those of the entry, which other rotations may have changed
        coordinate = orbit.coordinates[entry]
        paulis = _localpaulis(plan, orbit.terms[entry], position)
        partner = _findentry(orbit, coordinate ⊻ coordinate_bit)
        if partner == 0
            partner_weight = orbit.weights[entry] + weight_changes[paulis+1]
            partner = _addentry!(orbit, coordinate ⊻ coordinate_bit, orbit.terms[entry] ⊻ mask, partner_weight)
        end

        # the lower entry of a pair has the bit unset
        sign_from_entry = plan.signs[paulis+1]
        sign_from_partner = plan.signs[(paulis⊻plan.local_mask)+1]
        lower, upper, sign_from_lower, sign_from_upper = if iszero(coordinate & coordinate_bit)
            entry, partner, sign_from_entry, sign_from_partner
        else
            partner, entry, sign_from_partner, sign_from_entry
        end
        lower_coeff = orbit.coeffs[lower]
        upper_coeff = orbit.coeffs[upper]
        new_lower_coeff = mergefunc(lower_coeff * cos_val, upper_coeff * sin_val * sign_from_upper)
        new_upper_coeff = mergefunc(upper_coeff * cos_val, lower_coeff * sin_val * sign_from_lower)

        _setentry!(orbit, lower, new_lower_coeff, stage, truncation)
        _setentry!(orbit, upper, new_upper_coeff, stage, truncation)
    end
    return orbit
end

function _setentry!(orbit::LongOrbit{TT,CT}, entry::Int, coeff, stage::Int, truncation::LayerTruncation) where {TT,CT}
    if _istruncated(truncation, orbit.terms[entry], orbit.weights[entry], coeff)
        orbit.coeffs[entry] = zero(CT)
        orbit.present[entry] = false
    else
        orbit.coeffs[entry] = coeff
        orbit.present[entry] = true
    end
    orbit.stages[entry] = stage
    return orbit
end

# the Pauli string of the orbit at `coordinate`
function _termat(plan::SubLayerPlan{TT}, positions::Vector{Int32}, n_stages::Int, representative::TT, coordinate::UInt64) where {TT}
    pstr = representative
    for bit in 0:n_stages-1
        if isodd(coordinate >> bit)
            pstr ⊻= plan.masks[positions[bit+1]]
        end
    end
    return pstr
end


### The entries of an orbit

@inline _slotof(orbit::LongOrbit, coordinate::UInt64) = Int(_mixbits(coordinate) & (length(orbit.slots) - 1)) + 1

# the entry of `coordinate`, or 0 if there is none
function _findentry(orbit::LongOrbit, coordinate::UInt64)
    slots = orbit.slots
    slot = _slotof(orbit, coordinate)
    while true
        entry = Int(slots[slot])
        if entry == 0 || orbit.coordinates[entry] == coordinate
            return entry
        end
        slot = (slot & (length(slots) - 1)) + 1
    end
end

# a new entry, which is not present until it is given a coefficient
function _addentry!(orbit::LongOrbit{TT,CT}, coordinate::UInt64, pstr::TT, weight::Int) where {TT,CT}
    entry = orbit.n_entries + 1
    orbit.n_entries = entry
    if entry > length(orbit.coordinates)
        n_room = max(64, 2 * length(orbit.coordinates))
        resize!(orbit.coordinates, n_room)
        resize!(orbit.terms, n_room)
        resize!(orbit.coeffs, n_room)
        resize!(orbit.weights, n_room)
        resize!(orbit.present, n_room)
        resize!(orbit.stages, n_room)
    end

    orbit.coordinates[entry] = coordinate
    orbit.terms[entry] = pstr
    orbit.coeffs[entry] = zero(CT)
    orbit.weights[entry] = weight
    orbit.present[entry] = false
    orbit.stages[entry] = 0

    # the table stays at most half full
    if 2 * entry > length(orbit.slots)
        resize!(orbit.slots, 2 * length(orbit.slots))
        fill!(orbit.slots, zero(Int32))
        for earlier_entry in 1:entry-1
            _fillslot!(orbit, earlier_entry)
        end
    end
    _fillslot!(orbit, entry)
    return entry
end

function _fillslot!(orbit::LongOrbit, entry::Int)
    slots = orbit.slots
    slot = _slotof(orbit, orbit.coordinates[entry])
    while slots[slot] != 0
        slot = (slot & (length(slots) - 1)) + 1
    end
    slots[slot] = entry
    return orbit
end

# Only the slots of the entries are emptied, so that a table that has grown once costs no more for the small orbits after it.
function _emptyorbit!(orbit::LongOrbit)
    slots = orbit.slots
    if 4 * orbit.n_entries > length(slots)
        fill!(slots, zero(Int32))
    else
        # every entry is found before any slot is emptied, since an empty slot would end the search for the entries past it
        for entry in 1:orbit.n_entries
            slot = _slotof(orbit, orbit.coordinates[entry])
            while slots[slot] != entry
                slot = (slot & (length(slots) - 1)) + 1
            end
            orbit.stages[entry] = slot
        end
        for entry in 1:orbit.n_entries
            slots[orbit.stages[entry]] = zero(Int32)
        end
    end
    orbit.n_entries = 0
    return orbit
end
