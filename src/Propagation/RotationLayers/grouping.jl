###
##
# Grouping the records of a partition by orbit.
# A table finds the orbit of a record by the hash of its representative,
# and the coefficient of the record goes into the block of the orbit, at its coordinate.
##
###

"""
    _grouporbits!(task, sink, truncation, record_terms, record_coeffs, record_labels, lo, hi)

Finds the orbits of the records `lo` to `hi` and adds the coefficient of every record to the block of its orbit, at its coordinate.
The records of an orbit of more stages than a block holds are linked instead.
A Pauli string that anticommutes with no rotation goes to `sink` as it is.
"""
function _grouporbits!(task::TaskWorkspace{TT,CT}, sink, truncation, record_terms::Vector{TT}, record_coeffs::Vector{CT},
    record_labels::Vector{Int}, lo::Int, hi::Int) where {TT,CT}

    n_records = hi - lo + 1
    if !(1 <= lo && hi <= min(length(record_terms), length(record_coeffs), length(record_labels)))
        throw(ArgumentError("the records $lo to $hi are not among the records"))
    end

    table_length = max(16, nextpow(2, 2 * n_records))
    slot_mask = (table_length - 1) % UInt64
    slots = _ensurelength!(task.slots, table_length)
    fill!(view(slots, 1:table_length), zero(Int32))

    orbit_hashes = _ensurelength!(task.orbit_hashes, n_records)
    first_records = _ensurelength!(task.first_records, n_records)
    orbit_blocks = _ensurelength!(task.orbit_blocks, n_records)
    n_blocks = fill!(task.n_blocks, 0)
    n_orbits = 0

    # the records of a partition are numbered from `lo` on, and their links from 1 on
    long_orbit_records = _ensurelength!(task.long_orbit_records, n_records)
    last_records = _ensurelength!(task.last_records, n_records)
    next_records = _ensurelength!(task.next_records, n_records)
    n_long_orbits = 0
    task.first_partition_record = lo

    for i in lo:hi
        label = @inbounds record_labels[i]
        representative = @inbounds record_terms[i]
        n_stages = _nstages(label)

        if n_stages == 0
            coeff = @inbounds record_coeffs[i]
            if !_istruncated(truncation, representative, coeff)
                _emit!(sink, representative, coeff)
            end
            continue
        end

        # a slot is within the table, and an orbit in a slot is one of the orbits found so far
        hashbits = _hashbits(label, representative)
        slot = Int(hashbits & slot_mask) + 1
        orbit = Int(@inbounds slots[slot])
        @inbounds while orbit != 0 && !(orbit_hashes[orbit] == hashbits && record_terms[first_records[orbit]] == representative)
            slot = (slot & (table_length - 1)) + 1
            orbit = Int(slots[slot])
        end

        is_new = orbit == 0
        if is_new
            n_orbits += 1
            orbit = n_orbits
            @inbounds begin
                slots[slot] = orbit
                orbit_hashes[orbit] = hashbits
                first_records[orbit] = i
            end
        end

        if n_stages > _MAX_BLOCK_STAGES
            if is_new
                n_long_orbits += 1
                long_orbit_records[n_long_orbits] = i
            else
                next_records[last_records[orbit]-lo+1] = i
            end
            last_records[orbit] = i
            next_records[i-lo+1] = 0
            continue
        end

        if is_new
            block = n_blocks[n_stages] + 1
            n_blocks[n_stages] = block
            orbit_blocks[orbit] = block
            _newblock!(task, n_stages, block, i)
        end

        block_coeffs = task.block_coeffs[n_stages]
        block_present = task.block_present[n_stages]
        entry = (((@inbounds orbit_blocks[orbit]) - 1) << n_stages) + Int(_coordinate(label)) + 1
        if block_present[entry]
            block_coeffs[entry] = mergefunc(block_coeffs[entry], record_coeffs[i])
        else
            block_coeffs[entry] = record_coeffs[i]
            block_present[entry] = true
        end
    end

    task.n_long_orbits = n_long_orbits
    return task
end

# an empty block for an orbit of `n_stages` stages whose first record is `first_record`
function _newblock!(task::TaskWorkspace{TT,CT}, n_stages::Int, block::Int, first_record::Int) where {TT,CT}
    block_records = task.block_records[n_stages]
    block_coeffs = task.block_coeffs[n_stages]
    block_present = task.block_present[n_stages]

    n_entries = block << n_stages
    if n_entries > length(block_coeffs) || n_entries > length(block_present) || block > length(block_records)
        _ensurelength!(block_records, block)
        _ensurelength!(block_coeffs, n_entries)
        _ensurelength!(block_present, n_entries)
    end

    block_records[block] = first_record
    for entry in n_entries-(1<<n_stages)+1:n_entries
        block_coeffs[entry] = zero(CT)
        block_present[entry] = false
    end
    return task
end
