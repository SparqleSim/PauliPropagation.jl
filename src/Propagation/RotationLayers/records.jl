###
##
# The records of a sublayer, and their partitions.
# The partition of a record is read from the hash of its representative, so that all records of an orbit lie in one partition,
# and a partition is small enough to be worked on within the cache.
##
###

function _partitionbits(n_records::Int, record_bytes::Int)
    n_partitions = cld(n_records * record_bytes, _PARTITION_BYTES)
    if n_partitions <= 1
        return 0
    end
    return min(_MAX_PARTITION_BITS, 8 * sizeof(Int) - leading_zeros(n_partitions - 1))
end

# the highest bits of the hash, where the table of a partition reads the lowest
@inline _partitionof(hashbits::UInt64, n_bits::Int) = Int(hashbits >> (_HASH_BITS - n_bits)) + 1

"""
    _writerecords!(record_terms, record_coeffs, record_labels, partition_counts, n_bits, first_index, positions, plan, source)

Writes a record for every pair of Pauli string and coefficient of `source`, from `first_index` on, and counts the records of each partition.
Returns `false` if an orbit spans more stages than a label holds.
`positions` is scratch memory.
"""
function _writerecords!(record_terms::Vector{TT}, record_coeffs::Vector{CT}, record_labels::Vector{Int}, partition_counts, n_bits::Int,
    first_index::Int, positions::Vector{Int32}, plan::SubLayerPlan{TT}, source) where {TT,CT}

    index = first_index
    for (pstr, coeff) in source
        representative, coordinate, n_stages = _locateinorbit!(positions, plan, pstr)
        if n_stages > _MAX_STAGES
            return false
        end

        label = _label(representative, coordinate, n_stages)
        record_terms[index] = representative
        record_coeffs[index] = coeff
        record_labels[index] = label
        partition_counts[_partitionof(_hashbits(label, representative), n_bits)] += 1
        index += 1
    end

    return true
end

"""
    _partitionstarts!(partition_starts, partition_counts)

Turns the number of records of every partition and task into the index at which the task writes its first record of the partition.
The records of a partition lie one task after the other, from `partition_starts[p]` to `partition_starts[p+1] - 1`.
"""
function _partitionstarts!(partition_starts::Vector{Int}, partition_counts::Matrix{Int})
    n_partitions, n_tasks = size(partition_counts)
    resize!(partition_starts, n_partitions + 1)

    next_start = 1
    for partition in 1:n_partitions
        partition_starts[partition] = next_start
        for task_id in 1:n_tasks
            n_here = partition_counts[partition, task_id]
            partition_counts[partition, task_id] = next_start
            next_start += n_here
        end
    end
    partition_starts[n_partitions+1] = next_start

    return partition_starts
end

"""
    _sortrecords!(sorted_terms, sorted_coeffs, sorted_labels, cursors, n_bits, record_terms, record_coeffs, record_labels, lo, hi)

Copies the records `lo` to `hi` to where `cursors` points for their partition.
"""
function _sortrecords!(sorted_terms::Vector{TT}, sorted_coeffs::Vector{CT}, sorted_labels::Vector{Int}, cursors, n_bits::Int,
    record_terms::Vector{TT}, record_coeffs::Vector{CT}, record_labels::Vector{Int}, lo::Int, hi::Int) where {TT,CT}

    for i in lo:hi
        representative = record_terms[i]
        label = record_labels[i]
        partition = _partitionof(_hashbits(label, representative), n_bits)

        sorted_index = cursors[partition]
        sorted_terms[sorted_index] = representative
        sorted_coeffs[sorted_index] = record_coeffs[i]
        sorted_labels[sorted_index] = label
        cursors[partition] = sorted_index + 1
    end

    return sorted_terms
end

"""
    _sortrecords!(record_terms, record_coeffs, record_labels, partition_starts, cursors, n_bits)

Sorts the records by partition within their arrays.
`cursors` points to the first record of every partition, and a record that lies in the wrong partition is swapped to where the cursor of its own partition points.
"""
function _sortrecords!(record_terms::Vector{TT}, record_coeffs::Vector{CT}, record_labels::Vector{Int}, partition_starts::Vector{Int},
    cursors, n_bits::Int) where {TT,CT}

    for partition in 1:length(partition_starts)-1
        partition_end = partition_starts[partition+1] - 1
        while cursors[partition] <= partition_end
            i = cursors[partition]
            own_partition = _partitionof(_hashbits(record_labels[i], record_terms[i]), n_bits)

            if own_partition == partition
                cursors[partition] = i + 1
            else
                j = cursors[own_partition]
                record_terms[i], record_terms[j] = record_terms[j], record_terms[i]
                record_coeffs[i], record_coeffs[j] = record_coeffs[j], record_coeffs[i]
                record_labels[i], record_labels[j] = record_labels[j], record_labels[i]
                cursors[own_partition] = j + 1
            end
        end
    end

    return record_terms
end

# the counts of `n_partitions` partitions and `n_tasks` tasks
function _partitioncounts!(workspace::LayerWorkspace, n_partitions::Int, n_tasks::Int)
    if size(workspace.partition_counts) != (n_partitions, n_tasks)
        workspace.partition_counts = Matrix{Int}(undef, n_partitions, n_tasks)
    end
    return fill!(workspace.partition_counts, 0)
end
