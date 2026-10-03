###
##
# How a `RotationLayer` is propagated as a whole.
# The rotations of a layer commute, so a Pauli string and every string they make from it anticommute with the same
# rotations. The strings that anticommute with the same rotations and agree on what those rotations leave unchanged
# form a class, which no rotation of the layer leaves. Every Pauli string becomes a record, the records are partitioned
# by the hash of their class, and partition by partition every class is rotated, one rotation after the other in the
# order of the layer, with the truncations applied after each.
##
###

# whether the sum of the cache is propagated class by class
_propagatesinclasses(prop_cache::AbstractPauliPropagationCache) = _propagatesinclasses(StorageType(prop_cache), prop_cache)
_propagatesinclasses(::PropagationBase.StorageType, prop_cache) = false
_propagatesinclasses(::PropagationBase.DictStorage, prop_cache) = coefftype(prop_cache) <: Number

function _propagatesinclasses(::PropagationBase.ArrayStorage, prop_cache)
    main_terms, main_coeffs, _, _ = PropagationBase._mainauxarrays(prop_cache)
    return coefftype(prop_cache) <: Number && main_terms isa Vector && main_coeffs isa Vector
end

function _propagatesinclasses(storage::PropagationBase.MultiSumStorage, prop_cache)
    return all(zonecache -> _propagatesinclasses(storage.zonestorage, zonecache), zonecaches(prop_cache))
end

"""
    _applylayer!(layer::RotationLayer, prop_cache, theta, truncation; thread=true)

Applies the rotations of the layer class by class, in one pass over the sum.
A layer that acts on a qubit with two different Paulis takes one pass per sublayer (see `_classpasses`).
"""
function _applylayer!(layer::RotationLayer, prop_cache::AbstractPauliPropagationCache, theta, truncation; thread::Bool=true)
    workspace = _takeworkspace(paulitype(prop_cache), coefftype(prop_cache))
    try
        for rotations in _classpasses(layer)
            plan = _prepareclasses(layer, theta, paulitype(prop_cache), coefftype(prop_cache), nqubits(prop_cache), rotations)
            _applypass!(StorageType(prop_cache), prop_cache, plan, truncation, workspace; thread)
        end
    finally
        _putbackworkspace!(workspace)
    end
    return prop_cache
end

# the rotations of the sublayer one by one
function _applyrotations!(applyrotation!::F, prop_cache, layer::RotationLayer, sublayer::Vector{Int}, theta) where {F}
    for index in sublayer
        applyrotation!(prop_cache, PauliRotation(layer.symbols, layer.qinds[index]), _rotationangle(theta, index))
    end
    return prop_cache
end

# The truncation within a class, where the weight of every entry is known. Every coefficient below `min_abs_coeff` is
# truncated, whatever else `truncfunc` checks.
function _layertruncation(truncfunc, max_weight::Real, min_abs_coeff::Real)
    if isinf(max_weight)
        return (; truncfunc, max_weight=_UNLIMITED_WEIGHT, min_abs_coeff)
    else
        return (; truncfunc, max_weight=floor(Int, max_weight), min_abs_coeff)
    end
end

_nevertruncate(pstr, coeff) = false

# no Pauli string is heavier than this
const _UNLIMITED_WEIGHT = typemax(Int)

@inline _limitsweight(truncation) = truncation.max_weight != _UNLIMITED_WEIGHT

@inline function _istruncated(truncation, pstr, coeff)
    if _limitsweight(truncation) && countweight(pstr) > truncation.max_weight
        return true
    end
    return @inline truncation.truncfunc(pstr, coeff)
end


### Arrays

# The records are written into the auxiliary arrays and sorted into the main arrays, whose Pauli strings they replace.
# What the pass makes is written into the auxiliary arrays, which become the sum.
function _applypass!(::PropagationBase.ArrayStorage, prop_cache::AbstractPauliPropagationCache, plan, truncation, workspace; thread::Bool=true)
    task_partitioner, n_tasks = PropagationBase._preparetasks(activesize(prop_cache), thread)
    return _applypassintasks!(prop_cache, plan, truncation, workspace, task_partitioner, n_tasks)
end

function _applypassintasks!(prop_cache::AbstractPauliPropagationCache, plan, truncation, workspace, task_partitioner, n_tasks::Int)
    n_terms = activesize(prop_cache)
    if n_terms == 0
        return prop_cache
    end

    # the arrays do not grow while several tasks write to them
    if n_tasks > 1
        PropagationBase._ensurecapacity!(prop_cache, n_terms + n_terms ÷ 2)
    end
    main_terms, main_coeffs, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(prop_cache)
    record_labels = indices(prop_cache)
    sorted_labels = _ensurelength!(workspace.labels, n_terms)
    PropagationBase._checkfits(n_terms, main_terms, main_coeffs)
    PropagationBase._checkfits(n_terms, aux_terms, aux_coeffs)
    PropagationBase._checkfits(n_terms, record_labels, sorted_labels)

    tasks = _taskworkspaces!(workspace, n_tasks, length(plan.masks))
    n_bits = _partitionbits(n_terms, sizeof(eltype(main_terms)) + sizeof(eltype(main_coeffs)) + sizeof(Int))
    partition_counts = zeros(Int, (1 << n_bits) + 1, n_tasks)

    # a record for every Pauli string, at the index of the Pauli string
    function write_chunk!(task_id)
        chunk = task_partitioner[task_id]
        source = zip(view(main_terms, chunk), view(main_coeffs, chunk))
        _writerecords!(aux_terms, aux_coeffs, record_labels, view(partition_counts, :, task_id), n_bits, chunk.start, plan, source)
    end
    PropagationBase._eachtask(write_chunk!, n_tasks)

    partition_starts = _partitionstarts!(partition_counts)
    function sort_chunk!(task_id)
        chunk = task_partitioner[task_id]
        _sortrecords!(main_terms, main_coeffs, sorted_labels, view(partition_counts, :, task_id), n_bits,
            aux_terms, aux_coeffs, record_labels, chunk.start, chunk.stop)
    end
    PropagationBase._eachtask(sort_chunk!, n_tasks)

    if n_tasks == 1
        n_written = _rotatepartitions!(prop_cache, plan, truncation, tasks[1], sorted_labels, partition_starts)
    else
        n_written = _rotatepartitionsintasks!(prop_cache, plan, truncation, workspace, sorted_labels, partition_starts, n_tasks)
    end

    PropagationBase._commitwrite!(prop_cache, n_written, 0)
    return prop_cache
end

# One task takes the partitions in turn and writes into the auxiliary arrays as it goes. Returns the number of Pauli strings written.
function _rotatepartitions!(prop_cache::AbstractPauliPropagationCache, plan, truncation, task, sorted_labels::Vector{Int}, partition_starts)
    output = ArrayOutput(prop_cache, 0)
    for partition in 1:length(partition_starts)-1
        # the arrays of the cache are looked up for every partition, since they are replaced when they grow
        sorted_terms, sorted_coeffs, _, _ = PropagationBase._mainauxarrays(prop_cache)
        _rotatepartition!(output, task, plan, truncation, sorted_terms, sorted_coeffs, sorted_labels,
            partition_starts[partition], partition_starts[partition+1] - 1)
    end
    return output.n_written
end

# Several tasks share out the partitions and write into ranges of the auxiliary arrays that they reserve.
# What does not fit the arrays is left in the buffers of the tasks and written once the arrays have grown.
function _rotatepartitionsintasks!(prop_cache::AbstractPauliPropagationCache, plan, truncation, workspace, sorted_labels::Vector{Int},
    partition_starts, n_tasks::Int)

    sorted_terms, sorted_coeffs, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(prop_cache)
    tasks = workspace.tasks
    outputs = [task.output for task in tasks]
    n_reserved = Threads.Atomic{Int}(0)
    for task_id in 1:n_tasks
        _openoutput!(outputs[task_id], aux_terms, aux_coeffs, n_reserved)
    end

    n_partitions = length(partition_starts) - 1
    function rotate_partitions!(task_id)
        for partition in task_id:n_tasks:n_partitions
            _rotatepartition!(outputs[task_id], tasks[task_id], plan, truncation, sorted_terms, sorted_coeffs, sorted_labels,
                partition_starts[partition], partition_starts[partition+1] - 1)
        end
        _flush!(outputs[task_id])
    end
    PropagationBase._eachtask(rotate_partitions!, n_tasks)

    n_written = n_reserved[]
    n_left = sum(outputs[task_id].n_written for task_id in 1:n_tasks)
    if n_left > 0
        PropagationBase._ensurecapacity!(prop_cache, n_written + n_left)
        _, _, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(prop_cache)
        PropagationBase._checkfits(n_written + n_left, aux_terms, aux_coeffs)

        for task_id in 1:n_tasks
            output = outputs[task_id]
            copyto!(aux_terms, n_written + 1, output.buffer_terms, 1, output.n_written)
            copyto!(aux_coeffs, n_written + 1, output.buffer_coeffs, 1, output.n_written)
            n_written += output.n_written
        end
    end

    for task_id in 1:n_tasks
        _closeoutput!(outputs[task_id])
    end
    return n_written
end


### Any sum that is iterated and added to

# The records are kept in the workspace and sorted within their arrays. They hold all of the sum,
# so the sum is emptied and takes what the pass makes, instead of a second sum of its size.
function _applypass!(::PropagationBase.DictStorage, prop_cache::AbstractPauliPropagationCache, plan, truncation, workspace; thread::Bool=true)
    PropagationBase._checkauxempty(prop_cache)
    main_sum = mainsum(prop_cache)
    n_terms = length(main_sum)

    record_terms = _ensurelength!(workspace.terms, n_terms)
    record_coeffs = _ensurelength!(workspace.coeffs, n_terms)
    record_labels = _ensurelength!(workspace.labels, n_terms)

    task = first(_taskworkspaces!(workspace, 1, length(plan.masks)))
    n_bits = _partitionbits(n_terms, sizeof(eltype(record_terms)) + sizeof(eltype(record_coeffs)) + sizeof(Int))
    partition_counts = zeros(Int, (1 << n_bits) + 1, 1)

    _writerecords!(record_terms, record_coeffs, record_labels, view(partition_counts, :, 1), n_bits, 1, plan, main_sum)
    partition_starts = _partitionstarts!(partition_counts)
    _sortrecords!(record_terms, record_coeffs, record_labels, partition_starts, view(partition_counts, :, 1), n_bits)

    empty!(main_sum)
    for partition in 1:length(partition_starts)-1
        _rotatepartition!(main_sum, task, plan, truncation, record_terms, record_coeffs, record_labels,
            partition_starts[partition], partition_starts[partition+1] - 1)
    end

    return prop_cache
end


### Multi sums

# The Pauli strings of a class belong to many zones, so the records of a class are collected in the zone that the hash of
# the class picks. Every zone reads its Pauli strings twice: to count the records for every zone, and to write them
# there, each zone into a range of its own. It then rotates the classes collected in it. A zone of arrays keeps its
# records in its auxiliary arrays, so that its main arrays are free, and every zone writes what it makes into the main
# arrays of the zones that own it. Any other zone puts what it makes in its outbox, and at last the zones are emptied
# and take what the outboxes hold for them.
function _applypass!(::PropagationBase.MultiSumStorage, prop_cache::AbstractPauliPropagationCache, plan, truncation, workspace; thread::Bool=true)
    PropagationBase._checkauxempty(prop_cache)
    n_terms = length(prop_cache)
    if n_terms == 0
        return prop_cache
    end

    zone_caches = zonecaches(prop_cache)
    n_zones = nzones(prop_cache)
    tasks = _taskworkspaces!(workspace, n_zones, length(plan.masks))

    # The highest bits of the hash pick the zone, and the bits below them a partition within the zone. Every zone writes its
    # records into the partitions of all zones, so the most partitions apply to those of all zones together: a zone that
    # writes into more partitions than its processor cache holds the ends of finds none of them there.
    zone_bits = trailing_zeros(n_zones)
    record_bytes = sizeof(paulitype(prop_cache)) + sizeof(coefftype(prop_cache)) + sizeof(Int)
    n_bits = max(zone_bits, min(_MAX_PARTITION_BITS, zone_bits + _partitionbits(cld(n_terms, n_zones), record_bytes)))
    n_partitions_per_zone = 1 << (n_bits - zone_bits)

    # One row more than there are partitions, so that the counts of a partition are not a power of two apart, which would
    # put them all into one set of the processor cache. Every zone clears its own column, so that the columns are cleared in
    # parallel and each lies in the memory of the thread of its zone.
    partition_counts = Matrix{Int}(undef, n_zones * n_partitions_per_zone + 1, n_zones)

    # the labels that the records are counted by, read again when the records are written; every zone makes room for its own
    cached_labels = _cachedlabels!(workspace, n_zones)
    function count_zone!(zone_id)
        zonecache = zone_caches[zone_id]
        zone_counts = fill!(view(partition_counts, :, zone_id), 0)
        labels = _ensurelength!(cached_labels[zone_id], length(zonecache))
        _countrecords!(zone_counts, n_bits, zone_bits, plan, zonecache, labels)
    end
    PropagationBase._eachzone(count_zone!, prop_cache, thread)

    # the records of a zone are numbered from 1 on
    partition_starts = _partitionstarts!(partition_counts)
    zone_starts = [partition_starts[(zone_id-1)*n_partitions_per_zone+1] for zone_id in 1:n_zones+1]
    zone_terms, zone_coeffs, zone_labels = _zonerecords!(PropagationBase.zonestorage(prop_cache), workspace, zone_caches, diff(zone_starts))

    function write_zone!(zone_id)
        _writezonerecords!(zone_terms, zone_coeffs, zone_labels, zone_starts, view(partition_counts, :, zone_id), n_bits, zone_bits,
            zone_caches[zone_id], cached_labels[zone_id])
    end
    PropagationBase._eachzone(write_zone!, prop_cache, thread)

    # The classes of a layer differ widely in size, so the zones collect different amounts of work: every zone rotates
    # its own partitions first and then takes those that the other zones have not taken yet, one at a time.
    zone_storage = PropagationBase.zonestorage(prop_cache)
    outputs = _zoneoutputs!(zone_storage, prop_cache, tasks)
    n_partitions_taken = [Threads.Atomic{Int}(0) for _ in 1:n_zones]
    function rotate_zone!(zone_id)
        for offset in 0:n_zones-1
            records_zone = mod1(zone_id + offset, n_zones)
            first_record = zone_starts[records_zone]
            first_partition = (records_zone - 1) * n_partitions_per_zone
            while true
                partition_in_zone = Threads.atomic_add!(n_partitions_taken[records_zone], 1) + 1
                if partition_in_zone > n_partitions_per_zone
                    break
                end
                partition = first_partition + partition_in_zone
                lo = partition_starts[partition] - first_record + 1
                hi = partition_starts[partition+1] - first_record
                _rotatepartition!(outputs[zone_id], tasks[zone_id], plan, truncation, zone_terms[records_zone],
                    zone_coeffs[records_zone], zone_labels[records_zone], lo, hi)
            end
        end
        _finishoutput!(outputs[zone_id])
    end
    PropagationBase._eachzone(rotate_zone!, prop_cache, thread)

    # the records hold all of the sum, so every zone ends up with what the pass made for it alone
    collect_zone!(zone_id) = _collectzone!(zone_storage, prop_cache, zone_id, outputs)
    PropagationBase._eachzone(collect_zone!, prop_cache, thread)

    PropagationBase._syncsums!(prop_cache)
    return prop_cache
end

# What the task of every zone writes to: the main arrays of all zones of arrays, or else its outbox.
function _zoneoutputs!(::PropagationBase.ArrayStorage, prop_cache, tasks)
    zone_caches = zonecaches(prop_cache)
    n_zones = length(zone_caches)
    n_reserved = [Threads.Atomic{Int}(0) for _ in 1:n_zones]

    # the buffers of a task take about as much memory whatever the number of zones
    flush_length = max(256, 4 * _TASK_BUFFER_LENGTH ÷ n_zones)
    function zone_output(zone_id)
        task_outputs = _zonetaskoutputs!(tasks[zone_id], n_zones, flush_length)
        for target in 1:n_zones
            main_terms, main_coeffs, _, _ = PropagationBase._mainauxarrays(zone_caches[target])
            _openoutput!(task_outputs[target], main_terms, main_coeffs, n_reserved[target])
        end
        return ZoneOutputs(zonemap(prop_cache), task_outputs, n_reserved)
    end
    return [zone_output(zone_id) for zone_id in 1:n_zones]
end

_zoneoutputs!(::PropagationBase.StorageType, prop_cache, tasks) = outboxes(prop_cache)

# A zone of arrays takes what the tasks could not write into its main arrays, which grow for it. Its Pauli strings are all
# different, since every class was rotated in one zone.
function _collectzone!(::PropagationBase.ArrayStorage, prop_cache, zone_id::Int, outputs)
    zonecache = zonecaches(prop_cache)[zone_id]
    n_written = first(outputs).n_reserved[zone_id][]
    n_left = 0
    for output in outputs
        n_left += output.task_outputs[zone_id].n_written
    end

    if n_left > 0
        PropagationBase._ensurecapacity!(zonecache, n_written + n_left)
        main_terms, main_coeffs, _, _ = PropagationBase._mainauxarrays(zonecache)
        PropagationBase._checkfits(n_written + n_left, main_terms, main_coeffs)
        for output in outputs
            task_output = output.task_outputs[zone_id]
            copyto!(main_terms, n_written + 1, task_output.buffer_terms, 1, task_output.n_written)
            copyto!(main_coeffs, n_written + 1, task_output.buffer_coeffs, 1, task_output.n_written)
            n_written += task_output.n_written
        end
    end

    for output in outputs
        _closeoutput!(output.task_outputs[zone_id])
    end
    setactivesize!(zonecache, n_written)
    PropagationBase.setsortedprefix!(mainsum(zonecache), 0)
    return zonecache
end

# any other zone is emptied and takes what the outboxes hold for it
function _collectzone!(::PropagationBase.StorageType, prop_cache, zone_id::Int, outputs)
    empty!(zonecaches(prop_cache)[zone_id])
    PropagationBase._deliverto!(prop_cache, zone_id)
    return zonecaches(prop_cache)[zone_id]
end

# Arrays for the records that every zone collects, `n_records[zone_id]` of them.
# A zone of arrays keeps them in the auxiliary arrays and `indices` of its cache, any other zone in the workspace.
function _zonerecords!(::PropagationBase.ArrayStorage, workspace, zone_caches, n_records::Vector{Int})
    for (zone_id, zonecache) in enumerate(zone_caches)
        PropagationBase._ensurecapacity!(zonecache, n_records[zone_id])
    end
    zone_terms = [terms(auxsum(zonecache)) for zonecache in zone_caches]
    zone_coeffs = [coefficients(auxsum(zonecache)) for zonecache in zone_caches]
    zone_labels = [indices(zonecache) for zonecache in zone_caches]
    return zone_terms, zone_coeffs, zone_labels
end

function _zonerecords!(::PropagationBase.StorageType, workspace, zone_caches, n_records::Vector{Int})
    while length(workspace.zone_terms) < length(n_records)
        push!(workspace.zone_terms, similar(workspace.terms, 0))
        push!(workspace.zone_coeffs, similar(workspace.coeffs, 0))
        push!(workspace.zone_labels, Int[])
    end

    for zone_id in eachindex(n_records)
        _ensurelength!(workspace.zone_terms[zone_id], n_records[zone_id])
        _ensurelength!(workspace.zone_coeffs[zone_id], n_records[zone_id])
        _ensurelength!(workspace.zone_labels[zone_id], n_records[zone_id])
    end
    return workspace.zone_terms, workspace.zone_coeffs, workspace.zone_labels
end


### Records

# A record is a Pauli string, its coefficient and its label: a hash of its class above `_HASH_SHIFT`, and in the lowest
# bit whether the string has a class, that is whether any rotation of the layer anticommutes with it.
const _HASH_SHIFT = 1
const _HASH_BITS = 8 * sizeof(Int) - _HASH_SHIFT

@inline _labelhash(label::Int) = (label % UInt64) >> _HASH_SHIFT
@inline _hasclass(label::Int) = isodd(label)

# the records of a partition take about this many bytes
const _PARTITION_BYTES = 1 << 18

const _MAX_PARTITION_BITS = 12

# The tables of the classes of a partition hold what its records make as well, so a record counts this many times its
# size when the partitions are sized.
const _CLASS_BYTES_FACTOR = 4

# The number of bits of the hash that pick the partition of a record, so that a partition is small enough to be worked
# on within the cache.
function _partitionbits(n_records::Int, record_bytes::Int)
    n_partitions = cld(n_records * _CLASS_BYTES_FACTOR * record_bytes, _PARTITION_BYTES)
    if n_partitions <= 1
        return 0
    end
    return min(_MAX_PARTITION_BITS, 8 * sizeof(Int) - leading_zeros(n_partitions - 1))
end

# The partition of a record: the highest `n_bits` bits of the hash of its class, where the table of a partition reads the lowest.
@inline _partitionof(label::Int, n_bits::Int) = Int(_labelhash(label) >> (_HASH_BITS - n_bits)) + 1

# The zone that collects a record of a multi sum, and the partition of the record among those of all zones: the highest
# `n_zone_bits` bits of the hash pick the zone, and the bits below them, up to `n_bits`, a partition of the zone.
@inline function _zonepartitionof(label::Int, n_bits::Int, n_zone_bits::Int)
    partition = _partitionof(label, n_bits)
    return ((partition - 1) >> (n_bits - n_zone_bits)) + 1, partition
end

"""
    _writerecords!(record_terms, record_coeffs, record_labels, partition_counts, n_bits, first_index, plan, source)

Writes a record for every Pauli string of `source` and its coefficient, from `first_index` on, and counts the records of each partition.
"""
function _writerecords!(record_terms::Vector{TT}, record_coeffs::Vector{CT}, record_labels::Vector{Int}, partition_counts, n_bits::Int,
    first_index::Int, plan, source) where {TT,CT}

    index = first_index
    for (pstr, coeff) in source
        label = _classlabel(plan, pstr)
        record_terms[index] = pstr
        record_coeffs[index] = coeff
        record_labels[index] = label
        partition_counts[_partitionof(label, n_bits)] += 1
        index += 1
    end

    return record_terms
end

"""
    _partitionstarts!(partition_counts)

Turns the number of records of every partition and task into the index at which the task writes its first record of the partition,
and returns where every partition starts.
The records of a partition lie one task after the other, from `partition_starts[p]` to `partition_starts[p+1] - 1`.
"""
function _partitionstarts!(partition_counts::Matrix{Int})
    n_partitions, n_tasks = size(partition_counts)
    partition_starts = Vector{Int}(undef, n_partitions + 1)

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
        record_term = record_terms[i]
        label = record_labels[i]
        partition = _partitionof(label, n_bits)

        sorted_index = cursors[partition]
        sorted_terms[sorted_index] = record_term
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
            own_partition = _partitionof(record_labels[i], n_bits)

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

"""
    _countrecords!(partition_counts, n_bits, n_zone_bits, plan, source, cached_labels)

Counts the records of each partition of the zones that the Pauli strings of `source` make, and keeps their labels in
`cached_labels`, in the order of `source`.
"""
function _countrecords!(partition_counts, n_bits::Int, n_zone_bits::Int, plan, source, cached_labels::Vector{Int})
    source_index = 0
    for (pstr, _) in source
        label = _classlabel(plan, pstr)
        source_index += 1
        cached_labels[source_index] = label
        _, partition = _zonepartitionof(label, n_bits, n_zone_bits)
        partition_counts[partition] += 1
    end
    return partition_counts
end

# a vector of labels for every zone, which the zone makes long enough for its Pauli strings itself
function _cachedlabels!(workspace, n_zones::Int)
    cached_labels = workspace.cached_labels
    while length(cached_labels) < n_zones
        push!(cached_labels, Int[])
    end
    return cached_labels
end

"""
    _writezonerecords!(zone_terms, zone_coeffs, zone_labels, zone_starts, cursors, n_bits, n_zone_bits, zonecache, cached_labels)

Writes a record for every Pauli string of `zonecache` and its coefficient, with its label from `cached_labels`, to where
`cursors` points for its partition, in the arrays of the zone that the partition belongs to.
"""
function _writezonerecords!(zone_terms::Vector{Vector{TT}}, zone_coeffs::Vector{Vector{CT}}, zone_labels::Vector{Vector{Int}},
    zone_starts::Vector{Int}, cursors, n_bits::Int, n_zone_bits::Int, zonecache, cached_labels::Vector{Int}) where {TT,CT}

    source_index = 0
    for (pstr, coeff) in zonecache
        source_index += 1
        label = cached_labels[source_index]
        zone_id, partition = _zonepartitionof(label, n_bits, n_zone_bits)

        index = cursors[partition] - zone_starts[zone_id] + 1
        zone_terms[zone_id][index] = pstr
        zone_coeffs[zone_id][index] = coeff
        zone_labels[zone_id][index] = label
        cursors[partition] += 1
    end

    return zone_terms
end

# A hash of `_HASH_BITS` bits. The limbs are multiplied independently of each other, so that the hash of a wide Pauli string is no chain over its limbs.
@inline function _hashbits(pstr)
    limbs = _limbs(pstr)
    folded = zero(UInt64)
    for i in eachindex(limbs)
        folded ⊻= limbs[i] * _foldfactor(i)
    end
    return PropagationBase._mix64(folded) >> _HASH_SHIFT
end

# an odd factor for every limb
const _FOLD_FACTORS = ntuple(i -> PropagationBase._mix64(UInt64(i)) | one(UInt64), 64)

@inline _foldfactor(i::Int) = _FOLD_FACTORS[((i-1)&63)+1] + ((UInt64(i - 1) >> 6) << 1)

# a Pauli string as 64-bit limbs, the lowest first
@inline _limbs(pstr::NTupleInteger) = pstr.limbs
@inline _limbs(pstr::UInt128) = (pstr % UInt64, (pstr >> 64) % UInt64)
@inline _limbs(pstr::Union{UInt8,UInt16,UInt32,UInt64}) = (pstr % UInt64,)


### Where a pass writes the Pauli strings it makes

# the auxiliary arrays of a cache written by one task, which grow as they fill
mutable struct ArrayOutput{PC}
    prop_cache::PC
    n_written::Int
end

# a task writes to the sum once it holds this many Pauli strings
const _TASK_BUFFER_LENGTH = 4096

# The auxiliary arrays of a cache written by several tasks. A task collects what it makes in a buffer and copies it into
# a range of the arrays that it reserves. The arrays do not grow while the tasks write, so a task that finds them full
# keeps what it makes in its buffer.
mutable struct TaskOutput{TT,CT}
    buffer_terms::Vector{TT}
    buffer_coeffs::Vector{CT}
    n_written::Int
    sum_terms::Vector{TT}
    sum_coeffs::Vector{CT}
    n_reserved::Threads.Atomic{Int}
    is_full::Bool
    flush_length::Int
end

# Room past the flush length for the block of a class, so that the buffer does not grow for one.
_taskbufferroom(flush_length::Int) = flush_length + (1 << _MAX_BLOCK_KEY_BITS)

function TaskOutput(::Type{TT}, ::Type{CT}, flush_length::Int=_TASK_BUFFER_LENGTH) where {TT,CT}
    n_room = _taskbufferroom(flush_length)
    return TaskOutput{TT,CT}(Vector{TT}(undef, n_room), Vector{CT}(undef, n_room), 0, TT[], CT[], Threads.Atomic{Int}(0), false, flush_length)
end

const ArrayOutputs = Union{ArrayOutput,TaskOutput}

@inline function _emit!(output::ArrayOutputs, pstr, coeff)
    output_terms, output_coeffs = _roomtoemit!(output, 1)
    n_written = output.n_written + 1
    output_terms[n_written] = pstr
    output_coeffs[n_written] = coeff
    output.n_written = n_written
    return
end

# the arrays to write to, with room for `n_more` Pauli strings past the ones written
@inline function _roomtoemit!(output::ArrayOutput, n_more::Int)
    prop_cache = output.prop_cache
    n_needed = output.n_written + n_more
    if n_needed > capacity(prop_cache)
        PropagationBase._ensurecapacity!(prop_cache, n_needed)
    end

    aux_sum = auxsum(prop_cache)
    aux_terms = terms(aux_sum)
    aux_coeffs = coefficients(aux_sum)
    PropagationBase._checkfits(n_needed, aux_terms, aux_coeffs)
    return aux_terms, aux_coeffs
end

@inline function _roomtoemit!(output::TaskOutput, n_more::Int)
    if output.n_written + n_more > output.flush_length && !output.is_full
        _flush!(output)
    end

    n_needed = output.n_written + n_more
    if n_needed > length(output.buffer_terms) || n_needed > length(output.buffer_coeffs)
        _ensurelength!(output.buffer_terms, n_needed)
        _ensurelength!(output.buffer_coeffs, n_needed)
    end
    return output.buffer_terms, output.buffer_coeffs
end

# an empty buffer in front of the arrays that the tasks write to, of which `n_reserved` entries are taken
function _openoutput!(output::TaskOutput{TT,CT}, sum_terms::Vector{TT}, sum_coeffs::Vector{CT}, n_reserved::Threads.Atomic{Int}) where {TT,CT}
    output.n_written = 0
    output.sum_terms = sum_terms
    output.sum_coeffs = sum_coeffs
    output.n_reserved = n_reserved
    output.is_full = false
    return output
end

# Lets go of the arrays of the sum, and of a buffer that had to keep more than it started with room for.
function _closeoutput!(output::TaskOutput{TT,CT}) where {TT,CT}
    n_room = _taskbufferroom(output.flush_length)
    if length(output.buffer_terms) > n_room
        output.buffer_terms = Vector{TT}(undef, n_room)
        output.buffer_coeffs = Vector{CT}(undef, n_room)
    end
    output.sum_terms = TT[]
    output.sum_coeffs = CT[]
    output.n_written = 0
    return output
end

# Copies the buffer into a range of the arrays that no other task writes to, unless the arrays are full.
function _flush!(output::TaskOutput)
    n_buffered = output.n_written
    if n_buffered == 0 || output.is_full
        return output
    end

    n_room = min(length(output.sum_terms), length(output.sum_coeffs))
    n_taken = output.n_reserved[]
    while true
        if n_taken + n_buffered > n_room
            output.is_full = true
            return output
        end
        n_seen = Threads.atomic_cas!(output.n_reserved, n_taken, n_taken + n_buffered)
        if n_seen == n_taken
            break
        end
        n_taken = n_seen
    end

    copyto!(output.sum_terms, n_taken + 1, output.buffer_terms, 1, n_buffered)
    copyto!(output.sum_coeffs, n_taken + 1, output.buffer_coeffs, 1, n_buffered)
    output.n_written = 0
    return output
end

# The main arrays of the zones of a multi sum, written by one task: a Pauli string goes to the output of the zone that owns it.
struct ZoneOutputs{ZM,TO<:TaskOutput}
    zone_map::ZM
    task_outputs::Vector{TO}
    # how much of the main arrays of every zone the tasks have taken
    n_reserved::Vector{Threads.Atomic{Int}}
end

@inline function _emit!(output::ZoneOutputs, pstr, coeff)
    _emit!(output.task_outputs[PropagationBase.zoneof(output.zone_map, pstr)], pstr, coeff)
    return
end

# copies what the buffers hold into the zones, as far as they have room
function _finishoutput!(output::ZoneOutputs)
    foreach(_flush!, output.task_outputs)
    return output
end

_finishoutput!(output) = output

# a term sum takes the Pauli strings one by one: a Pauli sum adds them, the outbox of a multi sum passes them to the zone that owns them
@inline function _emit!(output::AbstractTermSum, pstr, coeff)
    push!(output, pstr, coeff)
    return
end


### The scratch memory of a layer

mutable struct TaskWorkspace{TT,CT}
    # the rotations that anticommute with the Pauli strings of a class, in the order of the layer
    rotations::Vector{Int32}

    # the classes of one partition, found through `slots`, and the records of each
    slots::Vector{Int32}
    class_of::Vector{Int32}
    class_keys::Vector{TT}
    class_hashes::Vector{UInt64}
    class_starts::Vector{Int}
    class_records::Vector{Int32}

    # the block of a class with few key bits: a coefficient and a Pauli string for every entry, and which are present
    block_coeffs::Vector{CT}
    block_terms::Vector{TT}
    block_words::Vector{UInt64}

    # The Pauli strings of a class with more key bits, their keys, whether they are present, and the last rotation that
    # mixed them, found through the first `class_table_length` of `class_slots`: at their key where `class_hash_shift`
    # is 0, and otherwise by the highest bits of the hash of their key from `class_hash_shift` on. The entries that a
    # rotation mixes or that make a partner, and their partners.
    entry_terms::Vector{TT}
    entry_keys::Vector{UInt64}
    entry_coeffs::Vector{CT}
    entry_present::Vector{Bool}
    entry_steps::Vector{Int32}
    n_entries::Int
    class_slots::Vector{Int32}
    class_table_length::Int
    class_hash_shift::Int
    class_is_open::Bool
    events::Vector{Int32}
    event_partners::Vector{Int32}

    # where the task writes when several tasks write to one array sum, and to the zones of a multi sum
    output::TaskOutput{TT,CT}
    zone_outputs::Vector{TaskOutput{TT,CT}}
end

function TaskWorkspace(::Type{TT}, ::Type{CT}) where {TT,CT}
    return TaskWorkspace{TT,CT}(Int32[], Int32[], Int32[], TT[], UInt64[], Int[], Int32[], CT[], TT[], UInt64[],
        TT[], UInt64[], CT[], Bool[], Int32[], 0, Int32[], 0, 64, false, Int32[], Int32[], TaskOutput(TT, CT), TaskOutput{TT,CT}[])
end

"""
    LayerWorkspace(TT, CT)

The scratch memory of a `RotationLayer` applied to Pauli strings of the type `TT` with coefficients of the type `CT`,
which the layers of a propagation reuse.
"""
mutable struct LayerWorkspace{TT,CT}
    tasks::Vector{TaskWorkspace{TT,CT}}

    # the labels of the records of an array sum, sorted by partition, or those of the records of a Pauli sum
    labels::Vector{Int}

    # the records of a sum that has no arrays to keep them in
    terms::Vector{TT}
    coeffs::Vector{CT}

    # the records that every zone of a multi sum without arrays collects
    zone_terms::Vector{Vector{TT}}
    zone_coeffs::Vector{Vector{CT}}
    zone_labels::Vector{Vector{Int}}

    # the labels of the Pauli strings of every zone of a multi sum, found when its records are counted
    cached_labels::Vector{Vector{Int}}
end

LayerWorkspace(::Type{TT}, ::Type{CT}) where {TT,CT} =
    LayerWorkspace{TT,CT}(TaskWorkspace{TT,CT}[], Int[], TT[], CT[], Vector{TT}[], Vector{CT}[], Vector{Int}[], Vector{Int}[])

# Workspaces that no layer is using. A layer takes one out and puts it back when it is done, so that the next layer
# uses the same memory, and propagations that run at the same time each have their own.
const _IDLE_WORKSPACES = Dict{DataType,Vector{Any}}()
const _IDLE_WORKSPACES_LOCK = ReentrantLock()

function _takeworkspace(::Type{TT}, ::Type{CT}) where {TT,CT}
    lock(_IDLE_WORKSPACES_LOCK)
    try
        idle_workspaces = get(_IDLE_WORKSPACES, LayerWorkspace{TT,CT}, nothing)
        if !isnothing(idle_workspaces) && !isempty(idle_workspaces)
            return pop!(idle_workspaces)::LayerWorkspace{TT,CT}
        end
    finally
        unlock(_IDLE_WORKSPACES_LOCK)
    end
    return LayerWorkspace(TT, CT)
end

function _putbackworkspace!(workspace::LayerWorkspace)
    lock(_IDLE_WORKSPACES_LOCK)
    try
        newlist() = Any[]
        push!(get!(newlist, _IDLE_WORKSPACES, typeof(workspace)), workspace)
    finally
        unlock(_IDLE_WORKSPACES_LOCK)
    end
    return
end

# the memory of `n_tasks` tasks for a pass of `n_rotations` rotations
function _taskworkspaces!(workspace::LayerWorkspace{TT,CT}, n_tasks::Int, n_rotations::Int) where {TT,CT}
    tasks = workspace.tasks
    while length(tasks) < n_tasks
        push!(tasks, TaskWorkspace(TT, CT))
    end

    for task_id in 1:n_tasks
        _ensurelength!(tasks[task_id].rotations, n_rotations)
    end
    return tasks
end

# the outputs of a task for `n_zones` zones, which flush at `flush_length`
function _zonetaskoutputs!(task::TaskWorkspace{TT,CT}, n_zones::Int, flush_length::Int) where {TT,CT}
    task_outputs = task.zone_outputs
    if length(task_outputs) != n_zones || any(task_output -> task_output.flush_length != flush_length, task_outputs)
        empty!(task_outputs)
        for _ in 1:n_zones
            push!(task_outputs, TaskOutput(TT, CT, flush_length))
        end
    end
    return task_outputs
end

# at least `n` elements
function _ensurelength!(array::Vector, n::Int)
    if length(array) < n
        resize!(array, max(n, 2 * length(array)))
    end
    return array
end
