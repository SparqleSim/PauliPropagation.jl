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
    _applylayer!(layer::RotationLayer, prop_cache, theta, truncfunc, min_abs_coeff; thread=true)

Applies the rotations of the layer class by class, in one pass over the sum, and truncates the Pauli strings for which
`truncfunc` returns `true` after every rotation. `truncfunc` truncates every coefficient below `min_abs_coeff`.
A layer that acts on a qubit with two different Paulis takes more than one pass (see `_classpasses`).
"""
function _applylayer!(layer::RotationLayer, prop_cache::AbstractPauliPropagationCache, theta, truncfunc::F, min_abs_coeff::Real;
    thread::Bool=true) where {F}

    workspace = _takeworkspace(paulitype(prop_cache), coefftype(prop_cache))
    try
        for rotations in _classpasses(layer)
            plan = _prepareclasses(layer, theta, paulitype(prop_cache), coefftype(prop_cache), nqubits(prop_cache), rotations;
                min_abs_coeff)
            _applypass!(prop_cache, plan, truncfunc, workspace; thread)
        end
    finally
        _putbackworkspace!(workspace)
    end
    return prop_cache
end

_nevertruncate(pstr, coeff) = false


### A pass over the sum

# One pass of the rotations of `plan` over the sum: the Pauli strings are grouped by the hash of their class, and every
# group is split into its classes, which are rotated.
function _applypass!(prop_cache::AbstractPauliPropagationCache, plan, truncfunc::F, workspace; thread::Bool=true) where {F}
    if length(prop_cache) == 0
        return prop_cache
    end
    sources = _recordsources(StorageType(prop_cache), prop_cache, workspace, thread)
    _applypass!(prop_cache, plan, truncfunc, workspace, sources, thread)
    return prop_cache
end

function _applypass!(prop_cache::AbstractPauliPropagationCache, plan, truncfunc::F, workspace, sources, thread::Bool) where {F}
    classlabel(pstr) = _classlabel(plan, pstr)
    rotategroup!(sink, task, group_terms, group_coeffs) = _rotategroup!(sink, task, plan, truncfunc, group_terms, group_coeffs)
    _applytogroups!(classlabel, rotategroup!, prop_cache, workspace, sources, thread)
    return prop_cache
end

"""
    _applytogroups!(labelof, applytogroup!, prop_cache, workspace, sources, thread)

Groups the Pauli strings of the sum by their label `labelof(pstr)`, a 64-bit hash, and calls
`applytogroup!(sink, task, group_terms, group_coeffs)` once for every group, with all of its Pauli strings and their
coefficients. What the calls write to their sinks replaces the sum.
Every Pauli string becomes a record: the records are counted by partition, written partition by partition, and the
partitions are grouped by tasks that each take the next partition that no task has taken yet.
The storage of the sum decides what every task reads (`sources`, from `_recordsources`), where the records are kept
(`_recordarrays!`), where the tasks write (`_passsinks!`) and how that becomes the sum (`_collectpass!`).
"""
function _applytogroups!(labelof::L, applytogroup!::G, prop_cache::AbstractPauliPropagationCache, workspace, sources,
    thread::Bool) where {L,G}

    storage = StorageType(prop_cache)
    n_terms = length(prop_cache)
    n_sources = length(sources)
    tasks = _ensurecount!(workspace.tasks, n_sources)

    # The highest bits of a label pick the zone that collects the record, and the bits below them a partition of the
    # zone. Every source writes into the partitions of all zones, so the most partitions apply to those of all zones
    # together: a source that writes into more partitions than its processor cache holds the ends of finds none of them there.
    n_zones = _nrecordzones(storage, prop_cache)
    zone_bits = trailing_zeros(n_zones)
    record_bytes = sizeof(paulitype(prop_cache)) + sizeof(coefftype(prop_cache)) + sizeof(Int)
    n_bits = max(zone_bits, min(_MAX_PARTITION_BITS, zone_bits + _partitionbits(cld(n_terms, n_zones), record_bytes)))
    n_partitions_per_zone = 1 << (n_bits - zone_bits)

    # One row more than there are partitions, so that the counts of a partition are not a power of two apart, which would
    # put them all into one set of the processor cache. Every source clears its own column, so that the columns are
    # cleared in parallel and each lies in the memory of the thread that reads the source.
    partition_counts = Matrix{Int}(undef, n_zones * n_partitions_per_zone + 1, n_sources)
    function count_source!(source_id)
        source, labels = sources[source_id]
        _countrecords!(labelof, fill!(view(partition_counts, :, source_id), 0), n_bits, zone_bits, source, labels)
    end
    _eachsource(count_source!, storage, prop_cache, n_sources, thread)

    # the records of a zone are numbered from 1 on
    partition_starts = _partitionstarts!(partition_counts)
    zone_starts = [partition_starts[(zone_id-1)*n_partitions_per_zone+1] for zone_id in 1:n_zones+1]
    record_terms, record_coeffs, record_labels = _recordarrays!(storage, prop_cache, workspace, diff(zone_starts))
    function write_source!(source_id)
        source, labels = sources[source_id]
        _writerecords!(record_terms, record_coeffs, record_labels, zone_starts, view(partition_counts, :, source_id), n_bits,
            zone_bits, source, labels)
    end
    _eachsource(write_source!, storage, prop_cache, n_sources, thread)

    sinks = _passsinks!(storage, prop_cache, workspace, tasks, n_sources)
    _applytopartitions!(applytogroup!, sinks, tasks, record_terms, record_coeffs, record_labels, partition_starts, zone_starts,
        n_partitions_per_zone, storage, prop_cache, thread)
    _collectpass!(storage, prop_cache, sinks, thread)
    return prop_cache
end

# The groups differ widely in size, so the partitions do too: every task takes the partitions of its own zone first and
# then those of the other zones, one at a time, each the next one that no task has taken yet.
function _applytopartitions!(applytogroup!::G, sinks, tasks, record_terms, record_coeffs, record_labels, partition_starts,
    zone_starts, n_partitions_per_zone::Int, storage, prop_cache, thread::Bool) where {G}

    n_zones = length(zone_starts) - 1
    n_partitions_taken = [Threads.Atomic{Int}(0) for _ in 1:n_zones]
    function apply_to_partitions!(task_id)
        for offset in 0:n_zones-1
            zone_id = mod1(task_id + offset, n_zones)
            first_record = zone_starts[zone_id]
            first_partition = (zone_id - 1) * n_partitions_per_zone
            while true
                partition_in_zone = Threads.atomic_add!(n_partitions_taken[zone_id], 1) + 1
                if partition_in_zone > n_partitions_per_zone
                    break
                end
                partition = first_partition + partition_in_zone
                _applytopartition!(applytogroup!, sinks[task_id], tasks[task_id], record_terms[zone_id], record_coeffs[zone_id],
                    record_labels[zone_id], partition_starts[partition] - first_record + 1, partition_starts[partition+1] - first_record)
            end
        end
        _finish!(sinks[task_id])
    end
    _eachsource(apply_to_partitions!, storage, prop_cache, length(sinks), thread)
    return sinks
end

"""
    _applytopartition!(applytogroup!, sink, task, record_terms, record_coeffs, record_labels, lo, hi)

Groups the records `lo` to `hi` by their label and calls `applytogroup!(sink, task, group_terms, group_coeffs)` for every
group, with the Pauli strings and coefficients of its records next to each other in the task's scratch.
"""
function _applytopartition!(applytogroup!::G, sink, task, record_terms::Vector{TT}, record_coeffs::Vector{CT},
    record_labels::Vector{Int}, lo::Int, hi::Int) where {G,TT,CT}

    if lo > hi
        return sink
    end
    if !(1 <= lo && hi <= min(length(record_terms), length(record_coeffs), length(record_labels)))
        throw(ArgumentError("the records $lo to $hi are not among the records"))
    end
    n_records = hi - lo + 1
    group_of = PropagationBase._ensurecapacity!(task.group_of, n_records)
    group_hashes = PropagationBase._ensurecapacity!(task.group_hashes, n_records)
    group_starts = PropagationBase._ensurecapacity!(task.group_starts, n_records + 1)

    # the groups found so far, through their hash
    table_length = max(16, nextpow(2, 2 * n_records))
    slot_mask = table_length - 1
    slots = PropagationBase._ensurecapacity!(task.slots, table_length)
    fill!(view(slots, 1:table_length), zero(Int32))
    n_groups = 0

    for i in lo:hi
        hashbits = record_labels[i] % UInt64
        slot = Int(hashbits & (slot_mask % UInt64)) + 1
        group = Int(slots[slot])
        while group != 0 && group_hashes[group] != hashbits
            slot = (slot & slot_mask) + 1
            group = Int(slots[slot])
        end
        if group == 0
            n_groups += 1
            group = n_groups
            group_hashes[group] = hashbits
            group_starts[group] = 0
            slots[slot] = group
        end
        group_starts[group] += 1
        group_of[i-lo+1] = group
    end

    # the Pauli strings and coefficients of the records one group after the other
    next_start = 1
    for group in 1:n_groups
        n_here = group_starts[group]
        group_starts[group] = next_start
        next_start += n_here
    end
    group_starts[n_groups+1] = next_start
    group_terms = PropagationBase._ensurecapacity!(task.group_terms, n_records)
    group_coeffs = PropagationBase._ensurecapacity!(task.group_coeffs, n_records)
    for i in 1:n_records
        group = group_of[i]
        position = group_starts[group]
        group_terms[position] = record_terms[lo-1+i]
        group_coeffs[position] = record_coeffs[lo-1+i]
        group_starts[group] = position + 1
    end
    for group in n_groups:-1:1
        group_starts[group+1] = group_starts[group]
    end
    group_starts[1] = 1

    for group in 1:n_groups
        records = group_starts[group]:group_starts[group+1]-1
        @inline applytogroup!(sink, task, view(group_terms, records), view(group_coeffs, records))
    end
    return sink
end

# runs `f` for every source: the zones of a multi sum as its zones are run, and any other sources as tasks
_eachsource(f::F, ::PropagationBase.MultiSumStorage, prop_cache, n_sources::Int, thread::Bool) where {F} =
    PropagationBase._eachzone(f, prop_cache, thread)
_eachsource(f::F, ::PropagationBase.StorageType, prop_cache, n_sources::Int, thread::Bool) where {F} =
    PropagationBase._eachtask(f, n_sources)

# the number of zones that collect records: those of a multi sum, or one
_nrecordzones(::PropagationBase.MultiSumStorage, prop_cache) = nzones(prop_cache)
_nrecordzones(::PropagationBase.StorageType, prop_cache) = 1


### Arrays

# Every task reads a range of the main arrays. Several tasks write into the main arrays at once, which do not grow while
# they do, so the arrays get room for half as many Pauli strings again first.
function _recordsources(::PropagationBase.ArrayStorage, prop_cache, workspace, thread::Bool)
    task_partitioner, n_tasks = PropagationBase._preparetasks(activesize(prop_cache), thread)
    return _arraysources(prop_cache, workspace, task_partitioner, n_tasks)
end

function _arraysources(prop_cache, workspace, task_partitioner, n_tasks::Int)
    n_terms = activesize(prop_cache)
    if n_tasks > 1
        PropagationBase._ensurecapacity!(prop_cache, n_terms + n_terms ÷ 2)
    end
    main_terms, main_coeffs, _, _ = PropagationBase._mainauxarrays(prop_cache)
    PropagationBase._checkfits(n_terms, main_terms, main_coeffs)
    labels = _ensurecount!(workspace.source_labels, n_tasks)
    function source(task_id)
        chunk = task_partitioner[task_id]
        return zip(view(main_terms, chunk), view(main_coeffs, chunk)), PropagationBase._ensurecapacity!(labels[task_id], length(chunk))
    end
    return [source(task_id) for task_id in 1:n_tasks]
end

# The records take the place of the auxiliary arrays and the indices, which hold nothing the sum needs.
function _recordarrays!(::PropagationBase.ArrayStorage, prop_cache, workspace, n_records::Vector{Int})
    _, _, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(prop_cache)
    record_labels = indices(prop_cache)
    PropagationBase._checkfits(only(n_records), aux_terms, aux_coeffs)
    PropagationBase._checkfits(only(n_records), record_labels, record_labels)
    return [aux_terms], [aux_coeffs], [record_labels]
end

# One task writes into the main arrays alone, and several share them, each through a buffer of its own.
function _passsinks!(::PropagationBase.ArrayStorage, prop_cache, workspace, tasks, n_tasks::Int)
    if n_tasks == 1
        return [ArraySink(prop_cache)]
    end
    n_reserved = Threads.Atomic{Int}(0)
    task_sink(task_id) = ArraySink(prop_cache, _sinkbuffer!(tasks[task_id], 1, _SINK_BUFFER_LENGTH)..., n_reserved)
    return [task_sink(task_id) for task_id in 1:n_tasks]
end

function _collectpass!(::PropagationBase.ArrayStorage, prop_cache, sinks, thread::Bool)
    _collectsinks!(prop_cache, sinks)
    return prop_cache
end


### Any sum that is iterated and added to

# The one task reads the sum and keeps the records in the workspace. They hold all of the sum, so the sum is emptied and
# takes what the pass makes, instead of a second sum of its size.
function _recordsources(::PropagationBase.DictStorage, prop_cache, workspace, thread::Bool)
    PropagationBase._checkauxempty(prop_cache)
    main_sum = mainsum(prop_cache)
    return [(main_sum, PropagationBase._ensurecapacity!(first(_ensurecount!(workspace.source_labels, 1)), length(main_sum)))]
end

_recordarrays!(::PropagationBase.DictStorage, prop_cache, workspace, n_records::Vector{Int}) =
    _zonerecords!(PropagationBase.DictStorage(), workspace, nothing, n_records)

function _passsinks!(::PropagationBase.DictStorage, prop_cache, workspace, tasks, n_tasks::Int)
    main_sum = mainsum(prop_cache)
    empty!(main_sum)
    return [main_sum]
end

_collectpass!(::PropagationBase.DictStorage, prop_cache, sinks, thread::Bool) = prop_cache


### Multi sums

# Every zone is a task that reads its own Pauli strings. The records of a class are collected in the zone that the hash of
# the class picks, in the auxiliary arrays of a zone of arrays and in the workspace otherwise, and what the tasks make goes
# to the zones that own it: into the main arrays of zones of arrays, and through the outboxes otherwise.
function _recordsources(::PropagationBase.MultiSumStorage, prop_cache, workspace, thread::Bool)
    PropagationBase._checkauxempty(prop_cache)
    zone_caches = zonecaches(prop_cache)
    labels = _ensurecount!(workspace.source_labels, length(zone_caches))
    return [(zonecache, PropagationBase._ensurecapacity!(labels[zone_id], length(zonecache)))
            for (zone_id, zonecache) in enumerate(zone_caches)]
end

_recordarrays!(::PropagationBase.MultiSumStorage, prop_cache, workspace, n_records::Vector{Int}) =
    _zonerecords!(PropagationBase.zonestorage(prop_cache), workspace, zonecaches(prop_cache), n_records)

_passsinks!(::PropagationBase.MultiSumStorage, prop_cache, workspace, tasks, n_zones::Int) =
    _zonesinks!(PropagationBase.zonestorage(prop_cache), prop_cache, tasks)

# the records hold all of the sum, so every zone ends up with what the pass made for it alone
function _collectpass!(::PropagationBase.MultiSumStorage, prop_cache, sinks, thread::Bool)
    zone_storage = PropagationBase.zonestorage(prop_cache)
    collect_zone!(zone_id) = _collectzone!(zone_storage, prop_cache, zone_id, sinks)
    PropagationBase._eachzone(collect_zone!, prop_cache, thread)
    PropagationBase._syncsums!(prop_cache)
    return prop_cache
end

# Every task writes through a sink for every zone of arrays, which the sinks of all tasks share, and any other zones
# through its outbox.
function _zonesinks!(::PropagationBase.ArrayStorage, prop_cache, tasks)
    zone_caches = zonecaches(prop_cache)
    n_zones = length(zone_caches)
    n_reserved = [Threads.Atomic{Int}(0) for _ in 1:n_zones]

    # the buffers of a task take about as much memory whatever the number of zones
    buffer_length = max(256, 4 * _SINK_BUFFER_LENGTH ÷ n_zones)
    function task_sinks(task_id)
        zone_sink(zone_id) = ArraySink(zone_caches[zone_id], _sinkbuffer!(tasks[task_id], zone_id, buffer_length)..., n_reserved[zone_id])
        return ZoneSinks(zonemap(prop_cache), [zone_sink(zone_id) for zone_id in 1:n_zones])
    end
    return [task_sinks(task_id) for task_id in 1:n_zones]
end

_zonesinks!(::PropagationBase.StorageType, prop_cache, tasks) = outboxes(prop_cache)

# A zone of arrays takes what the sinks of every task wrote for it. Its Pauli strings are all different, since every
# class was rotated by one task.
function _collectzone!(::PropagationBase.ArrayStorage, prop_cache, zone_id::Int, sinks)
    zonecache = zonecaches(prop_cache)[zone_id]
    _collectsinks!(zonecache, [zone_sinks.sinks[zone_id] for zone_sinks in sinks])
    return zonecache
end

# any other zone is emptied and takes what the outboxes hold for it
function _collectzone!(::PropagationBase.StorageType, prop_cache, zone_id::Int, sinks)
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
    zone_arrays = (workspace.zone_terms, workspace.zone_coeffs, workspace.zone_labels)
    for arrays in zone_arrays
        _ensurecount!(arrays, length(n_records))
        for zone_id in eachindex(n_records)
            PropagationBase._ensurecapacity!(arrays[zone_id], n_records[zone_id])
        end
    end
    return zone_arrays
end


### Records

# A record is a Pauli string, its coefficient and its label, a hash of its class.

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

# The zone that collects a record, and its partition among those of all zones: the highest `n_bits` bits of the hash of
# its class pick the partition, and the highest `n_zone_bits` of those the zone.
@inline function _zonepartitionof(label::Int, n_bits::Int, n_zone_bits::Int)
    partition = Int((label % UInt64) >> (64 - n_bits))
    return (partition >> (n_bits - n_zone_bits)) + 1, partition + 1
end

"""
    _countrecords!(labelof, partition_counts, n_bits, n_zone_bits, source, labels)

Counts the records of each partition that the Pauli strings of `source` make, and keeps their labels `labelof(pstr)` in
`labels`, in the order of `source`.
"""
function _countrecords!(labelof::L, partition_counts, n_bits::Int, n_zone_bits::Int, source, labels::AbstractVector{Int}) where {L}
    source_index = 0
    for (pstr, _) in source
        label = @inline labelof(pstr)
        source_index += 1
        labels[source_index] = label
        _, partition = _zonepartitionof(label, n_bits, n_zone_bits)
        partition_counts[partition] += 1
    end
    return partition_counts
end

"""
    _partitionstarts!(partition_counts)

Turns the number of records of every partition and source into the index at which the source writes its first record of
the partition, and returns where every partition starts.
The records of a partition lie one source after the other, from `partition_starts[p]` to `partition_starts[p+1] - 1`.
"""
function _partitionstarts!(partition_counts::Matrix{Int})
    n_partitions, n_sources = size(partition_counts)
    partition_starts = Vector{Int}(undef, n_partitions + 1)

    next_start = 1
    for partition in 1:n_partitions
        partition_starts[partition] = next_start
        for source_id in 1:n_sources
            n_here = partition_counts[partition, source_id]
            partition_counts[partition, source_id] = next_start
            next_start += n_here
        end
    end
    partition_starts[n_partitions+1] = next_start

    return partition_starts
end

"""
    _writerecords!(zone_terms, zone_coeffs, zone_labels, zone_starts, cursors, n_bits, n_zone_bits, source, labels)

Writes a record for every Pauli string of `source` and its coefficient, with its label from `labels`, to where
`cursors` points for its partition, in the arrays of the zone that the partition belongs to.
"""
function _writerecords!(zone_terms::Vector{Vector{TT}}, zone_coeffs::Vector{Vector{CT}}, zone_labels::Vector{Vector{Int}},
    zone_starts::Vector{Int}, cursors, n_bits::Int, n_zone_bits::Int, source, labels::AbstractVector{Int}) where {TT,CT}

    source_index = 0
    for (pstr, coeff) in source
        source_index += 1
        label = labels[source_index]
        zone_id, partition = _zonepartitionof(label, n_bits, n_zone_bits)

        index = cursors[partition] - zone_starts[zone_id] + 1
        zone_terms[zone_id][index] = pstr
        zone_coeffs[zone_id][index] = coeff
        zone_labels[zone_id][index] = label
        cursors[partition] += 1
    end

    return zone_terms
end

# A hash of a Pauli string. The limbs are mixed independently of each other, so that the hash of a wide Pauli string is
# no chain over its limbs. A product keeps the highest bit of a limb only in its own highest bit, so the high half of
# every limb is folded onto its low half first, or strings that differ only in which limb has that bit set would collide.
@inline function _hashbits(pstr)
    limbs = _limbs(pstr)
    folded = zero(UInt64)
    for i in eachindex(limbs)
        folded ⊻= (limbs[i] ⊻ (limbs[i] >> 32)) * _foldfactor(i)
    end
    return PropagationBase._mix64(folded)
end

# an odd factor for every limb
const _FOLD_FACTORS = ntuple(i -> PropagationBase._mix64(UInt64(i)) | one(UInt64), 64)

@inline _foldfactor(i::Int) = _FOLD_FACTORS[((i-1)&63)+1] + ((UInt64(i - 1) >> 6) << 1)

# a Pauli string as 64-bit limbs, the lowest first
@inline _limbs(pstr::NTupleInteger) = pstr.limbs
@inline _limbs(pstr::UInt128) = (pstr % UInt64, (pstr >> 64) % UInt64)
@inline _limbs(pstr::Union{UInt8,UInt16,UInt32,UInt64}) = (pstr % UInt64,)


### Where a pass writes the Pauli strings it makes

# A task writes what it makes to a sink, one Pauli string at a time through `_emit!`: an `ArraySink` to a sum held in
# arrays, `ZoneSinks` to the zones of a multi sum held in arrays, and to any other sum the sum itself, which adds them.

# a sink that shares the arrays of a sum copies its buffer into them whenever it holds this many Pauli strings
const _SINK_BUFFER_LENGTH = 4096

"""
    ArraySink

Where one task writes the Pauli strings it makes into the main arrays of a cache.
A sink that is the only one to write to the arrays writes into them directly and grows them as they fill.
Sinks that share the arrays each collect what they make in a buffer, and copy it into a range of the arrays that they reserve.
Shared arrays do not grow, so a sink that finds them full keeps what it makes in its buffer until `_collectsinks!` copies it.
"""
mutable struct ArraySink{TT,CT,PC}
    prop_cache::PC
    # the buffer of a shared sink, or the main arrays of the cache
    terms::Vector{TT}
    coeffs::Vector{CT}
    n_written::Int
    # how much of the main arrays the sinks that share them have taken
    n_reserved::Threads.Atomic{Int}
    is_shared::Bool
    is_full::Bool
end

# the only sink of the main arrays of `prop_cache`
function ArraySink(prop_cache)
    main_terms, main_coeffs, _, _ = PropagationBase._mainauxarrays(prop_cache)
    return ArraySink(prop_cache, main_terms, main_coeffs, 0, Threads.Atomic{Int}(0), false, false)
end

# one of the sinks that share the main arrays of `prop_cache`, with a buffer of its own
ArraySink(prop_cache, buffer_terms, buffer_coeffs, n_reserved::Threads.Atomic{Int}) =
    ArraySink(prop_cache, buffer_terms, buffer_coeffs, 0, n_reserved, true, false)

@inline function _emit!(sink::ArraySink, pstr, coeff)
    n_written = sink.n_written + 1
    if n_written > length(sink.terms) || n_written > length(sink.coeffs)
        _makeroom!(sink)
        n_written = sink.n_written + 1
    end
    sink.terms[n_written] = pstr
    sink.coeffs[n_written] = coeff
    sink.n_written = n_written
    return
end

# Room for one more Pauli string: the only sink grows the main arrays, and a shared sink copies its buffer into them, or
# grows its buffer when they are full.
function _makeroom!(sink::ArraySink)
    if sink.is_shared
        _flush!(sink)
        PropagationBase._ensurecapacity!(sink.terms, sink.n_written + 1)
        PropagationBase._ensurecapacity!(sink.coeffs, sink.n_written + 1)
    else
        PropagationBase._ensurecapacity!(sink.prop_cache, sink.n_written + 1)
        main_terms, main_coeffs, _, _ = PropagationBase._mainauxarrays(sink.prop_cache)
        sink.terms = main_terms
        sink.coeffs = main_coeffs
    end
    return sink
end

# Copies the buffer of a shared sink into a range of the main arrays that no other sink writes to, unless they are full.
function _flush!(sink::ArraySink)
    n_buffered = sink.n_written
    if n_buffered == 0 || sink.is_full
        return sink
    end

    main_terms, main_coeffs, _, _ = PropagationBase._mainauxarrays(sink.prop_cache)
    n_room = min(length(main_terms), length(main_coeffs))
    n_taken = sink.n_reserved[]
    while true
        if n_taken + n_buffered > n_room
            sink.is_full = true
            return sink
        end
        n_seen = Threads.atomic_cas!(sink.n_reserved, n_taken, n_taken + n_buffered)
        if n_seen == n_taken
            break
        end
        n_taken = n_seen
    end

    copyto!(main_terms, n_taken + 1, sink.terms, 1, n_buffered)
    copyto!(main_coeffs, n_taken + 1, sink.coeffs, 1, n_buffered)
    sink.n_written = 0
    return sink
end

# a shared sink copies what it holds into the arrays at the end of its task, as far as they have room
function _finish!(sink::ArraySink)
    if sink.is_shared
        _flush!(sink)
    end
    return sink
end

"""
    _collectsinks!(prop_cache, sinks)

Makes the Pauli strings that `sinks` wrote into the main arrays of `prop_cache` its sum: what the only sink wrote, or
what shared sinks copied into the arrays, followed by what their buffers kept when the arrays were full.
"""
function _collectsinks!(prop_cache, sinks::Vector{<:ArraySink})
    if !first(sinks).is_shared
        _setsum!(prop_cache, only(sinks).n_written)
        return prop_cache
    end

    n_written = first(sinks).n_reserved[]
    n_kept = sum(sink -> sink.n_written, sinks)
    if n_kept > 0
        PropagationBase._ensurecapacity!(prop_cache, n_written + n_kept)
        main_terms, main_coeffs, _, _ = PropagationBase._mainauxarrays(prop_cache)
        PropagationBase._checkfits(n_written + n_kept, main_terms, main_coeffs)
        for sink in sinks
            copyto!(main_terms, n_written + 1, sink.terms, 1, sink.n_written)
            copyto!(main_coeffs, n_written + 1, sink.coeffs, 1, sink.n_written)
            n_written += sink.n_written
        end
    end
    _setsum!(prop_cache, n_written)
    return prop_cache
end

# the first `n_written` Pauli strings of the main arrays, which are all different but not sorted, become the sum
function _setsum!(prop_cache, n_written::Int)
    setactivesize!(prop_cache, n_written)
    PropagationBase.setsortedprefix!(mainsum(prop_cache), 0)
    return prop_cache
end

# The sinks of one task for the zones of a multi sum, one per zone: a Pauli string goes to the sink of the zone that owns it.
struct ZoneSinks{ZM,S<:ArraySink}
    zone_map::ZM
    sinks::Vector{S}
end

@inline function _emit!(zone_sinks::ZoneSinks, pstr, coeff)
    _emit!(zone_sinks.sinks[PropagationBase.zoneof(zone_sinks.zone_map, pstr)], pstr, coeff)
    return
end

function _finish!(zone_sinks::ZoneSinks)
    foreach(_finish!, zone_sinks.sinks)
    return zone_sinks
end

# a term sum takes the Pauli strings one by one: a Pauli sum adds them, the outbox of a multi sum passes them to the zone that owns them
@inline function _emit!(term_sum::AbstractTermSum, pstr, coeff)
    push!(term_sum, pstr, coeff)
    return
end

_finish!(term_sum::AbstractTermSum) = term_sum
