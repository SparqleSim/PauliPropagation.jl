###
##
# Applying a function to the terms of a sum grouped by a label, in one pass over the sum. Every term becomes a record,
# the records are partitioned by their label, and partition by partition the records of every label are handed to the
# function, which writes what it makes to a sink.
##
###

# whether the terms of the sum can be grouped in one pass: a sum held in arrays needs its main arrays to be vectors
_canapplytogroups(prop_cache::AbstractPropagationCache) = _canapplytogroups(StorageType(prop_cache), prop_cache)
_canapplytogroups(::StorageType, prop_cache) = false
_canapplytogroups(::DictStorage, prop_cache) = true

function _canapplytogroups(::ArrayStorage, prop_cache)
    main_terms, main_coeffs, _, _ = _mainauxarrays(prop_cache)
    return main_terms isa Vector && main_coeffs isa Vector
end

function _canapplytogroups(storage::MultiSumStorage, prop_cache)
    return all(zonecache -> _canapplytogroups(storage.zonestorage, zonecache), zonecaches(prop_cache))
end

"""
    _applytogroups!(labelof, applytogroup!, prop_cache, workspace, sources, thread)

Groups the terms of the sum by their label `labelof(term)`, a 64-bit hash, and calls
`applytogroup!(sink, scratch, group_terms, group_coeffs)` once for every group, with all of its terms and their
coefficients and the scratch of the task. What the calls write to their sinks replaces the sum, and no two of those
terms may be the same.
Every term becomes a record: the records are counted by partition, written partition by partition, and the
partitions are grouped by tasks that each take the next partition that no task has taken yet.
The storage of the sum decides what every task reads (`sources`, from `_recordsources`), where the records are kept
(`_recordarrays!`), where the tasks write (`_passsinks!`) and how that becomes the sum (`_collectpass!`).
"""
function _applytogroups!(labelof::L, applytogroup!::G, prop_cache::AbstractPropagationCache, workspace, sources,
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
    record_bytes = sizeof(termtype(prop_cache)) + sizeof(coefftype(prop_cache)) + sizeof(Int)
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

Groups the records `lo` to `hi` by their label and calls `applytogroup!(sink, task.scratch, group_terms, group_coeffs)`
for every group, with the terms and coefficients of its records next to each other in the task's workspace.
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
    group_of = _ensurecapacity!(task.group_of, n_records)
    group_hashes = _ensurecapacity!(task.group_hashes, n_records)
    group_starts = _ensurecapacity!(task.group_starts, n_records + 1)

    # the groups found so far, through their hash
    table_length = max(16, nextpow(2, 2 * n_records))
    slot_mask = table_length - 1
    slots = _ensurecapacity!(task.slots, table_length)
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

    # the terms and coefficients of the records one group after the other
    next_start = 1
    for group in 1:n_groups
        n_here = group_starts[group]
        group_starts[group] = next_start
        next_start += n_here
    end
    group_starts[n_groups+1] = next_start
    group_terms = _ensurecapacity!(task.group_terms, n_records)
    group_coeffs = _ensurecapacity!(task.group_coeffs, n_records)
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
        @inline applytogroup!(sink, task.scratch, view(group_terms, records), view(group_coeffs, records))
    end
    return sink
end

# runs `f` for every source: the zones of a multi sum as its zones are run, and any other sources as tasks
_eachsource(f::F, ::MultiSumStorage, prop_cache, n_sources::Int, thread::Bool) where {F} =
    _eachzone(f, prop_cache, thread)
_eachsource(f::F, ::StorageType, prop_cache, n_sources::Int, thread::Bool) where {F} =
    _eachtask(f, n_sources)

# the number of zones that collect records: those of a multi sum, or one
_nrecordzones(::MultiSumStorage, prop_cache) = nzones(prop_cache)
_nrecordzones(::StorageType, prop_cache) = 1


### Arrays

# Every task reads a range of the main arrays. Several tasks write into the main arrays at once, which do not grow while
# they do, so the arrays get room for half as many terms again first.
function _recordsources(::ArrayStorage, prop_cache, workspace, thread::Bool)
    task_partitioner, n_tasks = _preparetasks(activesize(prop_cache), thread)
    return _arraysources(prop_cache, workspace, task_partitioner, n_tasks)
end

function _arraysources(prop_cache, workspace, task_partitioner, n_tasks::Int)
    n_terms = activesize(prop_cache)
    if n_tasks > 1
        _ensurecapacity!(prop_cache, n_terms + n_terms ÷ 2)
    end
    main_terms, main_coeffs, _, _ = _mainauxarrays(prop_cache)
    _checkfits(n_terms, main_terms, main_coeffs)
    labels = _ensurecount!(workspace.source_labels, n_tasks)
    function source(task_id)
        chunk = task_partitioner[task_id]
        return zip(view(main_terms, chunk), view(main_coeffs, chunk)), _ensurecapacity!(labels[task_id], length(chunk))
    end
    return [source(task_id) for task_id in 1:n_tasks]
end

# The records take the place of the auxiliary arrays and the indices, which hold nothing the sum needs.
function _recordarrays!(::ArrayStorage, prop_cache, workspace, n_records::Vector{Int})
    _, _, aux_terms, aux_coeffs = _mainauxarrays(prop_cache)
    record_labels = indices(prop_cache)
    _checkfits(only(n_records), aux_terms, aux_coeffs)
    _checkfits(only(n_records), record_labels, record_labels)
    return [aux_terms], [aux_coeffs], [record_labels]
end

# One task writes into the main arrays alone, and several share them, each through a buffer of its own.
function _passsinks!(::ArrayStorage, prop_cache, workspace, tasks, n_tasks::Int)
    if n_tasks == 1
        return [ArraySink(prop_cache)]
    end
    n_reserved = Threads.Atomic{Int}(0)
    task_sink(task_id) = ArraySink(prop_cache, _sinkbuffer!(tasks[task_id], 1, _SINK_BUFFER_LENGTH)..., n_reserved)
    return [task_sink(task_id) for task_id in 1:n_tasks]
end

function _collectpass!(::ArrayStorage, prop_cache, sinks, thread::Bool)
    _collectsinks!(prop_cache, sinks)
    return prop_cache
end


### Any sum that is iterated and added to

# The one task reads the sum and keeps the records in the workspace. They hold all of the sum, so the sum is emptied and
# takes what the pass makes, instead of a second sum of its size.
function _recordsources(::DictStorage, prop_cache, workspace, thread::Bool)
    _checkauxempty(prop_cache)
    main_sum = mainsum(prop_cache)
    return [(main_sum, _ensurecapacity!(first(_ensurecount!(workspace.source_labels, 1)), length(main_sum)))]
end

_recordarrays!(::DictStorage, prop_cache, workspace, n_records::Vector{Int}) =
    _zonerecords!(DictStorage(), workspace, nothing, n_records)

function _passsinks!(::DictStorage, prop_cache, workspace, tasks, n_tasks::Int)
    main_sum = mainsum(prop_cache)
    empty!(main_sum)
    return [main_sum]
end

_collectpass!(::DictStorage, prop_cache, sinks, thread::Bool) = prop_cache


### Multi sums

# Every zone is a task that reads its own terms. The records of a group are collected in the zone that their label
# picks, in the auxiliary arrays of a zone of arrays and in the workspace otherwise, and what the tasks make goes
# to the zones that own it: into the main arrays of zones of arrays, and through the outboxes otherwise.
function _recordsources(::MultiSumStorage, prop_cache, workspace, thread::Bool)
    _checkauxempty(prop_cache)
    zone_caches = zonecaches(prop_cache)
    labels = _ensurecount!(workspace.source_labels, length(zone_caches))
    return [(zonecache, _ensurecapacity!(labels[zone_id], length(zonecache)))
            for (zone_id, zonecache) in enumerate(zone_caches)]
end

_recordarrays!(::MultiSumStorage, prop_cache, workspace, n_records::Vector{Int}) =
    _zonerecords!(zonestorage(prop_cache), workspace, zonecaches(prop_cache), n_records)

_passsinks!(::MultiSumStorage, prop_cache, workspace, tasks, n_zones::Int) =
    _zonesinks!(zonestorage(prop_cache), prop_cache, tasks)

# the records hold all of the sum, so every zone ends up with what the pass made for it alone
function _collectpass!(::MultiSumStorage, prop_cache, sinks, thread::Bool)
    zone_storage = zonestorage(prop_cache)
    collect_zone!(zone_id) = _collectzone!(zone_storage, prop_cache, zone_id, sinks)
    _eachzone(collect_zone!, prop_cache, thread)
    _syncsums!(prop_cache)
    return prop_cache
end

# Every task writes through a sink for every zone of arrays, which the sinks of all tasks share, and any other zones
# through its outbox.
function _zonesinks!(::ArrayStorage, prop_cache, tasks)
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

_zonesinks!(::StorageType, prop_cache, tasks) = outboxes(prop_cache)

# A zone of arrays takes what the sinks of every task wrote for it. Its terms are all different, since every
# group was applied to by one task.
function _collectzone!(::ArrayStorage, prop_cache, zone_id::Int, sinks)
    zonecache = zonecaches(prop_cache)[zone_id]
    _collectsinks!(zonecache, [zone_sinks.sinks[zone_id] for zone_sinks in sinks])
    return zonecache
end

# any other zone is emptied and takes what the outboxes hold for it
function _collectzone!(::StorageType, prop_cache, zone_id::Int, sinks)
    empty!(zonecaches(prop_cache)[zone_id])
    _deliverto!(prop_cache, zone_id)
    return zonecaches(prop_cache)[zone_id]
end

# Arrays for the records that every zone collects, `n_records[zone_id]` of them.
# A zone of arrays keeps them in the auxiliary arrays and `indices` of its cache, any other zone in the workspace.
function _zonerecords!(::ArrayStorage, workspace, zone_caches, n_records::Vector{Int})
    for (zone_id, zonecache) in enumerate(zone_caches)
        _ensurecapacity!(zonecache, n_records[zone_id])
    end
    zone_terms = [terms(auxsum(zonecache)) for zonecache in zone_caches]
    zone_coeffs = [coefficients(auxsum(zonecache)) for zonecache in zone_caches]
    zone_labels = [indices(zonecache) for zonecache in zone_caches]
    return zone_terms, zone_coeffs, zone_labels
end

function _zonerecords!(::StorageType, workspace, zone_caches, n_records::Vector{Int})
    zone_arrays = (workspace.zone_terms, workspace.zone_coeffs, workspace.zone_labels)
    for arrays in zone_arrays
        _ensurecount!(arrays, length(n_records))
        for zone_id in eachindex(n_records)
            _ensurecapacity!(arrays[zone_id], n_records[zone_id])
        end
    end
    return zone_arrays
end


### Records

# A record is a term, its coefficient and its label, a 64-bit hash.

# the records of a partition take about this many bytes
const _PARTITION_BYTES = 1 << 18

const _MAX_PARTITION_BITS = 12

# What the records of a partition make is held next to them while its groups are applied to, so a record counts this
# many times its size when the partitions are sized.
const _RECORD_BYTES_FACTOR = 4

# The number of bits of the hash that pick the partition of a record, so that a partition is small enough to be worked
# on within the cache.
function _partitionbits(n_records::Int, record_bytes::Int)
    n_partitions = cld(n_records * _RECORD_BYTES_FACTOR * record_bytes, _PARTITION_BYTES)
    if n_partitions <= 1
        return 0
    end
    return min(_MAX_PARTITION_BITS, 8 * sizeof(Int) - leading_zeros(n_partitions - 1))
end

# The zone that collects a record, and its partition among those of all zones: the highest `n_bits` bits of its label pick
# the partition, and the highest `n_zone_bits` of those the zone.
@inline function _zonepartitionof(label::Int, n_bits::Int, n_zone_bits::Int)
    partition = Int((label % UInt64) >> (64 - n_bits))
    return (partition >> (n_bits - n_zone_bits)) + 1, partition + 1
end

"""
    _countrecords!(labelof, partition_counts, n_bits, n_zone_bits, source, labels)

Counts the records of each partition that the terms of `source` make, and keeps their labels `labelof(term)` in
`labels`, in the order of `source`.
"""
function _countrecords!(labelof::L, partition_counts, n_bits::Int, n_zone_bits::Int, source, labels::AbstractVector{Int}) where {L}
    source_index = 0
    for (term, _) in source
        label = @inline labelof(term)
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

Writes a record for every term of `source` and its coefficient, with its label from `labels`, to where
`cursors` points for its partition, in the arrays of the zone that the partition belongs to.
"""
function _writerecords!(zone_terms::Vector{Vector{TT}}, zone_coeffs::Vector{Vector{CT}}, zone_labels::Vector{Vector{Int}},
    zone_starts::Vector{Int}, cursors, n_bits::Int, n_zone_bits::Int, source, labels::AbstractVector{Int}) where {TT,CT}

    source_index = 0
    for (term, coeff) in source
        source_index += 1
        label = labels[source_index]
        zone_id, partition = _zonepartitionof(label, n_bits, n_zone_bits)

        index = cursors[partition] - zone_starts[zone_id] + 1
        zone_terms[zone_id][index] = term
        zone_coeffs[zone_id][index] = coeff
        zone_labels[zone_id][index] = label
        cursors[partition] += 1
    end

    return zone_terms
end

# A hash of a term. The limbs are mixed independently of each other, so that the hash of a wide term is
# no chain over its limbs. A product keeps the highest bit of a limb only in its own highest bit, so the high half of
# every limb is folded onto its low half first, or strings that differ only in which limb has that bit set would collide.
@inline function _hashbits(term)
    limbs = _limbs(term)
    folded = zero(UInt64)
    for i in eachindex(limbs)
        folded ⊻= (limbs[i] ⊻ (limbs[i] >> 32)) * _foldfactor(i)
    end
    return _mix64(folded)
end

# an odd factor for every limb
const _FOLD_FACTORS = ntuple(i -> _mix64(UInt64(i)) | one(UInt64), 64)

@inline _foldfactor(i::Int) = _FOLD_FACTORS[((i-1)&63)+1] + ((UInt64(i - 1) >> 6) << 1)

# a term as 64-bit limbs, the lowest first
@inline _limbs(term::NTupleInteger) = term.limbs
@inline _limbs(term::UInt128) = (term % UInt64, (term >> 64) % UInt64)
@inline _limbs(term::Union{UInt8,UInt16,UInt32,UInt64}) = (term % UInt64,)


### Where a pass writes the terms it makes

# A task writes what it makes to a sink, one term at a time through `_emit!`: an `ArraySink` to a sum held in
# arrays, `ZoneSinks` to the zones of a multi sum held in arrays, and to any other sum the sum itself, which adds them.

# a sink that shares the arrays of a sum copies its buffer into them whenever it holds this many terms
const _SINK_BUFFER_LENGTH = 4096

"""
    ArraySink

Where one task writes the terms it makes into the main arrays of a cache.
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
    main_terms, main_coeffs, _, _ = _mainauxarrays(prop_cache)
    return ArraySink(prop_cache, main_terms, main_coeffs, 0, Threads.Atomic{Int}(0), false, false)
end

# one of the sinks that share the main arrays of `prop_cache`, with a buffer of its own
ArraySink(prop_cache, buffer_terms, buffer_coeffs, n_reserved::Threads.Atomic{Int}) =
    ArraySink(prop_cache, buffer_terms, buffer_coeffs, 0, n_reserved, true, false)

@inline function _emit!(sink::ArraySink, term, coeff)
    n_written = sink.n_written + 1
    if n_written > length(sink.terms) || n_written > length(sink.coeffs)
        _makeroom!(sink)
        n_written = sink.n_written + 1
    end
    sink.terms[n_written] = term
    sink.coeffs[n_written] = coeff
    sink.n_written = n_written
    return
end

# Room for one more term: the only sink grows the main arrays, and a shared sink copies its buffer into them, or
# grows its buffer when they are full.
function _makeroom!(sink::ArraySink)
    if sink.is_shared
        _flush!(sink)
        _ensurecapacity!(sink.terms, sink.n_written + 1)
        _ensurecapacity!(sink.coeffs, sink.n_written + 1)
    else
        _ensurecapacity!(sink.prop_cache, sink.n_written + 1)
        main_terms, main_coeffs, _, _ = _mainauxarrays(sink.prop_cache)
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

    main_terms, main_coeffs, _, _ = _mainauxarrays(sink.prop_cache)
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

Makes the terms that `sinks` wrote into the main arrays of `prop_cache` its sum: what the only sink wrote, or
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
        _ensurecapacity!(prop_cache, n_written + n_kept)
        main_terms, main_coeffs, _, _ = _mainauxarrays(prop_cache)
        _checkfits(n_written + n_kept, main_terms, main_coeffs)
        for sink in sinks
            copyto!(main_terms, n_written + 1, sink.terms, 1, sink.n_written)
            copyto!(main_coeffs, n_written + 1, sink.coeffs, 1, sink.n_written)
            n_written += sink.n_written
        end
    end
    _setsum!(prop_cache, n_written)
    return prop_cache
end

# the first `n_written` terms of the main arrays, which are all different but not sorted, become the sum
function _setsum!(prop_cache, n_written::Int)
    setactivesize!(prop_cache, n_written)
    setsortedprefix!(mainsum(prop_cache), 0)
    return prop_cache
end

# The sinks of one task for the zones of a multi sum, one per zone: a term goes to the sink of the zone that owns it.
struct ZoneSinks{ZM,S<:ArraySink}
    zone_map::ZM
    sinks::Vector{S}
end

@inline function _emit!(zone_sinks::ZoneSinks, term, coeff)
    _emit!(zone_sinks.sinks[zoneof(zone_sinks.zone_map, term)], term, coeff)
    return
end

function _finish!(zone_sinks::ZoneSinks)
    foreach(_finish!, zone_sinks.sinks)
    return zone_sinks
end

# a term sum takes the terms one by one: a sum adds them, the outbox of a multi sum passes them to the zone that owns them
@inline function _emit!(term_sum::AbstractTermSum, term, coeff)
    push!(term_sum, term, coeff)
    return
end

_finish!(term_sum::AbstractTermSum) = term_sum
