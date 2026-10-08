###
##
# Applying a function to the terms of a sum label by label, in one pass over the sum. Every term is copied with its
# coefficient and label into the batch that its label picks, and batch by batch the terms of every label are handed to
# the function, which writes what it makes to a sink.
##
###

# whether the terms of the sum can be sorted by label in one pass: a sum held in arrays needs its main arrays to be vectors
_canapplytolabels(prop_cache::AbstractPropagationCache) = _canapplytolabels(StorageType(prop_cache), prop_cache)
_canapplytolabels(::StorageType, prop_cache) = false
_canapplytolabels(::DictStorage, prop_cache) = true

function _canapplytolabels(::ArrayStorage, prop_cache)
    main_terms, main_coeffs, _, _ = _mainauxarrays(prop_cache)
    return main_terms isa Vector && main_coeffs isa Vector
end

function _canapplytolabels(storage::MultiSumStorage, prop_cache)
    return all(zonecache -> _canapplytolabels(storage.zonestorage, zonecache), zonecaches(prop_cache))
end

"""
    _applytolabels!(labelof, applytolabel!, prop_cache, workspace, sources, thread)

Sorts the terms of the sum by their label `labelof(term)`, a 64-bit hash, and calls
`applytolabel!(sink, scratch, label_terms, label_coeffs)` once for every label, with all of its terms and their
coefficients and the scratch of the task. What the calls write to their sinks replaces the sum, and no two of those
terms may be the same.
Every term is copied with its coefficient and label into its batch: the terms of every batch are counted, then
copied, and the batches are worked through by tasks that each take the next batch that no task has taken yet.
The storage of the sum decides what every task reads (`sources`, from `_passsources`), where the batches are kept
(`_batcharrays!`), where the tasks write (`_passsinks!`) and how that becomes the sum (`_collectpass!`).
"""
Base.@nospecializeinfer function _applytolabels!(@nospecialize(labelof), @nospecialize(applytolabel!),
    prop_cache::AbstractPropagationCache, workspace, sources, thread::Bool)

    storage = StorageType(prop_cache)
    n_terms = length(prop_cache)
    n_sources = length(sources)
    tasks = _ensurecount!(workspace.tasks, n_sources)

    # The highest bits of a label pick the zone that collects the term, and the bits below them a batch of the
    # zone. Every source writes into the batches of all zones, so the most batches apply to those of all zones
    # together: a source that writes into more batches than its processor cache holds the ends of finds none of them there.
    n_zones = _npasszones(storage, prop_cache)
    zone_bits = trailing_zeros(n_zones)
    term_bytes = sizeof(termtype(prop_cache)) + sizeof(coefftype(prop_cache)) + sizeof(Int)
    n_bits = max(zone_bits, min(_MAX_BATCH_BITS, zone_bits + _batchbits(cld(n_terms, n_zones), term_bytes)))
    n_batches_per_zone = 1 << (n_bits - zone_bits)

    # For every batch and source, first how many terms the source sends to the batch, then where it copies the next
    # one. One row more than there are batches, so that the entries of a batch are not a power of two apart, which
    # would put them all into one set of the processor cache. Every source clears its own column, so that the columns
    # are cleared in parallel and each lies in the memory of the thread that reads the source.
    source_batch_indices = Matrix{Int}(undef, n_zones * n_batches_per_zone + 1, n_sources)
    function count_source!(source_id)
        source, labels = sources[source_id]
        _countbatchsizes!(labelof, fill!(view(source_batch_indices, :, source_id), 0), n_bits, zone_bits, source, labels)
    end
    _eachsource(count_source!, storage, prop_cache, n_sources, thread)

    # the batched terms of a zone are numbered from 1 on
    batch_starts = _batchstarts!(source_batch_indices)
    zone_starts = [batch_starts[(zone_id-1)*n_batches_per_zone+1] for zone_id in 1:n_zones+1]
    batched_terms, batched_coeffs, batched_labels = _batcharrays!(storage, prop_cache, workspace, diff(zone_starts))
    function write_source!(source_id)
        source, labels = sources[source_id]
        _copytobatches!(batched_terms, batched_coeffs, batched_labels, zone_starts, view(source_batch_indices, :, source_id),
            n_bits, zone_bits, source, labels)
    end
    _eachsource(write_source!, storage, prop_cache, n_sources, thread)

    sinks = _passsinks!(storage, prop_cache, workspace, tasks, n_sources)
    _applytobatches!(applytolabel!, sinks, tasks, batched_terms, batched_coeffs, batched_labels, batch_starts, zone_starts,
        n_batches_per_zone, storage, prop_cache, thread)
    _collectpass!(storage, prop_cache, sinks, thread)
    return prop_cache
end

# The labels differ widely in how many terms they have, so the batches do too: every task takes the batches of its own
# zone first and then those of the other zones, one at a time, each the next one that no task has taken yet.
function _applytobatches!(applytolabel!::G, sinks, tasks, batched_terms, batched_coeffs, batched_labels, batch_starts,
    zone_starts, n_batches_per_zone::Int, storage, prop_cache, thread::Bool) where {G}

    n_zones = length(zone_starts) - 1
    n_batches_taken = [Threads.Atomic{Int}(0) for _ in 1:n_zones]
    function apply_to_batches!(task_id)
        for offset in 0:n_zones-1
            zone_id = mod1(task_id + offset, n_zones)
            zone_start = zone_starts[zone_id]
            first_batch = (zone_id - 1) * n_batches_per_zone
            while true
                batch_in_zone = Threads.atomic_add!(n_batches_taken[zone_id], 1) + 1
                if batch_in_zone > n_batches_per_zone
                    break
                end
                batch = first_batch + batch_in_zone
                _applytobatch!(applytolabel!, sinks[task_id], tasks[task_id], batched_terms[zone_id], batched_coeffs[zone_id],
                    batched_labels[zone_id], batch_starts[batch] - zone_start + 1, batch_starts[batch+1] - zone_start)
            end
        end
        _finish!(sinks[task_id])
    end
    _eachsource(apply_to_batches!, storage, prop_cache, length(sinks), thread)
    return sinks
end

"""
    _applytobatch!(applytolabel!, sink, task, batched_terms, batched_coeffs, batched_labels, lo, hi)

Sorts the batched terms `lo` to `hi` by their label and calls
`applytolabel!(sink, task.scratch, label_terms, label_coeffs)` for every label, with its terms and their coefficients
next to each other in the task's workspace.
"""
function _applytobatch!(applytolabel!::G, sink, task, batched_terms::Vector{TT}, batched_coeffs::Vector{CT},
    batched_labels::Vector{Int}, lo::Int, hi::Int) where {G,TT,CT}

    if lo > hi
        return sink
    end
    if !(1 <= lo && hi <= min(length(batched_terms), length(batched_coeffs), length(batched_labels)))
        throw(ArgumentError("the terms $lo to $hi are not among the batched terms"))
    end
    n_terms = hi - lo + 1
    label_of = _ensurecapacity!(task.label_of, n_terms)
    label_values = _ensurecapacity!(task.label_values, n_terms)
    label_starts = _ensurecapacity!(task.label_starts, n_terms + 1)

    # the labels found so far, through their hash
    n_slots = max(16, nextpow(2, 2 * n_terms))
    slot_mask = n_slots - 1
    slots = _ensurecapacity!(task.slots, n_slots)
    fill!(view(slots, 1:n_slots), zero(Int32))
    n_labels = 0

    for i in lo:hi
        hashbits = batched_labels[i] % UInt64
        slot = Int(hashbits & (slot_mask % UInt64)) + 1
        label_index = Int(slots[slot])
        while label_index != 0 && label_values[label_index] != hashbits
            slot = (slot & slot_mask) + 1
            label_index = Int(slots[slot])
        end
        if label_index == 0
            n_labels += 1
            label_index = n_labels
            label_values[label_index] = hashbits
            label_starts[label_index] = 0
            slots[slot] = label_index
        end
        label_starts[label_index] += 1
        label_of[i-lo+1] = label_index
    end

    # the terms and coefficients one label after the other
    next_start = 1
    for label_index in 1:n_labels
        n_here = label_starts[label_index]
        label_starts[label_index] = next_start
        next_start += n_here
    end
    label_starts[n_labels+1] = next_start
    label_terms = _ensurecapacity!(task.label_terms, n_terms)
    label_coeffs = _ensurecapacity!(task.label_coeffs, n_terms)
    for i in 1:n_terms
        label_index = label_of[i]
        position = label_starts[label_index]
        label_terms[position] = batched_terms[lo-1+i]
        label_coeffs[position] = batched_coeffs[lo-1+i]
        label_starts[label_index] = position + 1
    end
    for label_index in n_labels:-1:1
        label_starts[label_index+1] = label_starts[label_index]
    end
    label_starts[1] = 1

    for label_index in 1:n_labels
        label_range = label_starts[label_index]:label_starts[label_index+1]-1
        @inline applytolabel!(sink, task.scratch, view(label_terms, label_range), view(label_coeffs, label_range))
    end
    return sink
end

# runs `f` for every source: the zones of a multi sum as its zones are run, and any other sources as tasks
_eachsource(f::F, ::MultiSumStorage, prop_cache, n_sources::Int, thread::Bool) where {F} =
    _eachzone(f, prop_cache, thread)
_eachsource(f::F, ::StorageType, prop_cache, n_sources::Int, thread::Bool) where {F} =
    _eachtask(f, n_sources)

# the number of zones that collect the batches: those of a multi sum, or one
_npasszones(::MultiSumStorage, prop_cache) = nzones(prop_cache)
_npasszones(::StorageType, prop_cache) = 1


### Arrays

# Every task reads a range of the main arrays. Several tasks write into the main arrays at once, which do not grow while
# they do, so the arrays get room for half as many terms again first.
function _passsources(::ArrayStorage, prop_cache, workspace, thread::Bool)
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

# The batches take the place of the auxiliary arrays and the indices, which hold nothing the sum needs.
function _batcharrays!(::ArrayStorage, prop_cache, workspace, n_zone_terms::Vector{Int})
    _, _, aux_terms, aux_coeffs = _mainauxarrays(prop_cache)
    batched_labels = indices(prop_cache)
    _checkfits(only(n_zone_terms), aux_terms, aux_coeffs)
    _checkfits(only(n_zone_terms), batched_labels, batched_labels)
    return [aux_terms], [aux_coeffs], [batched_labels]
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

# The one task reads the sum and keeps the batches in the workspace. They hold all of the sum, so the sum is emptied and
# takes what the pass makes, instead of a second sum of its size.
function _passsources(::DictStorage, prop_cache, workspace, thread::Bool)
    _checkauxempty(prop_cache)
    main_sum = mainsum(prop_cache)
    return [(main_sum, _ensurecapacity!(first(_ensurecount!(workspace.source_labels, 1)), length(main_sum)))]
end

_batcharrays!(::DictStorage, prop_cache, workspace, n_zone_terms::Vector{Int}) =
    _zonebatcharrays!(DictStorage(), workspace, nothing, n_zone_terms)

function _passsinks!(::DictStorage, prop_cache, workspace, tasks, n_tasks::Int)
    main_sum = mainsum(prop_cache)
    empty!(main_sum)
    return [main_sum]
end

_collectpass!(::DictStorage, prop_cache, sinks, thread::Bool) = prop_cache


### Multi sums

# Every zone is a task that reads its own terms. The terms of a label are collected in the zone that it
# picks, in the auxiliary arrays of a zone of arrays and in the workspace otherwise, and what the tasks make goes
# to the zones that own it: into the main arrays of zones of arrays, and through the outboxes otherwise.
function _passsources(::MultiSumStorage, prop_cache, workspace, thread::Bool)
    _checkauxempty(prop_cache)
    zone_caches = zonecaches(prop_cache)
    labels = _ensurecount!(workspace.source_labels, length(zone_caches))
    return [(zonecache, _ensurecapacity!(labels[zone_id], length(zonecache)))
            for (zone_id, zonecache) in enumerate(zone_caches)]
end

_batcharrays!(::MultiSumStorage, prop_cache, workspace, n_zone_terms::Vector{Int}) =
    _zonebatcharrays!(zonestorage(prop_cache), workspace, zonecaches(prop_cache), n_zone_terms)

_passsinks!(::MultiSumStorage, prop_cache, workspace, tasks, n_zones::Int) =
    _zonesinks!(zonestorage(prop_cache), prop_cache, tasks)

# the batches hold all of the sum, so every zone ends up with what the pass made for it alone
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
# label was applied to by one task.
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

# Arrays for the batches that every zone collects, `n_zone_terms[zone_id]` terms with their coefficients and labels.
# A zone of arrays keeps them in the auxiliary arrays and `indices` of its cache, any other zone in the workspace.
function _zonebatcharrays!(::ArrayStorage, workspace, zone_caches, n_zone_terms::Vector{Int})
    for (zone_id, zonecache) in enumerate(zone_caches)
        _ensurecapacity!(zonecache, n_zone_terms[zone_id])
    end
    zone_terms = [terms(auxsum(zonecache)) for zonecache in zone_caches]
    zone_coeffs = [coefficients(auxsum(zonecache)) for zonecache in zone_caches]
    zone_labels = [indices(zonecache) for zonecache in zone_caches]
    return zone_terms, zone_coeffs, zone_labels
end

function _zonebatcharrays!(::StorageType, workspace, zone_caches, n_zone_terms::Vector{Int})
    zone_arrays = (workspace.zone_terms, workspace.zone_coeffs, workspace.zone_labels)
    for arrays in zone_arrays
        _ensurecount!(arrays, length(n_zone_terms))
        for zone_id in eachindex(n_zone_terms)
            _ensurecapacity!(arrays[zone_id], n_zone_terms[zone_id])
        end
    end
    return zone_arrays
end


### Batches

# A batch holds terms with their coefficients and labels, those whose labels have the same highest bits.

# the terms of a batch, with their coefficients and labels, take about this many bytes
const _BATCH_BYTES = 1 << 18

const _MAX_BATCH_BITS = 12

# What the terms of a batch make is held next to them while its labels are applied to, so a term counts this many
# times its size when the batches are sized.
const _TERM_BYTES_FACTOR = 4

# The number of bits of the hash that pick the batch of a term, so that a batch is small enough to be worked
# on within the cache.
function _batchbits(n_terms::Int, term_bytes::Int)
    n_batches = cld(n_terms * _TERM_BYTES_FACTOR * term_bytes, _BATCH_BYTES)
    if n_batches <= 1
        return 0
    end
    return min(_MAX_BATCH_BITS, 8 * sizeof(Int) - leading_zeros(n_batches - 1))
end

# The zone that collects a term, and its batch among those of all zones: the highest `n_bits` bits of its label pick
# the batch, and the highest `n_zone_bits` of those the zone.
@inline function _zonebatchof(label::Int, n_bits::Int, n_zone_bits::Int)
    batch = Int((label % UInt64) >> (64 - n_bits))
    return (batch >> (n_bits - n_zone_bits)) + 1, batch + 1
end

"""
    _countbatchsizes!(labelof, batch_counts, n_bits, n_zone_bits, source, labels)

Counts the terms of `source` that go to each batch, and keeps their labels `labelof(term)` in `labels`, in the order
of `source`.
"""
function _countbatchsizes!(labelof::L, batch_counts, n_bits::Int, n_zone_bits::Int, source, labels::AbstractVector{Int}) where {L}
    source_index = 0
    for (term, _) in source
        label = @inline labelof(term)
        source_index += 1
        labels[source_index] = label
        _, batch = _zonebatchof(label, n_bits, n_zone_bits)
        batch_counts[batch] += 1
    end
    return batch_counts
end

"""
    _batchstarts!(source_batch_indices)

Turns the number of terms of every batch and source into the index at which the source copies its first term of the
batch, and returns where every batch starts.
The terms of a batch lie one source after the other, from `batch_starts[p]` to `batch_starts[p+1] - 1`.
"""
function _batchstarts!(source_batch_indices::Matrix{Int})
    n_batches, n_sources = size(source_batch_indices)
    batch_starts = Vector{Int}(undef, n_batches + 1)

    next_start = 1
    for batch in 1:n_batches
        batch_starts[batch] = next_start
        for source_id in 1:n_sources
            n_here = source_batch_indices[batch, source_id]
            source_batch_indices[batch, source_id] = next_start
            next_start += n_here
        end
    end
    batch_starts[n_batches+1] = next_start

    return batch_starts
end

"""
    _copytobatches!(zone_terms, zone_coeffs, zone_labels, zone_starts, next_indices, n_bits, n_zone_bits, source, labels)

Copies every term of `source` with its coefficient and its label from `labels` to `next_indices[batch]`, the index
of the next term that the source copies into its batch, which then moves on by one, in the arrays of the zone that the
batch belongs to.
"""
function _copytobatches!(zone_terms::Vector{Vector{TT}}, zone_coeffs::Vector{Vector{CT}}, zone_labels::Vector{Vector{Int}},
    zone_starts::Vector{Int}, next_indices, n_bits::Int, n_zone_bits::Int, source, labels::AbstractVector{Int}) where {TT,CT}

    source_index = 0
    for (term, coeff) in source
        source_index += 1
        label = labels[source_index]
        zone_id, batch = _zonebatchof(label, n_bits, n_zone_bits)

        index = next_indices[batch] - zone_starts[zone_id] + 1
        zone_terms[zone_id][index] = term
        zone_coeffs[zone_id][index] = coeff
        zone_labels[zone_id][index] = label
        next_indices[batch] += 1
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


### Workspaces

# The scratch memory of a pass: what every task works in, and the workspace of a pass, which the layers of a propagation
# reuse.

struct TaskWorkspace{TT,CT,KS}
    # the terms of one batch sorted by their label: the labels found through `slots`, and the terms and
    # coefficients one label after the other
    slots::Vector{Int32}
    label_of::Vector{Int32}
    label_values::Vector{UInt64}
    label_starts::Vector{Int}
    label_terms::Vector{TT}
    label_coeffs::Vector{CT}

    # what the function applied to the labels works in
    scratch::KS

    # the buffers of the task's sinks: one for an array sum that several tasks write to, one per zone for a multi sum
    sink_terms::Vector{Vector{TT}}
    sink_coeffs::Vector{Vector{CT}}
end

TaskWorkspace{TT,CT,KS}() where {TT,CT,KS} =
    TaskWorkspace{TT,CT,KS}(Int32[], Int32[], UInt64[], Int[], TT[], CT[], KS(), Vector{TT}[], Vector{CT}[])

"""
    LayerWorkspace(TT, CT, KS)

The scratch memory of a pass over terms of the type `TT` with coefficients of the type `CT`, whose tasks each apply a
function to the labels in a scratch of the type `KS`, which the layers of a propagation reuse.
"""
mutable struct LayerWorkspace{TT,CT,KS}
    tasks::Vector{TaskWorkspace{TT,CT,KS}}

    # the labels of the terms that every source holds, found when its terms are counted
    source_labels::Vector{Vector{Int}}

    # the batches of every zone that has no arrays to keep them in, those of a sum without arrays in the first
    zone_terms::Vector{Vector{TT}}
    zone_coeffs::Vector{Vector{CT}}
    zone_labels::Vector{Vector{Int}}
end

LayerWorkspace(::Type{TT}, ::Type{CT}, ::Type{KS}) where {TT,CT,KS} =
    LayerWorkspace{TT,CT,KS}(TaskWorkspace{TT,CT,KS}[], Vector{Int}[], Vector{TT}[], Vector{CT}[], Vector{Int}[])

# Workspaces that no layer is using. A layer takes one out and puts it back when it is done, so that the next layer
# uses the same memory, and propagations that run at the same time each have their own.
const _IDLE_WORKSPACES = IdDict{DataType,Vector{Any}}()
const _IDLE_WORKSPACES_LOCK = ReentrantLock()

function _takeworkspace(::Type{TT}, ::Type{CT}, ::Type{KS}) where {TT,CT,KS}
    lock(_IDLE_WORKSPACES_LOCK)
    try
        idle_workspaces = get(_IDLE_WORKSPACES, LayerWorkspace{TT,CT,KS}, nothing)
        if !isnothing(idle_workspaces) && !isempty(idle_workspaces)
            return pop!(idle_workspaces)::LayerWorkspace{TT,CT,KS}
        end
    finally
        unlock(_IDLE_WORKSPACES_LOCK)
    end
    return LayerWorkspace(TT, CT, KS)
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

# `vectors` with at least `n` elements, those added empty: the memory of `n` tasks, or vectors for `n` sources or zones
function _ensurecount!(vectors::Vector, n::Int)
    while length(vectors) < n
        push!(vectors, eltype(vectors)())
    end
    return vectors
end

# The buffer number `index` of the sinks of a task, `buffer_length` long. A buffer that grew when the arrays it writes to
# were full is replaced, so that it does not keep its memory.
function _sinkbuffer!(task::TaskWorkspace{TT,CT}, index::Int, buffer_length::Int) where {TT,CT}
    _ensurecount!(task.sink_terms, index)
    _ensurecount!(task.sink_coeffs, index)
    if length(task.sink_terms[index]) != buffer_length || length(task.sink_coeffs[index]) != buffer_length
        task.sink_terms[index] = Vector{TT}(undef, buffer_length)
        task.sink_coeffs[index] = Vector{CT}(undef, buffer_length)
    end
    return task.sink_terms[index], task.sink_coeffs[index]
end
