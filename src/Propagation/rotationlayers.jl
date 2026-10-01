###
##
# How a `RotationLayer` is propagated as a whole.
# The rotations of a sublayer commute, so the Pauli strings they turn into each other form orbits that never mix.
# Every Pauli string becomes a record, the records are partitioned by the hash of their orbit,
# and partition by partition the orbits are collected and the rotations applied to their coefficients.
##
###

# whether the sum of the cache is propagated orbit by orbit
_propagatesinorbits(prop_cache::AbstractPauliPropagationCache) = _propagatesinorbits(StorageType(prop_cache), prop_cache)
_propagatesinorbits(::PropagationBase.StorageType, prop_cache) = false
_propagatesinorbits(::PropagationBase.DictStorage, prop_cache) = coefftype(prop_cache) <: Number

function _propagatesinorbits(::PropagationBase.ArrayStorage, prop_cache)
    main_terms, main_coeffs, _, _ = PropagationBase._mainauxarrays(prop_cache)
    return coefftype(prop_cache) <: Number && main_terms isa Vector && main_coeffs isa Vector
end

function _propagatesinorbits(storage::PropagationBase.MultiSumStorage, prop_cache)
    return all(zonecache -> _propagatesinorbits(storage.zonestorage, zonecache), zonecaches(prop_cache))
end

"""
    _applyinorbits!(applyrotation!, layer::RotationLayer, prop_cache, theta, truncation; thread=true)

Applies the sublayers of the layer one after the other, orbit by orbit.
The Pauli strings of orbits with more rotations than a block holds meet the rotations one by one instead, in a cache of their own,
each rotation applied by `applyrotation!(cache, rotation, angle)`.
"""
function _applyinorbits!(applyrotation!::F, layer::RotationLayer, prop_cache::AbstractPauliPropagationCache, theta, truncation;
    thread::Bool=true) where {F}

    workspace = _takeworkspace(paulitype(prop_cache), coefftype(prop_cache))
    try
        for sublayer in layer.sublayers
            plan = _preparesublayer(layer, sublayer, theta, paulitype(prop_cache), nqubits(prop_cache))
            if StorageType(prop_cache) isa PropagationBase.MultiSumStorage
                # the zones rotate the orbits of more rotations than a block holds class by class
                long_plan = _prepareclasses(layer, theta, paulitype(prop_cache), nqubits(prop_cache), sublayer)
                _applysublayer!(StorageType(prop_cache), prop_cache, plan, truncation, workspace, long_plan; thread)
            else
                rotateonebyone!(cache) = _applyrotations!(applyrotation!, cache, layer, sublayer, theta)
                _applysublayer!(StorageType(prop_cache), prop_cache, plan, truncation, workspace, rotateonebyone!; thread)
            end
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

# The Pauli strings of the records `range`, whose orbits have more rotations than a block holds, after the rotations of the sublayer
# one by one. They are propagated as a `VectorPauliSum` whatever the sum of the layer, since a rotation costs less on arrays than on a dictionary.
function _rotatedonebyone(rotateonebyone!::F, n_qubits::Int, record_terms, record_coeffs, range) where {F}
    onebyone_sum = VectorPauliSum(n_qubits, record_terms[range], record_coeffs[range])
    if isempty(range)
        return onebyone_sum
    end
    return extractsum!(rotateonebyone!(PropagationCache(onebyone_sum)))
end

# the truncation within an orbit, where the weight of every entry is known
function _layertruncation(truncfunc, max_weight::Real)
    if isinf(max_weight)
        return (; truncfunc, max_weight=_UNLIMITED_WEIGHT)
    else
        return (; truncfunc, max_weight=floor(Int, max_weight))
    end
end

_nevertruncate(pstr, coeff) = false


### Arrays

# The records are written into the auxiliary arrays and sorted into the main arrays, whose Pauli strings they replace.
# What the sublayer makes is written into the auxiliary arrays, which become the sum.
function _applysublayer!(::PropagationBase.ArrayStorage, prop_cache::AbstractPauliPropagationCache, plan, truncation, workspace,
    rotateonebyone!::F; thread::Bool=true) where {F}

    task_partitioner, n_tasks = PropagationBase._preparetasks(activesize(prop_cache), thread)
    return _applysublayerintasks!(prop_cache, plan, truncation, workspace, rotateonebyone!, task_partitioner, n_tasks)
end

function _applysublayerintasks!(prop_cache::AbstractPauliPropagationCache, plan, truncation, workspace, rotateonebyone!::F,
    task_partitioner, n_tasks::Int) where {F}

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
    n_bits = _partitionbits(n_terms, _recordbytes(plan, sizeof(eltype(main_terms)) + sizeof(eltype(main_coeffs)) + sizeof(Int)))
    partition_counts = zeros(Int, (1 << n_bits) + 1, n_tasks)

    # a record for every Pauli string, at the index of the Pauli string
    function write_chunk!(task_id)
        chunk = task_partitioner[task_id]
        source = zip(view(main_terms, chunk), view(main_coeffs, chunk))
        _writerecords!(aux_terms, aux_coeffs, record_labels, view(partition_counts, :, task_id), n_bits,
            chunk.start, tasks[task_id].orbit_rotations, plan, source)
    end
    PropagationBase._eachtask(write_chunk!, n_tasks)

    partition_starts = _partitionstarts!(partition_counts)
    function sort_chunk!(task_id)
        chunk = task_partitioner[task_id]
        _sortrecords!(main_terms, main_coeffs, sorted_labels, view(partition_counts, :, task_id), n_bits,
            aux_terms, aux_coeffs, record_labels, chunk.start, chunk.stop)
    end
    PropagationBase._eachtask(sort_chunk!, n_tasks)

    # the last partition holds the Pauli strings that meet the rotations one by one, the others are transformed orbit by orbit
    onebyone_sum = _rotatedonebyone(rotateonebyone!, nqubits(prop_cache), main_terms, main_coeffs, partition_starts[end-1]:partition_starts[end]-1)
    orbit_partition_starts = view(partition_starts, 1:length(partition_starts)-1)

    if n_tasks == 1
        n_written = _transformpartitions!(prop_cache, plan, truncation, tasks[1], sorted_labels, orbit_partition_starts)
    else
        n_written = _transformpartitionsintasks!(prop_cache, plan, truncation, workspace, sorted_labels, orbit_partition_starts, n_tasks)
    end

    PropagationBase._commitwrite!(prop_cache, n_written, 0)
    add!(prop_cache, onebyone_sum)
    return prop_cache
end

# One task takes the partitions in turn and writes into the auxiliary arrays as it goes. Returns the number of Pauli strings written.
function _transformpartitions!(prop_cache::AbstractPauliPropagationCache, plan, truncation, task, sorted_labels::Vector{Int},
    partition_starts)

    output = ArrayOutput(prop_cache, 0)
    for partition in 1:length(partition_starts)-1
        # the arrays of the cache are looked up for every partition, since they are replaced when they grow
        sorted_terms, sorted_coeffs, _, _ = PropagationBase._mainauxarrays(prop_cache)
        _transformpartition!(output, task, plan, truncation, sorted_terms, sorted_coeffs, sorted_labels,
            partition_starts[partition], partition_starts[partition+1] - 1)
    end
    return output.n_written
end

# Several tasks share out the partitions and write into ranges of the auxiliary arrays that they reserve.
# What does not fit the arrays is left in the buffers of the tasks and written once the arrays have grown.
function _transformpartitionsintasks!(prop_cache::AbstractPauliPropagationCache, plan, truncation, workspace,
    sorted_labels::Vector{Int}, partition_starts, n_tasks::Int)

    sorted_terms, sorted_coeffs, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(prop_cache)
    tasks = workspace.tasks
    outputs = [task.output for task in tasks]
    n_reserved = Threads.Atomic{Int}(0)
    for task_id in 1:n_tasks
        _openoutput!(outputs[task_id], aux_terms, aux_coeffs, n_reserved)
    end

    n_partitions = length(partition_starts) - 1
    function transform_partitions!(task_id)
        for partition in task_id:n_tasks:n_partitions
            _transformpartition!(outputs[task_id], tasks[task_id], plan, truncation, sorted_terms, sorted_coeffs, sorted_labels,
                partition_starts[partition], partition_starts[partition+1] - 1)
        end
        _flush!(outputs[task_id])
    end
    PropagationBase._eachtask(transform_partitions!, n_tasks)

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

function _transformpartition!(output, task, plan, truncation, record_terms, record_coeffs, record_labels, lo::Int, hi::Int)
    if lo <= hi
        if _rotatesclasses(plan)
            _rotatepartitionbyclass!(output, task, plan, truncation, record_terms, record_coeffs, record_labels, lo, hi)
        else
            _grouporbits!(task, output, truncation, record_terms, record_coeffs, record_labels, lo, hi)
            _transformorbits!(output, task, plan, truncation, record_terms)
        end
    end
    return output
end


### Any sum that is iterated and added to

# The records are kept in the workspace and sorted within their arrays. They hold all of the sum,
# so the sum is emptied and takes what the sublayer makes, instead of a second sum of its size.
function _applysublayer!(::PropagationBase.DictStorage, prop_cache::AbstractPauliPropagationCache, plan, truncation, workspace,
    rotateonebyone!::F; thread::Bool=true) where {F}

    PropagationBase._checkauxempty(prop_cache)
    main_sum = mainsum(prop_cache)
    n_terms = length(main_sum)

    record_terms = _ensurelength!(workspace.terms, n_terms)
    record_coeffs = _ensurelength!(workspace.coeffs, n_terms)
    record_labels = _ensurelength!(workspace.labels, n_terms)

    task = first(_taskworkspaces!(workspace, 1, length(plan.masks)))
    n_bits = _partitionbits(n_terms, _recordbytes(plan, sizeof(eltype(record_terms)) + sizeof(eltype(record_coeffs)) + sizeof(Int)))
    partition_counts = zeros(Int, (1 << n_bits) + 1, 1)

    _writerecords!(record_terms, record_coeffs, record_labels, view(partition_counts, :, 1), n_bits, 1, task.orbit_rotations, plan, main_sum)
    partition_starts = _partitionstarts!(partition_counts)
    _sortrecords!(record_terms, record_coeffs, record_labels, partition_starts, view(partition_counts, :, 1), n_bits)

    # the last partition holds the Pauli strings that meet the rotations one by one
    onebyone_sum = _rotatedonebyone(rotateonebyone!, nqubits(prop_cache), record_terms, record_coeffs, partition_starts[end-1]:partition_starts[end]-1)

    empty!(main_sum)
    for partition in 1:length(partition_starts)-2
        _transformpartition!(main_sum, task, plan, truncation, record_terms, record_coeffs, record_labels,
            partition_starts[partition], partition_starts[partition+1] - 1)
    end
    add!(main_sum, onebyone_sum)

    return prop_cache
end


### Multi sums

# The Pauli strings of an orbit belong to many zones, so the records of an orbit are collected in the zone that the hash of
# the representative picks. Every zone reads its Pauli strings twice: to count the records for every zone, and to write them
# there, each zone into a range of its own. It then applies the rotations to the orbits collected in it, and those of orbits
# with more rotations than a block holds class by class, as `long_plan` prepares them. A zone of arrays keeps its records in
# its auxiliary arrays, so that its main arrays are free, and every zone writes what it makes into the main arrays of the
# zones that own it. Any other zone puts what it makes in its outbox, and at last the zones are emptied and take what the
# outboxes hold for them.
function _applysublayer!(::PropagationBase.MultiSumStorage, prop_cache::AbstractPauliPropagationCache, plan, truncation, workspace,
    long_plan; thread::Bool=true)

    PropagationBase._checkauxempty(prop_cache)
    n_terms = length(prop_cache)
    if n_terms == 0
        return prop_cache
    end

    zone_caches = zonecaches(prop_cache)
    n_zones = nzones(prop_cache)
    tasks = _taskworkspaces!(workspace, n_zones, length(plan.masks))

    # the highest bits of the hash pick the zone, and the bits below them a partition within the zone
    zone_bits = trailing_zeros(n_zones)
    record_bytes = _recordbytes(plan, sizeof(paulitype(prop_cache)) + sizeof(coefftype(prop_cache)) + sizeof(Int))
    n_bits = zone_bits + _partitionbits(cld(n_terms, n_zones), record_bytes)
    n_partitions_per_zone = 2 << (n_bits - zone_bits)

    # one row more than there are partitions, so that the counts of a partition are not a power of two apart, which would
    # put them all into one set of the processor cache
    partition_counts = zeros(Int, n_zones * n_partitions_per_zone + 1, n_zones)

    # The labels that the records are counted by, read again when the records are written. A zone of arrays also keeps its
    # records in place of its Pauli strings, so that no Pauli string is read twice.
    cached_labels = _cachedlabels!(workspace, zone_caches)
    function count_zone!(zone_id)
        zonecache = zone_caches[zone_id]
        _countrecords!(view(partition_counts, :, zone_id), n_bits, zone_bits, tasks[zone_id].orbit_rotations, plan, zonecache,
            cached_labels[zone_id], _inplacerecords(StorageType(zonecache), zonecache))
    end
    PropagationBase._eachzone(count_zone!, prop_cache, thread)

    # the records of a zone are numbered from 1 on
    partition_starts = _partitionstarts!(partition_counts)
    zone_starts = [partition_starts[(zone_id-1)*n_partitions_per_zone+1] for zone_id in 1:n_zones+1]
    zone_terms, zone_coeffs, zone_labels = _zonerecords!(PropagationBase.zonestorage(prop_cache), workspace, zone_caches, diff(zone_starts))

    function write_zone!(zone_id)
        _writezonerecords!(zone_terms, zone_coeffs, zone_labels, zone_starts, view(partition_counts, :, zone_id), n_bits, zone_bits,
            tasks[zone_id].orbit_rotations, plan, zone_caches[zone_id], cached_labels[zone_id])
    end
    PropagationBase._eachzone(write_zone!, prop_cache, thread)

    # the first half of the partitions of a zone holds orbits that fit a block, the second half the others
    zone_storage = PropagationBase.zonestorage(prop_cache)
    outputs = _zoneoutputs!(zone_storage, prop_cache, tasks)
    function transform_zone!(zone_id)
        first_record = zone_starts[zone_id]
        first_partition = (zone_id - 1) * n_partitions_per_zone
        for partition in first_partition+1:first_partition+n_partitions_per_zone
            lo = partition_starts[partition] - first_record + 1
            hi = partition_starts[partition+1] - first_record
            if partition - first_partition <= n_partitions_per_zone ÷ 2
                _transformpartition!(outputs[zone_id], tasks[zone_id], plan, truncation, zone_terms[zone_id], zone_coeffs[zone_id], zone_labels[zone_id],
                    lo, hi)
            elseif lo <= hi
                _rotatepartitionbyclass!(outputs[zone_id], tasks[zone_id], long_plan, truncation, zone_terms[zone_id], zone_coeffs[zone_id],
                    zone_labels[zone_id], lo, hi)
            end
        end
        _finishoutput!(outputs[zone_id])
    end
    PropagationBase._eachzone(transform_zone!, prop_cache, thread)

    # the records hold all of the sum, so every zone ends up with what the sublayer made for it alone
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
# different, since every orbit and class was transformed in one zone.
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

# The partition of a record is read from the hash of its representative, so that all records of an orbit lie in one partition,
# and a partition is small enough to be worked on within the cache.
function _partitionbits(n_records::Int, record_bytes::Int)
    n_partitions = cld(n_records * record_bytes, _PARTITION_BYTES)
    if n_partitions <= 1
        return 0
    end
    return min(_MAX_PARTITION_BITS, 8 * sizeof(Int) - leading_zeros(n_partitions - 1))
end

# The partition of a record: the highest bits of the hash of its orbit, where the table of a partition reads the lowest,
# and the partition after all of these for an orbit with more rotations than a block holds.
@inline function _partitionof(label::Int, n_bits::Int)
    if _norbitrotations(label) > _MAX_BLOCK_ROTATIONS
        return (1 << n_bits) + 1
    else
        return Int(_labelhash(label) >> (_HASH_BITS - n_bits)) + 1
    end
end

# The zone that collects a record of a multi sum, and the partition of the record among those of all zones. The highest
# `n_zone_bits` bits of the hash pick the zone, and the bits below them, up to `n_bits`, a partition of the zone. Every zone
# has as many partitions for orbits that fit a block as for those that do not, the latter after the former.
@inline function _zonepartitionof(label::Int, n_bits::Int, n_zone_bits::Int)
    hashbits = _labelhash(label)
    zone_index = Int(hashbits >> (_HASH_BITS - n_zone_bits))
    n_block_partitions = 1 << (n_bits - n_zone_bits)
    partition_in_zone = Int(hashbits >> (_HASH_BITS - n_bits)) & (n_block_partitions - 1)
    if _norbitrotations(label) > _MAX_BLOCK_ROTATIONS
        partition_in_zone += n_block_partitions
    end
    return zone_index + 1, 2 * n_block_partitions * zone_index + partition_in_zone + 1
end

"""
    _writerecords!(record_terms, record_coeffs, record_labels, partition_counts, n_bits, first_index, orbit_rotations, plan, source)

Writes a record for every Pauli string of `source` and its coefficient, from `first_index` on, and counts the records of each partition.
The record of a Pauli string whose orbit has more rotations than a block holds keeps the Pauli string itself.
`orbit_rotations` is scratch memory.
"""
function _writerecords!(record_terms::Vector{TT}, record_coeffs::Vector{CT}, record_labels::Vector{Int}, partition_counts, n_bits::Int,
    first_index::Int, orbit_rotations::Vector{Int32}, plan, source) where {TT,CT}

    index = first_index
    for (pstr, coeff) in source
        record_term, label = _record(orbit_rotations, plan, pstr)
        record_terms[index] = record_term
        record_coeffs[index] = coeff
        record_labels[index] = label
        partition_counts[_partitionof(label, n_bits)] += 1
        index += 1
    end

    return record_terms
end

# the Pauli string that the record of `pstr` keeps, and its label
@inline function _record(orbit_rotations::Vector{Int32}, plan, pstr)
    if _rotatesclasses(plan)
        return pstr, _classlabel(plan, pstr)
    end
    representative, index_in_orbit, n_orbit_rotations = _locateinorbit!(orbit_rotations, plan, pstr)
    return _recordterm(pstr, representative, n_orbit_rotations), _label(representative, index_in_orbit, n_orbit_rotations)
end

# the Pauli string that a record keeps
@inline function _recordterm(pstr, representative, n_orbit_rotations::Int)
    if n_orbit_rotations > _MAX_BLOCK_ROTATIONS
        return pstr
    else
        return representative
    end
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
    _countrecords!(partition_counts, n_bits, n_zone_bits, orbit_rotations, plan, source, cached_labels, record_terms)

Counts the records of each partition of the zones that the Pauli strings of `source` make.
The labels are kept in `cached_labels`, in the order of `source`, and the Pauli strings that the records keep in
`record_terms` unless it is `nothing`.
"""
function _countrecords!(partition_counts, n_bits::Int, n_zone_bits::Int, orbit_rotations::Vector{Int32}, plan, source,
    cached_labels::Vector{Int}, record_terms)

    source_index = 0
    for (pstr, _) in source
        record_term, label = _record(orbit_rotations, plan, pstr)
        source_index += 1
        cached_labels[source_index] = label
        if !isnothing(record_terms) && !_rotatesclasses(plan)
            record_terms[source_index] = record_term
        end
        _, partition = _zonepartitionof(label, n_bits, n_zone_bits)
        partition_counts[partition] += 1
    end
    return partition_counts
end

# the main arrays of a zone of arrays, where its records take the place of its Pauli strings
_inplacerecords(::PropagationBase.ArrayStorage, zonecache) = first(PropagationBase._mainauxarrays(zonecache))
_inplacerecords(::PropagationBase.StorageType, zonecache) = nothing

# room for the label of every Pauli string of every zone
function _cachedlabels!(workspace, zone_caches)
    cached_labels = workspace.cached_labels
    while length(cached_labels) < length(zone_caches)
        push!(cached_labels, Int[])
    end
    for (zone_id, zonecache) in enumerate(zone_caches)
        _ensurelength!(cached_labels[zone_id], length(zonecache))
    end
    return cached_labels
end

"""
    _writezonerecords!(zone_terms, zone_coeffs, zone_labels, zone_starts, cursors, n_bits, n_zone_bits, orbit_rotations, plan, zonecache,
        cached_labels)

Writes a record for every Pauli string of `zonecache` and its coefficient to where `cursors` points for its partition,
in the arrays of the zone that the partition belongs to. The labels are read from `cached_labels`, and so are the Pauli strings of
the records where the zone holds them, as a zone of arrays does and as every zone does for a class plan. Otherwise they are found again.
"""
function _writezonerecords!(zone_terms::Vector{Vector{TT}}, zone_coeffs::Vector{Vector{CT}}, zone_labels::Vector{Vector{Int}},
    zone_starts::Vector{Int}, cursors, n_bits::Int, n_zone_bits::Int, orbit_rotations::Vector{Int32}, plan, zonecache,
    cached_labels::Vector{Int}) where {TT,CT}

    # decided by the types, so that the loop holds only one of the two ways of reading a record
    holds_records = _rotatesclasses(plan) || !isnothing(_inplacerecords(StorageType(zonecache), zonecache))
    source_index = 0
    for (pstr, coeff) in zonecache
        source_index += 1
        record_term, label = if holds_records
            pstr, cached_labels[source_index]
        else
            _record(orbit_rotations, plan, pstr)
        end
        zone_id, partition = _zonepartitionof(label, n_bits, n_zone_bits)

        index = cursors[partition] - zone_starts[zone_id] + 1
        zone_terms[zone_id][index] = record_term
        zone_coeffs[zone_id][index] = coeff
        zone_labels[zone_id][index] = label
        cursors[partition] += 1
    end

    return zone_terms
end


### Labels

# an orbit of more rotations than this is not transformed as a block
const _MAX_BLOCK_ROTATIONS = 8

# the number of rotations of an orbit takes the lowest bits of a label, and any number above the most a block holds counts as one more
const _ROTATION_COUNT_BITS = 4

# a label holds a hash of the representative from this bit on
const _HASH_SHIFT = _ROTATION_COUNT_BITS + _MAX_BLOCK_ROTATIONS
const _HASH_BITS = 8 * sizeof(Int) - _HASH_SHIFT

# the records of a partition take about this many bytes
const _PARTITION_BYTES = 1 << 18

const _MAX_PARTITION_BITS = 12

"""
    _label(representative, index_in_orbit, n_orbit_rotations)

The label of a record: the number of rotations of its orbit, above it the index of the Pauli string in the orbit,
and above that a hash of the representative.
An orbit of more rotations than a block holds is labelled by the hash and by one more than the most a block holds,
an odd number, so that its records can be rotated as a class.
"""
@inline function _label(representative, index_in_orbit::UInt64, n_orbit_rotations::Int)
    if n_orbit_rotations > _MAX_BLOCK_ROTATIONS
        return ((_hashbits(representative) << _HASH_SHIFT) | UInt64(_MAX_BLOCK_ROTATIONS + 1)) % Int
    else
        return ((_hashbits(representative) << _HASH_SHIFT) | (index_in_orbit << _ROTATION_COUNT_BITS) | (n_orbit_rotations % UInt64)) % Int
    end
end

@inline _norbitrotations(label::Int) = Int((label % UInt64) & ((one(UInt64) << _ROTATION_COUNT_BITS) - 1))
@inline _indexinorbit(label::Int) = ((label % UInt64) >> _ROTATION_COUNT_BITS) & ((one(UInt64) << _MAX_BLOCK_ROTATIONS) - 1)
@inline _labelhash(label::Int) = (label % UInt64) >> _HASH_SHIFT

# A hash of `_HASH_BITS` bits. The limbs are multiplied independently of each other, so that the hash of a wide Pauli string is no chain over its limbs.
@inline function _hashbits(representative)
    limbs = _limbs(representative)
    folded = zero(UInt64)
    for i in eachindex(limbs)
        folded ⊻= limbs[i] * _foldfactor(i)
    end
    return PropagationBase._mix64(folded) >> _HASH_SHIFT
end

# an odd factor for every limb
const _FOLD_FACTORS = ntuple(i -> PropagationBase._mix64(UInt64(i)) | one(UInt64), 64)

@inline _foldfactor(i::Int) = _FOLD_FACTORS[((i-1)&63)+1] + ((UInt64(i - 1) >> 6) << 1)


### Grouping the records of a partition by orbit

"""
    _grouporbits!(task, output, truncation, record_terms, record_coeffs, record_labels, lo, hi)

Finds the orbits of the records `lo` to `hi` and adds the coefficient of every record to the block of its orbit, at the index of its Pauli string in the orbit.
A Pauli string that anticommutes with no rotation goes to `output` as it is.
"""
function _grouporbits!(task, output, truncation, record_terms::Vector{TT}, record_coeffs::Vector{CT},
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

    for i in lo:hi
        label = @inbounds record_labels[i]
        representative = @inbounds record_terms[i]
        n_orbit_rotations = _norbitrotations(label)

        if n_orbit_rotations == 0
            coeff = @inbounds record_coeffs[i]
            if !_istruncated(truncation, representative, coeff)
                _emit!(output, representative, coeff)
            end
            continue
        end

        # a slot is within the table, and an orbit in a slot is one of the orbits found so far
        hashbits = _labelhash(label)
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

        if is_new
            block = n_blocks[n_orbit_rotations] + 1
            n_blocks[n_orbit_rotations] = block
            orbit_blocks[orbit] = block
            _newblock!(task, n_orbit_rotations, block, i)
        end

        block_coeffs = task.block_coeffs[n_orbit_rotations]
        block_present = task.block_present[n_orbit_rotations]
        entry = (((@inbounds orbit_blocks[orbit]) - 1) << n_orbit_rotations) + Int(_indexinorbit(label)) + 1
        if block_present[entry]
            block_coeffs[entry] = mergefunc(block_coeffs[entry], record_coeffs[i])
        else
            block_coeffs[entry] = record_coeffs[i]
            block_present[entry] = true
        end
    end

    return task
end

# an empty block for an orbit of `n_orbit_rotations` rotations whose first record is `first_record`
function _newblock!(task, n_orbit_rotations::Int, block::Int, first_record::Int)
    block_records = task.block_records[n_orbit_rotations]
    block_coeffs = task.block_coeffs[n_orbit_rotations]
    block_present = task.block_present[n_orbit_rotations]

    n_entries = block << n_orbit_rotations
    if n_entries > length(block_coeffs) || n_entries > length(block_present) || block > length(block_records)
        _ensurelength!(block_records, block)
        _ensurelength!(block_coeffs, n_entries)
        _ensurelength!(block_present, n_entries)
    end

    block_records[block] = first_record
    for entry in n_entries-(1<<n_orbit_rotations)+1:n_entries
        block_coeffs[entry] = zero(eltype(block_coeffs))
        block_present[entry] = false
    end
    return task
end


### The rotations of a sublayer on one orbit

# The block of an orbit holds the coefficient of every Pauli string of the orbit at the index of the Pauli string in the orbit.
# The rotation of bit b mixes the coefficients of every two entries whose indices differ in bit b alone,
# the lower entry without the bit and the upper entry with it.
# The truncations are applied after every rotation, as they are between the gates of a circuit.

# no Pauli string is heavier than this
const _UNLIMITED_WEIGHT = typemax(Int)

@inline _limitsweight(truncation) = truncation.max_weight != _UNLIMITED_WEIGHT

@inline function _istruncated(truncation, pstr, coeff)
    if _limitsweight(truncation) && countweight(pstr) > truncation.max_weight
        return true
    end
    return @inline truncation.truncfunc(pstr, coeff)
end

@inline function _istruncated(truncation, pstr, weight::Int, coeff)
    return weight > truncation.max_weight || @inline truncation.truncfunc(pstr, coeff)
end

# Without a limit on the weight, the weights are not counted.
@inline function _weightiflimited(truncation, pstr)
    if _limitsweight(truncation)
        return countweight(pstr)
    else
        return 0
    end
end

const _NO_WEIGHT_CHANGES = ntuple(_ -> Int8(0), 16)

@inline function _weightchangesiflimited(truncation, plan)
    if _limitsweight(truncation)
        return plan.weight_changes
    else
        return _NO_WEIGHT_CHANGES
    end
end

# Applies the rotations of the sublayer to the orbits that `_grouporbits!` found, and writes their Pauli strings to `output`.
# The orbits of every number of rotations are transformed by a method of their own, in which that number is known to the compiler.
@eval function _transformorbits!(output, task, plan, truncation, record_terms)
    Base.Cartesian.@nexprs $_MAX_BLOCK_ROTATIONS n_orbit_rotations -> begin
        if task.n_blocks[n_orbit_rotations] > 0
            _transformblocks!(output, task, plan, truncation, record_terms, Val(n_orbit_rotations))
        end
    end
    return output
end

function _transformblocks!(output, task, plan, truncation, record_terms::Vector{TT}, ::Val{K}) where {TT,K}
    n_blocks = task.n_blocks[K]
    block_records = task.block_records[K]
    block_coeffs = task.block_coeffs[K]
    block_present = task.block_present[K]
    _checkblocks(task, n_blocks, block_records, block_coeffs, block_present, K)

    if plan.overlapping
        for block in 1:n_blocks
            representative = record_terms[block_records[block]]
            _transformblock!(output, task, plan, truncation, representative, block_coeffs, block_present, (block - 1) << K, Val(K), Val(true))
        end
    else
        for block in 1:n_blocks
            representative = record_terms[block_records[block]]
            _transformblock!(output, task, plan, truncation, representative, block_coeffs, block_present, (block - 1) << K, Val(K), Val(false))
        end
    end
    return output
end

# The block of an orbit.
# Where no two rotations of the sublayer share a qubit, a rotation finds the same Paulis in every entry, so its signs and its weight change
# are those of the representative. Otherwise the other rotations change the Paulis it finds, and they are read from every entry.
@inline function _transformblock!(output, task, plan, truncation, representative, block_coeffs::Vector{CT}, block_present::Vector{Bool},
    block_start::Int, ::Val{K}, ::Val{SharesQubits}) where {CT,K,SharesQubits}

    orbit_rotations = task.orbit_rotations
    local_paulis = task.local_paulis
    orbit_terms = task.orbit_terms
    orbit_weights = task.orbit_weights

    if _orbitrotations!(orbit_rotations, plan, representative) != K
        _throwwrongblock()
    end
    n_entries = 1 << K

    # the rotation of every bit doubles the Pauli strings of the orbit
    weight_changes = _weightchangesiflimited(truncation, plan)
    orbit_terms[1] = representative
    orbit_weights[1] = _weightiflimited(truncation, representative)

    for bit in 0:K-1
        rotation = orbit_rotations[bit+1]
        mask = plan.masks[rotation]
        n_lower = 1 << bit

        if SharesQubits
            @inbounds for entry in 1:n_lower
                orbit_terms[n_lower+entry] = orbit_terms[entry] ⊻ mask
                orbit_weights[n_lower+entry] = orbit_weights[entry]
            end
            if _limitsweight(truncation)
                @inbounds for entry in 1:n_lower
                    entry_paulis = _localpaulis(plan, orbit_terms[entry], rotation)
                    orbit_weights[n_lower+entry] += weight_changes[entry_paulis+1]
                end
            end
        else
            paulis = _localpaulis(plan, representative, rotation)
            local_paulis[bit+1] = paulis
            weight_change = Int(weight_changes[paulis+1])
            @inbounds for entry in 1:n_lower
                orbit_terms[n_lower+entry] = orbit_terms[entry] ⊻ mask
                orbit_weights[n_lower+entry] = orbit_weights[entry] + weight_change
            end
        end
    end

    for bit in _bitsinorder(K, plan.reversed)
        rotation = orbit_rotations[bit+1]
        cos_val = plan.cosines[rotation]
        sin_val = plan.sines[rotation]

        # Where no two rotations share a qubit, every lower entry has the Paulis of the representative on the qubits of the rotation,
        # so the rotation gives all lower entries one sign and all upper entries another.
        paulis = if SharesQubits
            0
        else
            local_paulis[bit+1]
        end
        representative_sign_from_lower = plan.signs[paulis+1]
        representative_sign_from_upper = plan.signs[(paulis⊻plan.local_mask)+1]

        n_lower = 1 << bit
        below_bit = n_lower - 1
        @inbounds for rank in 0:(n_entries>>1)-1
            # the entry of this rank among those without the bit, and the entry with the bit
            lower = (((rank & ~below_bit) << 1) | (rank & below_bit)) + 1
            upper = lower + n_lower

            if block_present[block_start+lower] || block_present[block_start+upper]
                sign_from_lower, sign_from_upper = if SharesQubits
                    _rotationsigns(plan, orbit_terms[lower], rotation)
                else
                    representative_sign_from_lower, representative_sign_from_upper
                end
                _rotateentries!(block_coeffs, block_present, block_start, lower, upper, cos_val, sin_val, sign_from_lower, sign_from_upper,
                    orbit_terms, orbit_weights, truncation)
            end
        end
    end

    _emitblock!(output, orbit_terms, block_coeffs, block_present, block_start, n_entries)
    return output
end

# The coefficients of the lower and the upper entry after the rotation, and whether the truncations keep them.
Base.@propagate_inbounds function _rotateentries!(block_coeffs::Vector{CT}, block_present, block_start::Int, lower::Int, upper::Int, cos_val, sin_val,
    sign_from_lower, sign_from_upper, orbit_terms, orbit_weights, truncation) where {CT}

    lower_coeff = block_coeffs[block_start+lower]
    upper_coeff = block_coeffs[block_start+upper]
    new_lower_coeff = mergefunc(lower_coeff * cos_val, upper_coeff * sin_val * sign_from_upper)
    new_upper_coeff = mergefunc(upper_coeff * cos_val, lower_coeff * sin_val * sign_from_lower)

    keep_lower = !_istruncated(truncation, orbit_terms[lower], orbit_weights[lower], new_lower_coeff)
    keep_upper = !_istruncated(truncation, orbit_terms[upper], orbit_weights[upper], new_upper_coeff)
    block_coeffs[block_start+lower] = ifelse(keep_lower, new_lower_coeff, zero(CT))
    block_coeffs[block_start+upper] = ifelse(keep_upper, new_upper_coeff, zero(CT))
    block_present[block_start+lower] = keep_lower
    block_present[block_start+upper] = keep_upper
    return
end

# the signs of the Pauli strings that `rotation` creates from `lower_pstr` and from its product with the generator
@inline function _rotationsigns(plan, lower_pstr, rotation::Integer)
    paulis = _localpaulis(plan, lower_pstr, rotation)
    return plan.signs[paulis+1], plan.signs[(paulis⊻plan.local_mask)+1]
end

# The bits of the rotations of an orbit in the order in which the rotations are applied.
# The bits follow the lowest qubits of the rotations, and so does the order of application, or its reverse in the Heisenberg picture.
@inline function _bitsinorder(n_orbit_rotations::Int, reversed::Bool)
    if reversed
        return n_orbit_rotations-1:-1:0
    else
        return 0:1:n_orbit_rotations-1
    end
end

# the loops over the entries of a block index the arrays of the task without bounds checks
function _checkblocks(task, n_blocks::Int, block_records, block_coeffs, block_present, n_orbit_rotations::Int)
    n_entries = 1 << n_orbit_rotations
    if !(1 <= n_orbit_rotations <= _MAX_BLOCK_ROTATIONS) || n_blocks > length(block_records) ||
       n_blocks * n_entries > min(length(block_coeffs), length(block_present)) ||
       n_entries > min(length(task.orbit_terms), length(task.orbit_weights)) ||
       n_orbit_rotations > min(length(task.orbit_rotations), length(task.local_paulis))
        throw(ArgumentError("the $n_blocks blocks of the orbits of $n_orbit_rotations rotations do not fit the workspace"))
    end
    return
end

@noinline _throwwrongblock() = throw(ArgumentError("the orbit of a block has another number of rotations than the block"))


### Where a sublayer writes the Pauli strings it makes

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

function TaskOutput(::Type{TT}, ::Type{CT}, flush_length::Int=_TASK_BUFFER_LENGTH) where {TT,CT}
    n_room = flush_length + (1 << _MAX_BLOCK_ROTATIONS)
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

# the entries of the block of an orbit that are present
function _emitblock!(output::ArrayOutputs, orbit_terms, block_coeffs, block_present, block_start::Int, n_entries::Int)
    output_terms, output_coeffs = _roomtoemit!(output, n_entries)
    n_written = output.n_written

    # every entry is written, and the next one writes over it if it is not present
    @inbounds for entry in 1:n_entries
        output_terms[n_written+1] = orbit_terms[entry]
        output_coeffs[n_written+1] = block_coeffs[block_start+entry]
        n_written += block_present[block_start+entry]
    end

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
    n_room = output.flush_length + (1 << _MAX_BLOCK_ROTATIONS)
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

function _emitblock!(output::ZoneOutputs, orbit_terms, block_coeffs, block_present, block_start::Int, n_entries::Int)
    for entry in 1:n_entries
        if block_present[block_start+entry]
            _emit!(output, orbit_terms[entry], block_coeffs[block_start+entry])
        end
    end
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

function _emitblock!(output::AbstractTermSum, orbit_terms, block_coeffs, block_present, block_start::Int, n_entries::Int)
    for entry in 1:n_entries
        if block_present[block_start+entry]
            push!(output, orbit_terms[entry], block_coeffs[block_start+entry])
        end
    end
    return
end


### The scratch memory of a layer

mutable struct TaskWorkspace{TT,CT}
    # the rotations that anticommute with the Pauli string at hand, and the Paulis on their qubits
    orbit_rotations::Vector{Int32}
    local_paulis::Vector{Int}

    # the orbits of one partition
    slots::Vector{Int32}
    orbit_hashes::Vector{UInt64}
    first_records::Vector{Int}
    orbit_blocks::Vector{Int}

    # A block of coefficients for every orbit. The blocks of the orbits of `k` rotations lie one after the other at index `k`,
    # so that every loop over them runs the same number of times.
    n_blocks::Vector{Int}
    block_records::Vector{Vector{Int}}
    block_coeffs::Vector{Vector{CT}}
    block_present::Vector{Vector{Bool}}

    # the Pauli strings of one orbit and their weights
    orbit_terms::Vector{TT}
    orbit_weights::Vector{Int}

    # the classes of one partition, found through `slots`, and the records of each
    class_of::Vector{Int32}
    class_keys::Vector{TT}
    class_hashes::Vector{UInt64}
    class_starts::Vector{Int}
    class_records::Vector{Int32}

    # The Pauli strings of one class, whether they are present, and the last rotation that mixed them, found through the
    # first `class_table_length` of `class_slots` by the highest bits of their hash from `class_hash_shift` on.
    entry_terms::Vector{TT}
    entry_coeffs::Vector{CT}
    entry_present::Vector{Bool}
    entry_steps::Vector{Int32}
    n_entries::Int
    class_slots::Vector{Int32}
    class_table_length::Int
    class_hash_shift::Int

    # where the task writes when several tasks write to one array sum, and to the zones of a multi sum
    output::TaskOutput{TT,CT}
    zone_outputs::Vector{TaskOutput{TT,CT}}
end

function TaskWorkspace(::Type{TT}, ::Type{CT}) where {TT,CT}
    n_entries = 1 << _MAX_BLOCK_ROTATIONS
    return TaskWorkspace{TT,CT}(Int32[], Int[], Int32[], UInt64[], Int[], Int[],
        zeros(Int, _MAX_BLOCK_ROTATIONS), [Int[] for _ in 1:_MAX_BLOCK_ROTATIONS],
        [CT[] for _ in 1:_MAX_BLOCK_ROTATIONS], [Bool[] for _ in 1:_MAX_BLOCK_ROTATIONS],
        Vector{TT}(undef, n_entries), Vector{Int}(undef, n_entries),
        Int32[], TT[], UInt64[], Int[], Int32[], TT[], CT[], Bool[], Int32[], 0, Int32[], 0, 64, TaskOutput(TT, CT), TaskOutput{TT,CT}[])
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

# the memory of `n_tasks` tasks for a sublayer of `n_rotations` rotations
function _taskworkspaces!(workspace::LayerWorkspace{TT,CT}, n_tasks::Int, n_rotations::Int) where {TT,CT}
    tasks = workspace.tasks
    while length(tasks) < n_tasks
        push!(tasks, TaskWorkspace(TT, CT))
    end

    for task_id in 1:n_tasks
        _ensurelength!(tasks[task_id].orbit_rotations, n_rotations)
        _ensurelength!(tasks[task_id].local_paulis, n_rotations)
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


### What a sublayer reads from a Pauli string

# The rotations that anticommute with a Pauli string span its orbit. The representative of the orbit is the Pauli string
# times the generators of some of those rotations, the selected ones, and the index of the Pauli string in the orbit has a bit
# set for each of them, the rotations taken in the order of their lowest qubits. A rotation is selected if its pivot bit,
# which lies on its lowest qubit, is set once the generators of the selected rotations below it have been multiplied out.

# a sublayer with more distances between the qubits of its rotations is read rotation by rotation
const _MAX_ROTATION_GROUPS = 4

"""
    _preparesublayer(layer::RotationLayer, sublayer, theta, TT, nqubits)

The rotations `sublayer` of the `layer` with the parameter `theta`, prepared for Pauli strings of the type `TT` on `nqubits` qubits.
The rotations are kept in the order in which they are applied.
"""
function _preparesublayer(layer::RotationLayer, sublayer::Vector{Int}, theta, ::Type{TT}, nqubits::Int) where {TT}
    symbols = layer.symbols
    n_rotations = length(sublayer)

    masks = Vector{TT}(undef, n_rotations)
    pivots = Vector{Int}(undef, n_rotations)
    qinds = Vector{NTuple{2,Int}}(undef, n_rotations)
    angles = [_rotationangle(theta, index) for index in sublayer]

    # per qubit: the rotation whose lower qubit it is and the upper qubit of that rotation, 0 where there is none,
    # and the rotations whose upper qubit it is
    rotation_at_lower = zeros(Int32, nqubits)
    upper_of_lower = zeros(Int32, nqubits)
    rotations_at_upper = [Int32[] for _ in 1:nqubits]
    rotations_on_qubit = [Int32[] for _ in 1:nqubits]

    # the low bit of every qubit a rotation acts on with its first and with its second symbol
    role_masks = [zero(TT), zero(TT)]

    for (rotation, index) in enumerate(sublayer)
        rotation_qinds = layer.qinds[index]
        _check_qind_range(nqubits, rotation_qinds)

        masks[rotation] = symboltoint(TT, symbols, rotation_qinds)
        qinds[rotation] = (rotation_qinds[1], get(rotation_qinds, 2, 0))

        lower_qind, lower_role = findmin(rotation_qinds)
        pivots[rotation] = 2 * (lower_qind - 1) + _pivotoffset(symbols[lower_role])
        if rotation_at_lower[lower_qind] != 0
            throw(ArgumentError("Two rotations of a sublayer have the lowest qubit $lower_qind."))
        end
        rotation_at_lower[lower_qind] = rotation

        if length(rotation_qinds) == 2
            upper_qind = maximum(rotation_qinds)
            upper_of_lower[lower_qind] = upper_qind
            push!(rotations_at_upper[upper_qind], rotation)
        end

        for (role, qind) in enumerate(rotation_qinds)
            push!(rotations_on_qubit[qind], rotation)
            role_masks[role] |= symboltoint(TT, :X, qind)
        end
    end

    symbol_codes = (UInt8(symboltoint(symbols[1])), UInt8(symboltoint(get(symbols, 2, :I))))
    local_mask = symbol_codes[1] | (symbol_codes[2] << 2)

    # empty if the rotations are read one by one
    groups = _rotationgroups(TT, symbols, qinds)
    if !_readsingroups(groups)
        empty!(groups)
    end

    # for each of the 16 combinations of Paulis on the qubits of a rotation
    signs = _localsigns(local_mask)
    weight_changes = _localweightchanges(local_mask)

    return (; masks, pivots, qinds, cosines=cos.(angles), sines=sin.(angles),
        rotation_at_lower, upper_of_lower, rotations_at_upper, role_masks=(role_masks[1], role_masks[2]), symbol_codes,
        local_mask, signs, weight_changes, groups,
        # whether the rotations are applied in the reverse order of their lowest qubits, as in the Heisenberg picture
        reversed=!issorted(pivots),
        # whether any two rotations share a qubit
        overlapping=any(rotations -> length(rotations) > 1, rotations_on_qubit))
end

# whether the rotations are read from the whole Pauli string at once
_readsingroups(groups) = length(groups) <= _MAX_ROTATION_GROUPS

# X and Z flip the low bit of their qubit, Y only the high bit
function _pivotoffset(symbol::Symbol)
    if symbol == :Y
        return 1
    else
        return 0
    end
end

# The rotations of a sublayer whose qubits are the same distance apart: the low bit of the lower qubit of every rotation,
# the distance from the lower to the upper qubit in bits (0 for rotations on one qubit), the offset of the pivot bit,
# and the Paulis of the generator on the lower and on the upper qubit.
function _rotationgroups(::Type{TT}, symbols::Vector{Symbol}, qinds::Vector{NTuple{2,Int}}) where {TT}
    lower_masks = Dict{Tuple{Int,Symbol,Symbol},TT}()

    for (first_qind, second_qind) in qinds
        if second_qind == 0
            group = (0, symbols[1], :I)
            lower_qind = first_qind
        elseif first_qind < second_qind
            group = (2 * (second_qind - first_qind), symbols[1], symbols[2])
            lower_qind = first_qind
        else
            group = (2 * (first_qind - second_qind), symbols[2], symbols[1])
            lower_qind = second_qind
        end
        lower_masks[group] = get(lower_masks, group, zero(TT)) | symboltoint(TT, :X, lower_qind)
    end

    return [(; lower_mask, shift, pivot_offset=_pivotoffset(lower_symbol), lower_pauli=UInt8(symboltoint(lower_symbol)),
        upper_pauli=UInt8(symboltoint(upper_symbol))) for ((shift, lower_symbol, upper_symbol), lower_mask) in lower_masks]
end

# the sign a rotation gives the Pauli string it creates from each combination of Paulis on its qubits, 0 where it commutes with them
function _localsigns(local_mask::UInt8)
    function sign_of(index)
        paulis = UInt8(index - 1)
        if commutes(local_mask, paulis)
            return Int8(0)
        end
        _, sign = paulirotationproduct(local_mask, paulis)
        return Int8(sign)
    end
    return ntuple(sign_of, 16)
end

# how the weight of each combination of Paulis on the qubits of a rotation changes under the product with the generator
function _localweightchanges(local_mask::UInt8)
    weight_change_of(index) = Int8(countweight(UInt8(index - 1) ⊻ local_mask) - countweight(UInt8(index - 1)))
    return ntuple(weight_change_of, 16)
end

# the Paulis of `pstr` on the qubits of `rotation`, the first qubit in the low bits
@inline function _localpaulis(plan, pstr, rotation::Integer)
    first_qind, second_qind = plan.qinds[rotation]
    paulis = Int(_getpaulibits(pstr, first_qind))
    if second_qind != 0
        paulis |= Int(_getpaulibits(pstr, second_qind)) << 2
    end
    return paulis
end

"""
    _locateinorbit!(orbit_rotations, plan, pstr)

Returns the representative of the orbit of `pstr`, the index of `pstr` in the orbit, and the number of rotations that span the orbit.
`orbit_rotations` is scratch memory.
"""
@inline function _locateinorbit!(orbit_rotations::Vector{Int32}, plan, pstr)
    if isempty(plan.groups)
        return _locatebyrotation!(orbit_rotations, plan, pstr)
    else
        return _locatebygroup(plan, pstr)
    end
end

"""
    _orbitrotations!(orbit_rotations, plan, representative)

Writes the rotations that span the orbit of `representative` into `orbit_rotations`, in the order of the bits of an index in the orbit, and returns their number.
"""
@inline function _orbitrotations!(orbit_rotations::Vector{Int32}, plan, representative)
    if isempty(plan.groups)
        return _anticommuting!(orbit_rotations, plan, representative)
    end

    anticommuting = _limbs(_anticommutingmask(plan, representative))
    n_found = 0
    for limb_index in eachindex(anticommuting)
        limb = anticommuting[limb_index]
        while limb != 0
            qind = 32 * (limb_index - 1) + (trailing_zeros(limb) >> 1) + 1
            limb &= limb - 1
            n_found += 1
            orbit_rotations[n_found] = plan.rotation_at_lower[qind]
        end
    end
    return n_found
end

# a Pauli string as 64-bit limbs, the lowest first
@inline _limbs(pstr::NTupleInteger) = pstr.limbs
@inline _limbs(pstr::UInt128) = (pstr % UInt64, (pstr >> 64) % UInt64)
@inline _limbs(pstr::Union{UInt8,UInt16,UInt32,UInt64}) = (pstr % UInt64,)

@inline _readbit(pstr, bit::Int) = isodd(PropagationBase._wordat(pstr, bit) >> (bit & 63))

# the low bit of every qubit on which `pstr` anticommutes with the Pauli `symbol_code`
@inline function _locallyanticommuting(symbol_code::UInt8, pstr::TT) where {TT}
    low_bits = alternatingmask(pstr)
    if symbol_code == 0x01
        # X anticommutes with Y and Z, which have the high bit set
        return _shiftdown(pstr, 1) & low_bits
    elseif symbol_code == 0x02
        # Y anticommutes with X and Z, which have the low bit set
        return pstr & low_bits
    else
        # Z anticommutes with X and Y, whose bits differ
        return (pstr ⊻ _shiftdown(pstr, 1)) & low_bits
    end
end

# the low bit of every qubit on which `pstr` anticommutes with the rotation acting there
@inline function _candidates(plan, pstr)
    candidates = _locallyanticommuting(plan.symbol_codes[1], pstr) & plan.role_masks[1]
    if plan.symbol_codes[2] != 0x00
        candidates |= _locallyanticommuting(plan.symbol_codes[2], pstr) & plan.role_masks[2]
    end
    return candidates
end

# Shifts of a whole Pauli string. Below 64 bits, a shift only moves bits between neighbouring limbs.
@inline _shiftdown(pstr, shift::Int) = pstr >> shift
@inline _shiftup(pstr, shift::Int) = pstr << shift

@inline function _shiftdown(pstr::NTupleInteger{N}, shift::Int) where {N}
    if !(0 <= shift < 64)
        return pstr >> shift
    end
    limbs = pstr.limbs
    shifted_limb(i) = (limbs[i] >> shift) | ifelse(i < N, limbs[min(i + 1, N)] << (64 - shift), zero(UInt64))
    return NTupleInteger{N}(ntuple(shifted_limb, Val(N)))
end

@inline function _shiftup(pstr::NTupleInteger{N}, shift::Int) where {N}
    if !(0 <= shift < 64)
        return pstr << shift
    end
    limbs = pstr.limbs
    shifted_limb(i) = (limbs[i] << shift) | ifelse(i > 1, limbs[max(i - 1, 1)] >> (64 - shift), zero(UInt64))
    return NTupleInteger{N}(ntuple(shifted_limb, Val(N)))
end


### Rotations read in groups, from the whole Pauli string at once

# the low bit of the lower qubit of every rotation of the group that anticommutes with `pstr`
@inline function _groupanticommuting(group, candidates)
    if group.shift == 0
        return candidates & group.lower_mask
    else
        # a rotation on two qubits anticommutes if the Pauli string anticommutes with it on exactly one of them
        return (candidates ⊻ _shiftdown(candidates, group.shift)) & group.lower_mask
    end
end

@inline function _anticommutingmask(plan, pstr::TT) where {TT}
    candidates = _candidates(plan, pstr)
    anticommuting = zero(TT)
    for group in plan.groups
        anticommuting |= _groupanticommuting(group, candidates)
    end
    return anticommuting
end

# the bits of the Pauli `pauli` on every qubit whose low bit is set in `low_bits`
@inline function _paulionqubits(pauli::UInt8, low_bits::TT) where {TT}
    if pauli == 0x01
        return low_bits
    elseif pauli == 0x02
        return _shiftup(low_bits, 1)
    elseif pauli == 0x03
        return low_bits | _shiftup(low_bits, 1)
    else
        return zero(TT)
    end
end

@inline function _locatebygroup(plan, pstr::TT) where {TT}
    candidates = _candidates(plan, pstr)
    anticommuting = zero(TT)
    pivot_bits = zero(TT)

    for group in plan.groups
        anticommuting_here = _groupanticommuting(group, candidates)
        anticommuting |= anticommuting_here
        pivot_bits |= _shiftdown(pstr, group.pivot_offset) & anticommuting_here
    end

    selected = pivot_bits
    if plan.overlapping
        selected = _selectinturn(plan, pivot_bits, anticommuting)
    end

    flipped = zero(TT)
    for group in plan.groups
        selected_here = selected & group.lower_mask
        flipped ⊻= _paulionqubits(group.lower_pauli, selected_here)
        if group.shift != 0
            flipped ⊻= _shiftup(_paulionqubits(group.upper_pauli, selected_here), group.shift)
        end
    end

    index_in_orbit, n_orbit_rotations = _compressbits(selected, anticommuting)
    return pstr ⊻ flipped, index_in_orbit, n_orbit_rotations
end

# The generator of a selected rotation flips the pivot bit of the rotation on its upper qubit.
# Every round selects the rotations one qubit further up a run of rotations that share qubits, until nothing changes.
@inline function _selectinturn(plan, pivot_bits::TT, anticommuting::TT) where {TT}
    selected = pivot_bits
    while true
        flipped_pivots = zero(TT)
        for group in plan.groups
            if group.shift != 0
                flipped_pivots ⊻= _shiftup(selected & group.lower_mask, group.shift)
            end
        end

        newly_selected = (pivot_bits ⊻ flipped_pivots) & anticommuting
        if newly_selected == selected
            return selected
        end
        selected = newly_selected
    end
end

# the bits of `bits` where `mask` is set, moved next to each other, and their number
@inline function _compressbits(bits::TT, mask::TT) where {TT}
    bit_limbs = _limbs(bits)
    mask_limbs = _limbs(mask)
    compressed = zero(UInt64)
    n_bits = 0

    for limb_index in eachindex(mask_limbs)
        mask_limb = mask_limbs[limb_index]
        bit_limb = bit_limbs[limb_index]
        while mask_limb != 0
            bit = trailing_zeros(mask_limb)
            compressed |= ((bit_limb >> (bit & 63)) & one(UInt64)) << (n_bits & 63)
            n_bits += 1
            mask_limb &= mask_limb - 1
        end
    end

    return compressed, n_bits
end


### Rotations read one by one, through the qubits on which the Pauli string anticommutes with them

"""
    _anticommuting!(orbit_rotations, plan, pstr)

Writes the rotations that anticommute with `pstr` into `orbit_rotations`, in the order of their lowest qubits, and returns their number.
A rotation on two qubits anticommutes if `pstr` anticommutes with it on exactly one of them.
"""
function _anticommuting!(orbit_rotations::Vector{Int32}, plan, pstr)
    candidates = _candidates(plan, pstr)
    candidate_limbs = _limbs(candidates)
    n_found = 0

    for limb_index in eachindex(candidate_limbs)
        limb = candidate_limbs[limb_index]
        while limb != 0
            qind = 32 * (limb_index - 1) + (trailing_zeros(limb) >> 1) + 1
            limb &= limb - 1

            # the rotation of which this is the lower qubit
            rotation = plan.rotation_at_lower[qind]
            if rotation != 0
                upper_qind = plan.upper_of_lower[qind]
                if upper_qind == 0 || !_readbit(candidates, 2 * (upper_qind - 1))
                    n_found += 1
                    orbit_rotations[n_found] = rotation
                end
            end

            # the rotations of which this is the upper qubit
            for upper_rotation in plan.rotations_at_upper[qind]
                if !_readbit(candidates, plan.pivots[upper_rotation] & ~1)
                    n_found += 1
                    orbit_rotations[n_found] = upper_rotation
                end
            end
        end
    end

    _sortbypivot!(orbit_rotations, n_found, plan.pivots)
    return n_found
end

# The rotations are found in the order of the qubits on which the Pauli string anticommutes, which is close to that of their lowest qubits.
function _sortbypivot!(orbit_rotations::Vector{Int32}, n::Int, pivots::Vector{Int})
    for i in 2:n
        rotation = orbit_rotations[i]
        j = i - 1
        while j >= 1 && pivots[orbit_rotations[j]] > pivots[rotation]
            orbit_rotations[j+1] = orbit_rotations[j]
            j -= 1
        end
        orbit_rotations[j+1] = rotation
    end
    return orbit_rotations
end

@inline function _locatebyrotation!(orbit_rotations::Vector{Int32}, plan, pstr)
    n_orbit_rotations = _anticommuting!(orbit_rotations, plan, pstr)
    representative = pstr
    index_in_orbit = zero(UInt64)

    # a Pauli string of an orbit with more rotations than a block holds needs no representative
    if n_orbit_rotations > _MAX_BLOCK_ROTATIONS
        return representative, index_in_orbit, n_orbit_rotations
    end

    for bit in 1:n_orbit_rotations
        rotation = orbit_rotations[bit]
        if _readbit(representative, plan.pivots[rotation])
            representative ⊻= plan.masks[rotation]
            index_in_orbit |= one(UInt64) << (bit - 1)
        end
    end

    return representative, index_in_orbit, n_orbit_rotations
end
