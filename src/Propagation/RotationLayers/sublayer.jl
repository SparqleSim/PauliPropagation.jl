###
##
# A sublayer applied to a propagation cache.
# Every Pauli string becomes a record, the records are sorted by partition, and partition by partition
# the records of every orbit are collected and the rotations applied to the orbit.
# On arrays, all of this is done by several tasks: they share out the Pauli strings, and then the partitions.
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

"""
    _applyinorbits!(applyrotation!, layer::RotationLayer, prop_cache, theta, truncation; thread=true)

Applies the sublayers of the layer one after the other, orbit by orbit.
A sublayer that meets an orbit of more stages than a label holds is applied by `applyrotation!(rotation, angle)`, rotation by rotation.
"""
function _applyinorbits!(applyrotation!::F, layer::RotationLayer, prop_cache::AbstractPauliPropagationCache, theta, truncation;
    thread::Bool=true) where {F}

    workspace = _takeworkspace(paulitype(prop_cache), coefftype(prop_cache))
    try
        for sublayer in layer.sublayers
            plan = SubLayerPlan(layer, sublayer, theta, paulitype(prop_cache), nqubits(prop_cache))
            if !_applysublayer!(StorageType(prop_cache), prop_cache, plan, truncation, workspace; thread)
                _applyrotations!(applyrotation!, layer, sublayer, theta)
            end
        end
    finally
        _putbackworkspace!(workspace)
    end
    return prop_cache
end

# the rotations of the sublayer one by one
function _applyrotations!(applyrotation!::F, layer::RotationLayer, sublayer::Vector{Int}, theta) where {F}
    for index in sublayer
        applyrotation!(PauliRotation(layer.symbols, layer.qinds[index]), _rotationangle(theta, index))
    end
    return
end


### Arrays

# Returns `false`, with the sum of the cache as it was, if the sum holds an orbit of more stages than a label holds.
# The records are written into the auxiliary arrays and sorted into the main arrays, whose Pauli strings they replace.
# What the sublayer makes is written into the auxiliary arrays, which become the sum.
function _applysublayer!(::PropagationBase.ArrayStorage, prop_cache::AbstractPauliPropagationCache, plan::SubLayerPlan, truncation,
    workspace::LayerWorkspace; thread::Bool=true)

    task_partitioner, n_tasks = PropagationBase._preparetasks(activesize(prop_cache), thread)
    return _applysublayerintasks!(prop_cache, plan, truncation, workspace, task_partitioner, n_tasks)
end

function _applysublayerintasks!(prop_cache::AbstractPauliPropagationCache, plan::SubLayerPlan{TT}, truncation,
    workspace::LayerWorkspace{TT,CT}, task_partitioner, n_tasks::Int) where {TT,CT}

    n_terms = activesize(prop_cache)
    if n_terms == 0
        return true
    end

    # the arrays do not grow while several tasks write to them
    if n_tasks > 1
        PropagationBase._ensurecapacity!(prop_cache, n_terms + n_terms ÷ 2)
    end
    main_terms, main_coeffs, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(prop_cache)
    record_labels = indices(prop_cache)
    sorted_labels = _ensurelength!(workspace.sorted_labels, n_terms)
    PropagationBase._checkfits(n_terms, main_terms, main_coeffs)
    PropagationBase._checkfits(n_terms, aux_terms, aux_coeffs)
    PropagationBase._checkfits(n_terms, record_labels, sorted_labels)

    tasks = _taskworkspaces!(workspace, n_tasks, length(plan.masks))
    n_bits = _partitionbits(n_terms, sizeof(TT) + sizeof(CT) + sizeof(Int))
    n_partitions = 1 << n_bits
    partition_counts = _partitioncounts!(workspace, n_partitions, n_tasks)

    # a record for every Pauli string, at the index of the Pauli string
    is_written = Vector{Bool}(undef, n_tasks)
    function write_chunk!(task_id)
        chunk = task_partitioner[task_id]
        source = zip(view(main_terms, chunk), view(main_coeffs, chunk))
        is_written[task_id] = _writerecords!(aux_terms, aux_coeffs, record_labels, view(partition_counts, :, task_id), n_bits,
            chunk.start, tasks[task_id].positions, plan, source)
    end
    PropagationBase._eachtask(write_chunk!, n_tasks)

    if !all(is_written)
        return false
    end

    partition_starts = _partitionstarts!(workspace.partition_starts, partition_counts)
    function sort_chunk!(task_id)
        chunk = task_partitioner[task_id]
        _sortrecords!(main_terms, main_coeffs, sorted_labels, view(partition_counts, :, task_id), n_bits,
            aux_terms, aux_coeffs, record_labels, chunk.start, chunk.stop)
    end
    PropagationBase._eachtask(sort_chunk!, n_tasks)

    if n_tasks == 1
        n_written = _transformpartitions!(prop_cache, plan, truncation, tasks[1], sorted_labels, partition_starts)
    else
        n_written = _transformpartitionsintasks!(prop_cache, plan, truncation, workspace, sorted_labels, partition_starts, n_tasks)
    end

    PropagationBase._commitwrite!(prop_cache, n_written, 0)
    return true
end

# One task takes the partitions in turn and writes into the auxiliary arrays as it goes. Returns the number of pairs written.
function _transformpartitions!(prop_cache::AbstractPauliPropagationCache, plan::SubLayerPlan, truncation, task::TaskWorkspace,
    sorted_labels::Vector{Int}, partition_starts::Vector{Int})

    sink = ArraySink(prop_cache, 0)
    for partition in 1:length(partition_starts)-1
        # the arrays of the cache are looked up for every partition, since they are replaced when they grow
        sorted_terms, sorted_coeffs, _, _ = PropagationBase._mainauxarrays(prop_cache)
        _transformpartition!(sink, task, plan, truncation, sorted_terms, sorted_coeffs, sorted_labels,
            partition_starts[partition], partition_starts[partition+1] - 1)
    end
    return sink.n_written
end

# Several tasks share out the partitions and write into ranges of the auxiliary arrays that they reserve.
# What does not fit the arrays is left in the buffers of the tasks and written once the arrays have grown.
function _transformpartitionsintasks!(prop_cache::AbstractPauliPropagationCache, plan::SubLayerPlan, truncation, workspace::LayerWorkspace,
    sorted_labels::Vector{Int}, partition_starts::Vector{Int}, n_tasks::Int)

    sorted_terms, sorted_coeffs, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(prop_cache)
    tasks = workspace.tasks
    sinks = workspace.sinks
    n_reserved = Threads.Atomic{Int}(0)
    for task_id in 1:n_tasks
        _opensink!(sinks[task_id], aux_terms, aux_coeffs, n_reserved)
    end

    n_partitions = length(partition_starts) - 1
    function transform_partitions!(task_id)
        for partition in task_id:n_tasks:n_partitions
            _transformpartition!(sinks[task_id], tasks[task_id], plan, truncation, sorted_terms, sorted_coeffs, sorted_labels,
                partition_starts[partition], partition_starts[partition+1] - 1)
        end
        _flush!(sinks[task_id])
    end
    PropagationBase._eachtask(transform_partitions!, n_tasks)

    n_written = n_reserved[]
    n_left = sum(sinks[task_id].n_written for task_id in 1:n_tasks)
    if n_left > 0
        PropagationBase._ensurecapacity!(prop_cache, n_written + n_left)
        _, _, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(prop_cache)
        PropagationBase._checkfits(n_written + n_left, aux_terms, aux_coeffs)

        for task_id in 1:n_tasks
            sink = sinks[task_id]
            copyto!(aux_terms, n_written + 1, sink.buffer_terms, 1, sink.n_written)
            copyto!(aux_coeffs, n_written + 1, sink.buffer_coeffs, 1, sink.n_written)
            n_written += sink.n_written
        end
    end

    for task_id in 1:n_tasks
        _closesink!(sinks[task_id])
    end
    return n_written
end

function _transformpartition!(sink, task::TaskWorkspace, plan::SubLayerPlan, truncation, record_terms, record_coeffs, record_labels,
    lo::Int, hi::Int)

    if lo <= hi
        _grouporbits!(task, sink, truncation, record_terms, record_coeffs, record_labels, lo, hi)
        _transformorbits!(sink, task, plan, truncation, record_terms, record_coeffs, record_labels)
    end
    return sink
end


### Any sum that is iterated and added to

# The records are kept in the workspace and sorted within their arrays. They hold all of the sum,
# so the sum is emptied and takes what the sublayer makes, instead of a second sum of its size.
function _applysublayer!(::PropagationBase.DictStorage, prop_cache::AbstractPauliPropagationCache, plan::SubLayerPlan{TT}, truncation,
    workspace::LayerWorkspace{TT,CT}; thread::Bool=true) where {TT,CT}

    PropagationBase._checkauxempty(prop_cache)
    main_sum = mainsum(prop_cache)
    n_terms = length(main_sum)

    record_terms = _ensurelength!(workspace.record_terms, n_terms)
    record_coeffs = _ensurelength!(workspace.record_coeffs, n_terms)
    record_labels = _ensurelength!(workspace.record_labels, n_terms)

    task = first(_taskworkspaces!(workspace, 1, length(plan.masks)))
    n_bits = _partitionbits(n_terms, sizeof(TT) + sizeof(CT) + sizeof(Int))
    partition_counts = _partitioncounts!(workspace, 1 << n_bits, 1)

    if !_writerecords!(record_terms, record_coeffs, record_labels, view(partition_counts, :, 1), n_bits, 1, task.positions, plan, main_sum)
        return false
    end

    partition_starts = _partitionstarts!(workspace.partition_starts, partition_counts)
    _sortrecords!(record_terms, record_coeffs, record_labels, partition_starts, view(partition_counts, :, 1), n_bits)

    empty!(main_sum)
    sink = SumSink(main_sum)
    for partition in 1:length(partition_starts)-1
        _transformpartition!(sink, task, plan, truncation, record_terms, record_coeffs, record_labels,
            partition_starts[partition], partition_starts[partition+1] - 1)
    end

    return true
end


### Writing to a sink

const ArraysSink = Union{ArraySink,TaskSink}

@inline function _emit!(sink::ArraysSink, pstr, coeff)
    sink_terms, sink_coeffs = _roomtoemit!(sink, 1)
    n_written = sink.n_written + 1
    sink_terms[n_written] = pstr
    sink_coeffs[n_written] = coeff
    sink.n_written = n_written
    return
end

# the entries of the block of an orbit that are present
function _emitblock!(sink::ArraysSink, orbit_terms, block_coeffs, block_present, block_start::Int, n_entries::Int)
    sink_terms, sink_coeffs = _roomtoemit!(sink, n_entries)
    n_written = sink.n_written

    # every entry is written, and the next one writes over it if it is not present
    @inbounds for entry in 1:n_entries
        sink_terms[n_written+1] = orbit_terms[entry]
        sink_coeffs[n_written+1] = block_coeffs[block_start+entry]
        n_written += block_present[block_start+entry]
    end

    sink.n_written = n_written
    return
end

# the arrays to write to, with room for `n_more` pairs past the ones written
@inline function _roomtoemit!(sink::ArraySink, n_more::Int)
    prop_cache = sink.prop_cache
    n_needed = sink.n_written + n_more
    if n_needed > capacity(prop_cache)
        PropagationBase._ensurecapacity!(prop_cache, n_needed)
    end

    aux_sum = auxsum(prop_cache)
    aux_terms = terms(aux_sum)
    aux_coeffs = coefficients(aux_sum)
    PropagationBase._checkfits(n_needed, aux_terms, aux_coeffs)
    return aux_terms, aux_coeffs
end

@inline function _roomtoemit!(sink::TaskSink, n_more::Int)
    if sink.n_written + n_more > _TASK_BUFFER_LENGTH && !sink.is_full
        _flush!(sink)
    end

    n_needed = sink.n_written + n_more
    if n_needed > length(sink.buffer_terms) || n_needed > length(sink.buffer_coeffs)
        _ensurelength!(sink.buffer_terms, n_needed)
        _ensurelength!(sink.buffer_coeffs, n_needed)
    end
    return sink.buffer_terms, sink.buffer_coeffs
end

# an empty buffer in front of the arrays that the tasks write to, of which `n_reserved` pairs are taken
function _opensink!(sink::TaskSink{TT,CT}, sum_terms::Vector{TT}, sum_coeffs::Vector{CT}, n_reserved::Threads.Atomic{Int}) where {TT,CT}
    sink.n_written = 0
    sink.sum_terms = sum_terms
    sink.sum_coeffs = sum_coeffs
    sink.n_reserved = n_reserved
    sink.is_full = false
    return sink
end

# Lets go of the arrays of the sum, and of a buffer that had to keep more than it started with room for.
function _closesink!(sink::TaskSink{TT,CT}) where {TT,CT}
    n_room = _TASK_BUFFER_LENGTH + (1 << _MAX_BLOCK_STAGES)
    if length(sink.buffer_terms) > n_room
        sink.buffer_terms = Vector{TT}(undef, n_room)
        sink.buffer_coeffs = Vector{CT}(undef, n_room)
    end
    sink.sum_terms = TT[]
    sink.sum_coeffs = CT[]
    sink.n_written = 0
    return sink
end

# Copies the buffer into a range of the arrays that no other task writes to, unless the arrays are full.
function _flush!(sink::TaskSink)
    n_buffered = sink.n_written
    if n_buffered == 0 || sink.is_full
        return sink
    end

    n_room = min(length(sink.sum_terms), length(sink.sum_coeffs))
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

    copyto!(sink.sum_terms, n_taken + 1, sink.buffer_terms, 1, n_buffered)
    copyto!(sink.sum_coeffs, n_taken + 1, sink.buffer_coeffs, 1, n_buffered)
    sink.n_written = 0
    return sink
end

@inline function _emit!(sink::SumSink, pstr, coeff)
    add!(sink.term_sum, pstr, coeff)
    return
end

function _emitblock!(sink::SumSink, orbit_terms, block_coeffs, block_present, block_start::Int, n_entries::Int)
    for entry in 1:n_entries
        if block_present[block_start+entry]
            add!(sink.term_sum, orbit_terms[entry], block_coeffs[block_start+entry])
        end
    end
    return
end
