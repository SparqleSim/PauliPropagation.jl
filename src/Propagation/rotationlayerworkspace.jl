###
##
# The scratch memory of a `RotationLayer`: what every task of a pass works in, and the workspace that the layers of a
# propagation reuse.
##
###

struct TaskWorkspace{TT,CT}
    # the records of one partition grouped by the hash of their class: the groups found through `slots`, and the Pauli
    # strings and coefficients of the records one group after the other
    slots::Vector{Int32}
    group_of::Vector{Int32}
    group_hashes::Vector{UInt64}
    group_starts::Vector{Int}
    group_terms::Vector{TT}
    group_coeffs::Vector{CT}

    # the rotations that anticommute with the Pauli strings of a class, in the order of the layer
    rotations::Vector{Int32}

    # what the kernels rotate a class in
    block::BlockScratch{TT,CT}
    table::TableScratch{TT,CT}

    # the buffers of the task's sinks: one for an array sum that several tasks write to, one per zone for a multi sum
    sink_terms::Vector{Vector{TT}}
    sink_coeffs::Vector{Vector{CT}}
end

TaskWorkspace{TT,CT}() where {TT,CT} = TaskWorkspace{TT,CT}(Int32[], Int32[], UInt64[], Int[], TT[], CT[], Int32[],
    BlockScratch{TT,CT}(), TableScratch{TT,CT}(), Vector{TT}[], Vector{CT}[])

"""
    LayerWorkspace(TT, CT)

The scratch memory of a `RotationLayer` applied to Pauli strings of the type `TT` with coefficients of the type `CT`,
which the layers of a propagation reuse.
"""
mutable struct LayerWorkspace{TT,CT}
    tasks::Vector{TaskWorkspace{TT,CT}}

    # the labels of the Pauli strings that every source holds, found when its records are counted
    source_labels::Vector{Vector{Int}}

    # the records of every zone that has no arrays to keep them in, those of a Pauli sum in the first
    zone_terms::Vector{Vector{TT}}
    zone_coeffs::Vector{Vector{CT}}
    zone_labels::Vector{Vector{Int}}
end

LayerWorkspace(::Type{TT}, ::Type{CT}) where {TT,CT} =
    LayerWorkspace{TT,CT}(TaskWorkspace{TT,CT}[], Vector{Int}[], Vector{TT}[], Vector{CT}[], Vector{Int}[])

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
