###
##
# The scratch memory of a layer.
# A sublayer turns every Pauli string into a record: the representative of its orbit, its coefficient and a label.
# An array sum keeps the records in the arrays of its propagation cache, so a layer needs little memory of its own.
##
###

# an orbit of more stages than this is not transformed as a block
const _MAX_BLOCK_STAGES = 8

# the number of stages of an orbit takes the lowest bits of a label
const _STAGE_BITS = 7

# a label holds the coordinate of this many stages
const _MAX_STAGES = 8 * sizeof(Int) - _STAGE_BITS

# the label of an orbit that is transformed as a block holds a hash of the representative from this bit on
const _HASH_SHIFT = _STAGE_BITS + _MAX_BLOCK_STAGES
const _HASH_BITS = 8 * sizeof(Int) - _HASH_SHIFT

# the records of a partition take about this many bytes
const _PARTITION_BYTES = 1 << 18

const _MAX_PARTITION_BITS = 12

# a task writes to the sum once it has made this many pairs
const _TASK_BUFFER_LENGTH = 4096


### Labels

"""
    _label(representative, coordinate, n_stages)

The label of a record: the number of stages of its orbit, above it the coordinate of the Pauli string in the orbit,
and above that a hash of the representative, if the orbit is transformed as a block.
An orbit of more stages needs the bits for its coordinate, and its representative is hashed wherever the hash is needed.
"""
@inline function _label(representative, coordinate::UInt64, n_stages::Int)
    label = (coordinate << _STAGE_BITS) | (n_stages % UInt64)
    if n_stages <= _MAX_BLOCK_STAGES
        label |= _hashbits(representative) << _HASH_SHIFT
    end
    return label % Int
end

@inline _nstages(label::Int) = Int((label % UInt64) & ((one(UInt64) << _STAGE_BITS) - 1))

@inline function _coordinate(label::Int)
    coordinate = (label % UInt64) >> _STAGE_BITS
    if _nstages(label) <= _MAX_BLOCK_STAGES
        coordinate &= (one(UInt64) << _MAX_BLOCK_STAGES) - 1
    end
    return coordinate
end

@inline function _hashbits(label::Int, representative)
    if _nstages(label) <= _MAX_BLOCK_STAGES
        return (label % UInt64) >> _HASH_SHIFT
    else
        return _hashbits(representative)
    end
end

# A hash of `_HASH_BITS` bits. The limbs are multiplied independently of each other, so that the hash of a wide Pauli string is no chain over its limbs.
@inline function _hashbits(representative)
    limbs = _limbs(representative)
    folded = zero(UInt64)
    for i in eachindex(limbs)
        folded ⊻= limbs[i] * _foldfactor(i)
    end
    return _mixbits(folded) >> _HASH_SHIFT
end

_splitmix(x::UInt64) = _mixbits(x * 0x9e3779b97f4a7c15)

@inline function _mixbits(x::UInt64)
    x = (x ⊻ (x >> 33)) * 0xff51afd7ed558ccd
    x = (x ⊻ (x >> 33)) * 0xc4ceb9fe1a85ec53
    return x ⊻ (x >> 33)
end

# an odd factor for every limb
const _FOLD_FACTORS = ntuple(i -> _splitmix(UInt64(i)) | one(UInt64), 64)

@inline _foldfactor(i::Int) = _FOLD_FACTORS[((i-1)&63)+1] + ((UInt64(i - 1) >> 6) << 1)


### Where a sublayer writes the pairs of Pauli string and coefficient it makes

# the auxiliary arrays of a cache written by one task, which grow as they fill
mutable struct ArraySink{PC}
    prop_cache::PC
    n_written::Int
end

# The auxiliary arrays of a cache written by several tasks. A task collects what it makes in a buffer and copies it into
# a range of the arrays that it reserves. The arrays do not grow while the tasks write, so a task that finds them full
# keeps what it makes in its buffer.
mutable struct TaskSink{TT,CT}
    buffer_terms::Vector{TT}
    buffer_coeffs::Vector{CT}
    n_written::Int
    sum_terms::Vector{TT}
    sum_coeffs::Vector{CT}
    n_reserved::Threads.Atomic{Int}
    is_full::Bool
end

function TaskSink(::Type{TT}, ::Type{CT}) where {TT,CT}
    n_room = _TASK_BUFFER_LENGTH + (1 << _MAX_BLOCK_STAGES)
    return TaskSink{TT,CT}(Vector{TT}(undef, n_room), Vector{CT}(undef, n_room), 0, TT[], CT[], Threads.Atomic{Int}(0), false)
end

# any other sum, through `add!`
struct SumSink{TS}
    term_sum::TS
end


### The memory of one task

# the entries of an orbit of more stages than a block holds, which `_transformlongorbit!` works on
mutable struct LongOrbit{TT,CT}
    n_entries::Int
    coordinates::Vector{UInt64}
    terms::Vector{TT}
    coeffs::Vector{CT}
    weights::Vector{Int}
    present::Vector{Bool}
    # the last stage that changed the entry
    stages::Vector{Int}
    # the entry of every coordinate, 0 where a slot is empty
    slots::Vector{Int32}
end

function LongOrbit(::Type{TT}, ::Type{CT}) where {TT,CT}
    return LongOrbit{TT,CT}(0, UInt64[], TT[], CT[], Int[], Bool[], Int[], zeros(Int32, 64))
end

# What a rotation does to the pairs of an orbit in which other rotations act on its qubits.
# The other rotations change the Paulis that the rotation finds there, so the signs and the weight change
# depend on whether those rotations are selected in a pair, which two bits of its coordinate tell.
struct SharedStage
    neighbor_bits::NTuple{2,Int}
    signs_from_lower::NTuple{4,Int8}
    signs_from_upper::NTuple{4,Int8}
    weight_changes::NTuple{4,Int8}
    # whether the tables hold, which they do unless more than two rotations share a qubit
    is_tabulated::Bool
end

mutable struct TaskWorkspace{TT,CT}
    # the rotations that anticommute with the Pauli string at hand, and the Paulis on their qubits
    positions::Vector{Int32}
    local_paulis::Vector{Int}
    shared_stages::Vector{SharedStage}

    # the orbits of one partition
    slots::Vector{Int32}
    orbit_hashes::Vector{UInt64}
    first_records::Vector{Int}
    orbit_blocks::Vector{Int}

    # the orbits of more stages than a block holds: their first records, and the next record of the same orbit for every record
    n_long_orbits::Int
    first_partition_record::Int
    long_orbit_records::Vector{Int}
    last_records::Vector{Int}
    next_records::Vector{Int}
    long_orbit::LongOrbit{TT,CT}

    # A block of coefficients for every orbit. The blocks of the orbits with `k` stages lie one after the other at index `k`,
    # so that every loop over them runs the same number of times.
    n_blocks::Vector{Int}
    block_records::Vector{Vector{Int}}
    block_coeffs::Vector{Vector{CT}}
    block_present::Vector{Vector{Bool}}

    # the Pauli strings of one orbit and their weights
    orbit_terms::Vector{TT}
    orbit_weights::Vector{Int}
end

function TaskWorkspace(::Type{TT}, ::Type{CT}) where {TT,CT}
    n_entries = 1 << _MAX_BLOCK_STAGES
    return TaskWorkspace{TT,CT}(Int32[], Int[], Vector{SharedStage}(undef, _MAX_BLOCK_STAGES), Int32[], UInt64[], Int[], Int[],
        0, 1, Int[], Int[], Int[], LongOrbit(TT, CT),
        zeros(Int, _MAX_BLOCK_STAGES), [Int[] for _ in 1:_MAX_BLOCK_STAGES],
        [CT[] for _ in 1:_MAX_BLOCK_STAGES], [Bool[] for _ in 1:_MAX_BLOCK_STAGES],
        Vector{TT}(undef, n_entries), Vector{Int}(undef, n_entries))
end


### The memory of a layer

"""
    LayerWorkspace(TT, CT)

The scratch memory of a `RotationLayer` applied to Pauli strings of the type `TT` with coefficients of the type `CT`.
"""
mutable struct LayerWorkspace{TT,CT}
    # the labels of the records that are sorted by partition
    sorted_labels::Vector{Int}

    # the records of a sum that has no arrays to keep them in
    record_terms::Vector{TT}
    record_coeffs::Vector{CT}
    record_labels::Vector{Int}

    # the records that every zone of a multi sum collects
    zone_terms::Vector{Vector{TT}}
    zone_coeffs::Vector{Vector{CT}}
    zone_labels::Vector{Vector{Int}}

    # per partition, and per partition and task
    partition_starts::Vector{Int}
    partition_counts::Matrix{Int}

    tasks::Vector{TaskWorkspace{TT,CT}}
    sinks::Vector{TaskSink{TT,CT}}
end

function LayerWorkspace(::Type{TT}, ::Type{CT}) where {TT,CT}
    return LayerWorkspace{TT,CT}(Int[], TT[], CT[], Int[], Vector{TT}[], Vector{CT}[], Vector{Int}[], Int[], zeros(Int, 0, 0),
        TaskWorkspace{TT,CT}[], TaskSink{TT,CT}[])
end

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
        push!(workspace.sinks, TaskSink(TT, CT))
    end

    for task_id in 1:n_tasks
        _ensurelength!(tasks[task_id].positions, n_rotations)
        _ensurelength!(tasks[task_id].local_paulis, n_rotations)
    end
    return tasks
end

# at least `n` elements
function _ensurelength!(array::Vector, n::Int)
    if length(array) < n
        resize!(array, max(n, 2 * length(array)))
    end
    return array
end
