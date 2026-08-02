tonumber(x::Number) = x

_invertfunc(func::F) where {F<:Function} = args -> !func(args...)

# thread=false runs everything on a single thread
maxtasks(thread::Bool) = thread ? Threads.nthreads() : 1

# Below this many elements, a single sequential task is used regardless of thread count
# found via trial and error
const _MIN_ELEMS_PER_TASK = 16384

# Bundles the task_partitioner + n_tasks setup shared by every task-partitioned merge/apply pass.
function _preparetasks(n::Int, thread::Bool)
    task_partitioner = AK.TaskPartitioner(n, maxtasks(thread), _MIN_ELEMS_PER_TASK)
    return task_partitioner, task_partitioner.num_tasks
end

# Turns per-task element counts into cumulative write offsets: offsets[t] is where task t's
# output begins, and offsets[end] - 1 is the total count.
function _offsetsfromcounts(counts::AbstractVector{Int})
    offsets = Vector{Int}(undef, length(counts) + 1)
    offsets[1] = 1
    for t in eachindex(counts)
        offsets[t+1] = offsets[t] + counts[t]
    end
    return offsets
end

# the positions of the set bits of `mask`, ascending
function _masksetbits(mask::TT) where {TT}
    bits = Int[]
    while !iszero(mask)
        push!(bits, trailing_zeros(mask))
        mask &= mask - one(TT)
    end
    return bits
end

# for CPU-only code using Threads.@spawn
# GPU extensions override this for their array for fallback functionality
_iscpuarray(::AbstractArray) = true

function _thrownotimplemented(::Type{T}, func_name::Symbol) where T
    error("Function '", func_name, "' not implemented for type '", T, "'.")
end

function _thrownotimplemented(::T, func_name::Symbol) where T
    _thrownotimplemented(T, func_name)
end