###
##
# How the package threads. A pass over a term sum is split into tasks of at least
# `_MIN_ELEMS_PER_TASK` elements, and a propagation keeps a worker task pinned to every thread of
# the default pool for the whole of its gate loop, so that every pass hands its tasks to workers
# that are already running instead of starting a task per pass and per gate.
#
# `@threads` and `@spawn` start their tasks one after the other from the calling thread, about
# 3 us each, so a pass on 64 threads costs 200 us before any work is done, several times per gate.
# A worker that is already running and watching the round counter picks its tasks up within
# microseconds however many threads there are. And because task `i` always goes to thread `i`,
# the part of a sum it works stays on the memory of the NUMA node that first touched it, which is
# what the bandwidth of a merge depends on. This is what an OpenMP runtime does between parallel
# regions; Julia's own threads go to sleep instead.
#
# The workers are the threads of the default pool and nothing else, as with `@threads`. The task
# that owns them is on one of those threads too: it works its share of every round and waits for
# the rest. When the calling thread is not in the pool (the main thread of Julia 1.12 is an
# interactive thread), `f` is run on a task pinned to a pool thread and the caller waits once for
# all of it. Waking the caller once per round instead would go through the OS thousands of times,
# and on a machine with as many cores as threads each wake-up waits a scheduler slice behind a
# spinning worker: milliseconds per round, 4x on 18 steps of the 6x6 TFIM.
#
# An idle worker spins for a bounded time and then sleeps on a condition. A spinning worker never
# yields its thread to another task, so every few spins it also asks the scheduler whether a task
# is queued for its thread or for the pool (the kernels of AcceleratedKernels start tasks of their
# own, and so may a `@threads` loop in a step) and sleeps at once if so; the scheduler then runs
# that task on the thread, and the next round wakes the worker.
##
###


### Splitting a pass into tasks

# Below this many elements, a pass runs as a single task regardless of thread count; found via
# trial and error.
const _MIN_ELEMS_PER_TASK = 16384

# thread=false runs everything on a single thread
maxtasks(thread::Bool) = thread ? Threads.nthreads() : 1

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

# Runs `f(task_id)` for every task of one pass: on the workers of the propagation in progress,
# task `i` on thread `i`, or else on a task spawned per call. `f` is called once per task, so neither
# this function nor `_round!` and `_worktasks` specialize on it: every pass would compile them anew.
Base.@nospecializeinfer function _eachtask(@nospecialize(f), n_tasks::Int)
    n_tasks == 1 && return f(1)

    workers = _currentworkers()
    if workers === nothing
        @sync for task_id in 1:n_tasks
            Threads.@spawn f(task_id)
        end
    else
        _round!(f, workers, n_tasks)
    end
    return
end

# The array to keep in place of `array`, with room for `n` elements. During a propagation, a growing
# array on the CPU is copied into a new one by the workers, each copying its own stripe; otherwise
# `array` itself is resized.
function _resizearray(array, n::Int)
    if n <= length(array) || !_iscpuarray(array) || _currentworkers() === nothing
        return resize!(array, n)
    end
    resized_array = similar(array, n)
    task_partitioner, n_tasks = _preparetasks(length(array), true)
    function copy_stripe!(task_id)
        chunk = task_partitioner[task_id]
        copyto!(resized_array, chunk.start, array, chunk.start, length(chunk))
    end
    _eachtask(copy_stripe!, n_tasks)
    return resized_array
end


### The workers of a propagation

mutable struct Workers
    @atomic round::Int          # bumped once per round; a worker runs when it sees it move
    @atomic pending::Int        # workers that have not finished the current round
    @atomic stop::Bool
    inround::Bool               # set by the owner while it works its share of a round
    job::Any                    # the f of the current round, boxed once for all workers to call
    n_tasks::Int                # the number of tasks of the current round
    error::Any                  # an exception a worker hit, rethrown by the owner
    tasks::Vector{Task}
    n_workers::Int              # one per thread of the default pool
    owner_id::Int               # the worker id of the owner, which works a share of every round itself
    owner::Task                 # the task that opened the workers, the only one that may use them
    wakeup::Threads.Condition   # where a worker that spun out waits for the next round
end

# The workers of the propagation in progress, if any. They take one round at a time from their
# owner alone, so a propagation started meanwhile from another task spawns its tasks instead, and
# so does a pass that the owner runs as its share of a round.
mutable struct WorkerSlot
    @atomic current::Union{Nothing,Workers}
end
const _WORKERS = WorkerSlot(nothing)

function _currentworkers()
    workers = @atomic :acquire _WORKERS.current
    return (workers !== nothing && workers.owner === current_task() && !workers.inround) ? workers : nothing
end

"""
    withworkers(f)

Run `f()` with a worker task pinned to every thread of the default pool, so that every threaded pass of a propagation inside `f` hands its tasks to them instead of starting a task per pass and per gate, and the same part of a sum lands on the same thread every time.
`propagate!` does this around its gate loop; call it yourself around a loop that applies gates one by one.
If the calling thread is not one of the pool (the main thread of Julia 1.12 is an interactive thread), `f` runs on a task pinned to one that is, and the caller waits for it.
Nothing changes when the pool has one thread, when workers are already up, or inside a `@threads` region, where a pinned task could not run.
"""
function withworkers(f::F) where {F}
    thread_ids = Threads.threadpooltids(:default)
    if length(thread_ids) == 1 || ccall(:jl_in_threaded_region, Cint, ()) != 0 || (@atomic :acquire _WORKERS.current) !== nothing
        return f()
    end

    # the owner has to stay on its pool thread; a task that may move (`@spawn`) is moved once
    owner_id = something(findfirst(==(Threads.threadid()), thread_ids), 0)
    (owner_id == 0 || !current_task().sticky) && return _onpoolthread(f, first(thread_ids))

    workers = Workers(0, 0, false, false, nothing, 0, nothing, Task[], length(thread_ids), owner_id,
        current_task(), Threads.Condition())
    (@atomicreplace _WORKERS.current nothing => workers).success || return f()

    _startworkers!(workers, thread_ids)
    try
        return f()
    finally
        _stopworkers!(workers)
        @atomic :release _WORKERS.current = nothing
    end
end

# Run `withworkers(f)` on a task pinned to pool thread `thread_id` and wait for it. An exception
# of `f` comes back as itself. An interrupt of the wait is passed on through the stop flag, which
# the owner acts on at its next round, and the wait then resumes until the owner has stopped.
function _onpoolthread(f::F, thread_id::Int) where {F}
    task = Task(() -> withworkers(f))
    task.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), task, thread_id - 1) == 1 || return f()
    schedule(task)
    try
        return fetch(task)
    catch err
        err isa TaskFailedException && rethrow(err.task.exception)
        workers = @atomic :acquire _WORKERS.current
        workers !== nothing && workers.owner === task && (@atomic workers.stop = true)
        try wait(task) catch end
        rethrow()
    end
end

# a worker on every pool thread but the owner's
function _startworkers!(workers::Workers, thread_ids)
    for (worker_id, thread_id) in enumerate(thread_ids)
        worker_id == workers.owner_id && continue
        task = Task(() -> _workerloop(workers, worker_id))
        task.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, thread_id - 1) == 1 ||
            error("could not pin a worker to thread $thread_id")
        schedule(task)
        push!(workers.tasks, task)
    end
    return
end

function _stopworkers!(workers::Workers)
    @atomic workers.stop = true
    _nextround!(workers)
    foreach(wait, workers.tasks)
    return
end



### A round of tasks

# One round over the tasks of a pass: publish the job, move the round on, work the owner's share,
# and wait for the workers.
Base.@nospecializeinfer function _round!(@nospecialize(f), workers::Workers, n_tasks::Int)
    # stopped from outside, by an interrupt of the task that waits for the owner
    (@atomic :acquire workers.stop) && throw(InterruptException())

    # f on its own: a worker that took it out of a tuple would copy it, and a closure holds the
    # gate's rule and mask, each as wide as a term
    workers.job = f
    workers.n_tasks = n_tasks
    workers.inround = true
    @atomic :release workers.pending = length(workers.tasks)
    _nextround!(workers)

    _worktasks(f, workers.owner_id, workers.n_workers, n_tasks)

    # the workers are all running by now, so this wait is bounded by their work
    while (@atomic :acquire workers.pending) > 0
        _spinwait()
    end
    workers.inround = false

    if workers.error !== nothing
        err = workers.error
        workers.error = nothing
        throw(err)
    end

    return
end

# Works every task striped onto this worker, round after round, until told to stop.
function _workerloop(workers::Workers, worker_id::Int)
    seen = 0
    while true
        seen = _awaitround(workers, seen)
        (@atomic workers.stop) && return

        f, n_tasks = workers.job, workers.n_tasks
        try
            _worktasks(f, worker_id, workers.n_workers, n_tasks)
        catch err
            workers.error = err
        end
        @atomic :release workers.pending -= 1
    end
end

# the tasks of one round striped onto this worker, each called through dynamic dispatch
Base.@nospecializeinfer function _worktasks(@nospecialize(f), worker_id::Int, n_workers::Int, n_tasks::Int)
    for task_id in worker_id:n_workers:n_tasks
        f(task_id)
    end
    return
end


### Waiting for the next round

# moves the round on and wakes the workers that sleep on it
function _nextround!(workers::Workers)
    @atomic :release workers.round += 1
    lock(workers.wakeup)
    try
        notify(workers.wakeup; all=true)
    finally
        unlock(workers.wakeup)
    end
    return
end

# How long an idle worker spins before it sleeps. It has to outlast the wait for the slowest task
# of a round, which is milliseconds on a large sum, because a worker that dozes off every round
# costs a few microseconds to wake, per worker, from the owner's thread: the very cost of
# `@threads` (200 us on 64 threads; 18 steps of the 6x6 TFIM went from 2.0 to 2.3 s with a 200 us
# window). It also caps how long a task started elsewhere in the process waits for a thread.
const _WORKER_SPIN_NS = 20_000_000

# Spin until the round moves past `seen`, or sleep on the condition once the spin window is out
# or the owner asks for it.
function _awaitround(workers::Workers, seen::Int)
    t_start = time_ns()
    spins = 0
    while true
        round = @atomic :acquire workers.round
        round != seen && return round
        _spinwait()
        spins += 1
        if spins == 64
            spins = 0
            (time_ns() - t_start > _WORKER_SPIN_NS || _taskpending()) && break
        end
    end

    # the round is looked at again under the lock, which `_nextround!` takes to notify
    lock(workers.wakeup)
    try
        while (@atomic :acquire workers.round) == seen
            wait(workers.wakeup)
        end
    finally
        unlock(workers.wakeup)
    end
    return @atomic :acquire workers.round
end

# Whether the scheduler has a task queued for this thread or for the pool. `workqueue_for` and
# `checktaskempty` are what the scheduler itself looks at to find one; on a Julia without them a
# worker sleeps after every spin window instead.
@static if isdefined(Base, :workqueue_for) && isdefined(Base, :checktaskempty)
    _taskpending() = !isempty(Base.workqueue_for(Threads.threadid())) || !Base.checktaskempty()
else
    _taskpending() = true
end

# One turn of a spin-wait: a CPU pause, a safepoint (a collection started by a working thread waits
# on every other thread), and the core offered to the OS for any thread that wants it, which on a
# machine with as many busy threads as cores would otherwise wait a scheduler slice of milliseconds.
# It costs nothing measurable when no thread wants the core.
@inline function _spinwait()
    ccall(:jl_cpu_pause, Cvoid, ())
    GC.safepoint()
    @static if Sys.iswindows()
        ccall(:SwitchToThread, stdcall, Cint, ())
    else
        ccall(:sched_yield, Cint, ())
    end
    return
end
