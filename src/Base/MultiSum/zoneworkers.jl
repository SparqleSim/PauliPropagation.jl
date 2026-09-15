###
##
# Zone workers: one task pinned to every thread of the default pool for the whole of a propagation,
# handed the zones round by round through an atomic counter.
#
# `@threads` starts its tasks one after the other from the calling thread, about 3 us each, so a
# round on 64 threads costs 200 us before any zone is worked, twice per gate. A worker that is
# already running and watching the round counter picks its zones up within microseconds however
# many threads there are. And because zone `z` always goes to the same thread, a zone's arrays
# stay on the memory of the NUMA node that first touched them, which is what the merge's bandwidth
# depends on. This is what an OpenMP runtime does between parallel regions; Julia's own threads go
# to sleep instead.
#
# The workers are the threads of the default pool and nothing else, as with `@threads`. The task
# that owns them is on one of those threads too: it works its share of every round and waits for
# the rest. When the calling thread is not in the pool (the main thread of Julia 1.12 is an
# interactive thread), `f` is run on a task pinned to a pool thread and the caller waits once for
# all of it. Waking the caller once per round instead would go through the OS thousands of times,
# and on a machine with as many cores as threads each wake-up waits a scheduler slice behind a
# spinning worker: milliseconds per round, 4x on 18 steps of the 6x6 TFIM.
#
# An idle worker spins for a bounded time and then sleeps on a condition, so that a task started
# elsewhere in the process gets a thread within that time. A task that a worker or the owner
# starts itself (a threaded reduction inside a zone, say) runs on the starter's own thread while
# it waits for it, so that costs nothing but the parallelism it hoped for.
##
###

mutable struct ZoneWorkers
    @atomic round::Int          # bumped once per round; a worker runs when it sees it move
    @atomic pending::Int        # workers that have not finished the current round
    @atomic stop::Bool
    job::Any                    # (zonefunc, n_zones) of the current round
    error::Any                  # the first exception a worker hit, rethrown by the owner
    tasks::Vector{Task}
    n_workers::Int              # one per thread of the default pool
    owner_id::Int               # the worker id of the owner, which works a share of every round itself
    owner::Task                 # the task that opened the workers, the only one that may use them
    wakeup::Threads.Condition   # where a worker that spun out waits for the next round
end

# The workers of the propagation in progress, if any. The workers take one job at a time from
# their owner alone, so a propagation started meanwhile from another task uses `@threads`.
mutable struct ZoneWorkerSlot
    @atomic current::Union{Nothing,ZoneWorkers}
end
const _ZONEWORKERS = ZoneWorkerSlot(nothing)

function _currentzoneworkers()
    workers = @atomic :acquire _ZONEWORKERS.current
    return (workers !== nothing && workers.owner === current_task()) ? workers : nothing
end

"""
    withzoneworkers(f)

Run `f()` with a worker task pinned to every thread of the default pool, so that the zones of a multi sum are handed to them on every gate instead of starting a task per zone and per gate.
`propagate!` does this around its gate loop; call it yourself around a loop that applies gates one by one.
If the calling thread is not one of the pool (the main thread of Julia 1.12 is an interactive thread), `f` runs on a task pinned to one that is, and the caller waits for it.
Nothing changes when the pool has one thread, when workers are already up, or inside a `@threads` region, where a pinned task could not run.
"""
function withzoneworkers(f::F) where {F}
    thread_ids = Threads.threadpooltids(:default)
    if length(thread_ids) == 1 || ccall(:jl_in_threaded_region, Cint, ()) != 0 || (@atomic :acquire _ZONEWORKERS.current) !== nothing
        return f()
    end

    # the owner has to stay on its pool thread; a task that may move (`@spawn`) is moved once
    owner_id = something(findfirst(==(Threads.threadid()), thread_ids), 0)
    (owner_id == 0 || !current_task().sticky) && return _onpoolthread(f, first(thread_ids))

    workers = ZoneWorkers(0, 0, false, nothing, nothing, Task[], length(thread_ids), owner_id,
        current_task(), Threads.Condition())
    (@atomicreplace _ZONEWORKERS.current nothing => workers).success || return f()

    _startzoneworkers!(workers, thread_ids)
    try
        return f()
    finally
        _stopzoneworkers!(workers)
        @atomic :release _ZONEWORKERS.current = nothing
    end
end

# Run `withzoneworkers(f)` on a task pinned to pool thread `thread_id` and wait for it. An exception
# of `f` comes back as itself. An interrupt of the wait is passed on through the stop flag, which
# the owner acts on at its next round, and the wait then resumes until the owner has stopped.
function _onpoolthread(f::F, thread_id::Int) where {F}
    task = Task(() -> withzoneworkers(f))
    task.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), task, thread_id - 1) == 1 || return f()
    schedule(task)
    try
        return fetch(task)
    catch err
        err isa TaskFailedException && rethrow(err.task.exception)
        workers = @atomic :acquire _ZONEWORKERS.current
        workers !== nothing && workers.owner === task && (@atomic workers.stop = true)
        try wait(task) catch end
        rethrow()
    end
end

# a worker on every pool thread but the owner's
function _startzoneworkers!(workers::ZoneWorkers, thread_ids)
    for (worker_id, thread_id) in enumerate(thread_ids)
        worker_id == workers.owner_id && continue
        task = Task(() -> _zoneworkerloop(workers, worker_id))
        task.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, thread_id - 1) == 1 ||
            error("could not pin a zone worker to thread $thread_id")
        schedule(task)
        push!(workers.tasks, task)
    end
    return
end

function _stopzoneworkers!(workers::ZoneWorkers)
    @atomic workers.stop = true
    _nextround!(workers)
    foreach(wait, workers.tasks)
    return
end

# moves the round on and wakes the workers that sleep on it
function _nextround!(workers::ZoneWorkers)
    @atomic :release workers.round += 1
    lock(workers.wakeup)
    try
        notify(workers.wakeup; all=true)
    finally
        unlock(workers.wakeup)
    end
    return
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

# How long an idle worker spins before it sleeps. It has to outlast the wait for the slowest zone
# of a round, which is milliseconds on a large sum, because a worker that dozes off every round
# costs a few microseconds to wake, per worker, from the owner's thread: the very cost of
# `@threads` (200 us on 64 threads; 18 steps of the 6x6 TFIM went from 2.0 to 2.3 s with a 200 us
# window). It also caps how long a task started elsewhere in the process waits for a thread.
const _WORKER_SPIN_NS = 20_000_000

# Spin until the round moves past `seen`, or sleep on the condition once the spin window is out.
function _awaitround(workers::ZoneWorkers, seen::Int)
    t_start = time_ns()
    spins = 0
    while true
        round = @atomic :acquire workers.round
        round != seen && return round
        _spinwait()
        spins += 1
        if spins == 64
            spins = 0
            time_ns() - t_start > _WORKER_SPIN_NS && break
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

# Work every zone striped onto this worker, round after round, until told to stop.
function _zoneworkerloop(workers::ZoneWorkers, worker_id::Int)
    seen = 0
    while true
        seen = _awaitround(workers, seen)
        (@atomic workers.stop) && return

        zonefunc, n_zones = workers.job::Tuple{Any,Int}
        try
            _workzones(zonefunc, worker_id, workers.n_workers, n_zones)
        catch err
            workers.error = err
        end
        @atomic :release workers.pending -= 1
    end
end

# a function barrier, so the zones run compiled for the round's zonefunc
@noinline function _workzones(zonefunc::F, worker_id::Int, n_workers::Int, n_zones::Int) where {F}
    for zone_id in worker_id:n_workers:n_zones
        zonefunc(zone_id)
    end
    return
end

# one round over the zones: publish the job, move the round on, work the owner's share, and wait
function _eachzoneworker(zonefunc::F, workers::ZoneWorkers, n_zones::Int) where {F}
    # stopped from outside, by an interrupt of the task that waits for the owner
    (@atomic :acquire workers.stop) && throw(InterruptException())

    workers.job = (zonefunc, n_zones)
    @atomic :release workers.pending = length(workers.tasks)
    _nextround!(workers)

    _workzones(zonefunc, workers.owner_id, workers.n_workers, n_zones)

    # the workers are all running by now, so this wait is bounded by their work
    while (@atomic :acquire workers.pending) > 0
        _spinwait()
    end

    if workers.error !== nothing
        err = workers.error
        workers.error = nothing
        throw(err)
    end

    return
end
