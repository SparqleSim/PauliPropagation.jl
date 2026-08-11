using Base.Threads

println("Threads: $(nthreads())")

### Memory bandwidth vs thread count

function readsum(data, lo, hi)
    total = zero(eltype(data))
    @inbounds @simd for ii in lo:hi
        total += data[ii]
    end
    return total
end

function bandwidth(data, sums, n_tasks)
    chunk = cld(length(data), n_tasks)
    t0 = time_ns()
    @sync for task_id in 1:n_tasks
        Threads.@spawn sums[task_id] = readsum(data, (task_id - 1) * chunk + 1, min(task_id * chunk, length(data)))
    end
    return sizeof(data) / ((time_ns() - t0) * 1e-9) / 2^30
end

sums = zeros(nthreads())
for bytes in (8 * 2^20, 64 * 2^20, 512 * 2^20)
    data = rand(Float64, bytes ÷ 8)
    print(rpad("read $(bytes ÷ 2^20) MiB:", 20))
    for n_tasks in (1, 2, 4, 8, nthreads())
        rate = maximum(bandwidth(data, sums, n_tasks) for _ in 1:5)
        print(" ", n_tasks, "thr=", round(rate, digits=1), " GiB/s")
    end
    println()
end

### Per-thread throughput under an all-core load, on work that fits in L1

function spin(rounds)
    x = 1.0000001
    acc = 1.0
    @inbounds for _ in 1:rounds
        acc = acc * x + 1e-9
    end
    return acc
end

function corespread(n_tasks, rounds, sums)
    times = zeros(n_tasks)
    @sync for task_id in 1:n_tasks
        Threads.@spawn begin
            t0 = time_ns()
            sums[task_id] = spin(rounds)
            times[task_id] = (time_ns() - t0) * 1e-9
        end
    end
    return times
end

spin(10)
println("\nIdentical compute-bound work on every thread (seconds each):")
for n_tasks in (1, 4, 8, nthreads())
    times = argmin(maximum, [corespread(n_tasks, 200_000_000, sums) for _ in 1:3])
    println(rpad("  $n_tasks threads:", 16), " min=", round(minimum(times), digits=3),
        " max=", round(maximum(times), digits=3),
        " max/min=", round(maximum(times) / minimum(times), digits=2))
end

### Cost of one @threads region

function emptyregion(n_zones)
    @threads for _ in 1:n_zones
        nothing
    end
end

emptyregion(nthreads())
println("\nEmpty @threads region (microseconds):")
for n_zones in (2, 4, 8, nthreads())
    reps = 20_000
    t = @elapsed for _ in 1:reps
        emptyregion(n_zones)
    end
    println(rpad("  $n_zones iterations:", 20), round(t / reps * 1e6, digits=2))
end
