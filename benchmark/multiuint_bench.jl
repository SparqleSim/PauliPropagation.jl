# Benchmark: MultiUInt vs the legacy BitIntegers wide-integer backing.
#
# Issue #145 asks for a GPU-friendly integer type for >32-qubit Pauli
# strings. `MultiUInt` is opt-in (see `getinttype(...; use_multiuint=true)`)
# precisely because we do not want to switch the whole library over to it
# unless it is at worst competitive on CPU. This script measures that.
#
# It compares, at a few >64-bit widths, the per-operation cost of the
# operations that dominate `propagate` — `countweight`, `commutes`,
# `pauliprod` — plus an end-to-end `propagate` call, for both integer
# backings. The GPU section (run only when CUDA is functional) times the
# same bitwise kernels on `MultiUInt`, which the BitIntegers path cannot
# run on device at all.
#
# Usage:
#   julia --project=. benchmark/multiuint_bench.jl
#
# Notes on methodology: each timing is the minimum wall-clock over a few
# repeats after a warm-up call (minimum is the standard choice — it is the
# run least perturbed by GC and OS noise). For trustworthy GPU timings, run
# this with the device otherwise idle.

using PauliPropagation
using Random
using Printf

# Detect CUDA at top level (not inside a function): loading a package advances
# the world age, so an in-function `@eval using CUDA` would leave the module
# invisible to the already-compiled caller. CUDA is optional — the GPU section
# is simply skipped when it is absent or non-functional.
const _CUDA_OK = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# ---- Timing helpers ---------------------------------------------------

# A global sink that consumes every timed result, so the compiler cannot
# elide a "pure" kernel whose return value would otherwise be unused.
const _SINK = Ref{Any}(nothing)

"""Minimum wall-clock of `f` over `n` repeats, after one warm-up call."""
function bestof(f; n::Int=5)
    _SINK[] = f()                         # warm-up (compilation + caches)
    best = Inf
    for _ in 1:n
        t = @elapsed (_SINK[] = f())
        best = min(best, t)
    end
    return best
end

fmt_time(t) = t >= 1 ? @sprintf("%7.3f s ", t) : @sprintf("%7.2f ms", t * 1e3)

function report(label, t_legacy, t_multi)
    ratio = t_multi / t_legacy
    @printf("  %-26s legacy %s   multiuint %s   (×%.2f)\n",
            label, fmt_time(t_legacy), fmt_time(t_multi), ratio)
end

# ---- Random Pauli string of a chosen integer type ---------------------

function randpauli(::Type{T}, nq::Int, k::Int, rng) where {T}
    p = zero(T)
    for _ in 1:k
        # encoding: 1=X, 2=Z, 3=Y (0 = identity, skipped)
        p = setpauli(p, rand(rng, 1:3), rand(rng, 1:nq))
    end
    return p
end

# ---- Micro-benchmarks: the hot bitwise ops over a batch ---------------

# Each kernel accumulates a checksum so the work cannot be elided.

function sum_countweight(batch)
    s = 0
    @inbounds for p in batch
        s += countweight(p)
    end
    return s
end

function sum_commutes(as, bs)
    s = 0
    @inbounds for i in eachindex(as, bs)
        s += commutes(as[i], bs[i]) ? 1 : 0
    end
    return s
end

function sum_pauliprod(as, bs)
    s = 0
    @inbounds for i in eachindex(as, bs)
        term, _coeff = pauliprod(as[i], bs[i])
        s += countweight(term)
    end
    return s
end

function microbench(nq::Int; batch_size::Int=100_000, k::Int=8)
    T_legacy = getinttype(nq; use_multiuint=false)
    T_multi  = getinttype(nq; use_multiuint=true)
    @printf("\n[micro] nq=%d  (legacy %s vs %s, batch=%d)\n",
            nq, T_legacy, T_multi, batch_size)

    rng = MersenneTwister(2026)
    a_leg = [randpauli(T_legacy, nq, k, rng) for _ in 1:batch_size]
    rng = MersenneTwister(2026)
    b_leg = [randpauli(T_legacy, nq, k, rng) for _ in 1:batch_size]
    rng = MersenneTwister(2026)
    a_mul = [randpauli(T_multi, nq, k, rng) for _ in 1:batch_size]
    rng = MersenneTwister(2026)
    b_mul = [randpauli(T_multi, nq, k, rng) for _ in 1:batch_size]

    report("countweight",
           bestof(() -> sum_countweight(a_leg)),
           bestof(() -> sum_countweight(a_mul)))
    report("commutes",
           bestof(() -> sum_commutes(a_leg, b_leg)),
           bestof(() -> sum_commutes(a_mul, b_mul)))
    report("pauliprod",
           bestof(() -> sum_pauliprod(a_leg, b_leg)),
           bestof(() -> sum_pauliprod(a_mul, b_mul)))
end

# ---- End-to-end propagate ---------------------------------------------

function observable(::Type{T}, nq::Int) where {T}
    # Single Z on the middle qubit (encoding 2 = Z).
    term = setpauli(zero(T), 2, cld(nq, 2))
    return PauliString(nq, term, 1.0)
end

function propagatebench(; nq::Int=40, nl::Int=4, max_weight::Int=6)
    T_legacy = getinttype(nq; use_multiuint=false)
    T_multi  = getinttype(nq; use_multiuint=true)
    @printf("\n[propagate] nq=%d nl=%d max_weight=%d  (legacy %s vs %s)\n",
            nq, nl, max_weight, T_legacy, T_multi)

    topo = bricklayertopology(nq; periodic=false)
    circ = hardwareefficientcircuit(nq, nl; topology=topo)
    Random.seed!(42)
    thetas = randn(countparameters(circ))

    run(::Type{T}) where {T} =
        propagate(circ, observable(T, nq), thetas; max_weight=max_weight, min_abs_coeff=0.0)

    # Sanity: both backings must agree on the result.
    v_legacy = overlapwithzero(run(T_legacy))
    v_multi  = overlapwithzero(run(T_multi))
    @printf("  result legacy=%.10f  multiuint=%.10f  agree=%s\n",
            v_legacy, v_multi, isapprox(v_legacy, v_multi; atol=1e-10))

    report("propagate",
           bestof(() -> run(T_legacy); n=3),
           bestof(() -> run(T_multi); n=3))
end

# ---- GPU section (MultiUInt only — BitIntegers can't run on device) ---

function gpubench(; nq::Int=256, batch_size::Int=100_000, k::Int=8)
    if !_CUDA_OK
        @info "CUDA not functional — skipping GPU benchmark."
        return
    end

    T = getinttype(nq; use_multiuint=true)
    @printf("\n[gpu] nq=%d  type=%s  batch=%d  device=%s\n",
            nq, T, batch_size, string(CUDA.device()))

    rng = MersenneTwister(7)
    a = [randpauli(T, nq, k, rng) for _ in 1:batch_size]
    b = [randpauli(T, nq, k, rng) for _ in 1:batch_size]
    a_gpu = CUDA.CuArray(a)
    b_gpu = CUDA.CuArray(b)

    sync(f) = (r = f(); CUDA.synchronize(); r)
    t_cw  = bestof(() -> sync(() -> countweight.(a_gpu)))
    t_cm  = bestof(() -> sync(() -> commutes.(a_gpu, b_gpu)))
    t_xor = bestof(() -> sync(() -> a_gpu .⊻ b_gpu))
    @printf("  countweight %s   commutes %s   xor-product %s\n",
            fmt_time(t_cw), fmt_time(t_cm), fmt_time(t_xor))
    @printf("  (%.1f M countweight/s)\n", batch_size / t_cw / 1e6)
end

function run_benchmarks()
    println("="^72)
    println("MultiUInt vs BitIntegers — issue #145 benchmark")
    println("="^72)
    microbench(64)
    microbench(128)
    propagatebench(nq=40, nl=4, max_weight=6)
    gpubench(nq=256)
    println("\nDone.")
end

run_benchmarks()
