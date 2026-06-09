# benchmarks/benchmark_ntuple.jl
#
# CPU benchmark: NTupleUInt{N,UInt64} vs BitIntegers-backed types.
#
# Covers the operations requested by the reviewer:
#   propagate(), countweight(), pauliprod(), and sorting.
#
# Run with:
#   julia --project=. benchmarks/benchmark_ntuple.jl

let dir = joinpath(@__DIR__, "..")
    dir ∉ LOAD_PATH && push!(LOAD_PATH, dir)
end

using PauliPropagation
using Random
using Printf

# ── helpers ────────────────────────────────────────────────────────────────────

function make_random_pstrs(::Type{T}, n; nq) where {T}
    [begin
        p = zero(T)
        for _ in 1:8
            p = setpauli(p, rand(0:3), rand(1:nq))
        end
        p
    end for _ in 1:n]
end

function bench_section(title)
    println()
    println("=" ^ 60)
    println("  $title")
    println("=" ^ 60)
end

"""
    bench(f; samples=100, evals=1)

Minimal `@btime` replacement that needs no external package.
Runs `f()` once to warm up, then `samples` times and reports the
minimum elapsed wall time (matching BenchmarkTools' default metric).
"""
function bench(f; samples::Int=100, evals::Int=1)
    # warm-up
    f()
    GC.gc()

    best_ns = typemax(Int64)
    for _ in 1:samples
        t0 = time_ns()
        for _ in 1:evals
            f()
        end
        t1 = time_ns()
        dt = (t1 - t0) ÷ evals
        best_ns = min(best_ns, Int64(dt))
    end

    # format like BenchmarkTools: pick the right unit
    if best_ns < 1_000
        @printf("  %d ns\n", best_ns)
    elseif best_ns < 1_000_000
        @printf("  %.3f μs\n", best_ns / 1e3)
    elseif best_ns < 1_000_000_000
        @printf("  %.3f ms\n", best_ns / 1e6)
    else
        @printf("  %.3f s\n",  best_ns / 1e9)
    end
end

# ── 1. propagate() ─────────────────────────────────────────────────────────────

bench_section("propagate()  —  64 qubits, 2-layer hardware-efficient circuit")

let nq = 64
    Random.seed!(1)
    topo   = bricklayertopology(nq; periodic=false)
    circ   = hardwareefficientcircuit(nq, 2; topology=topo)
    thetas = randn(length(circ))

    T_bi = getinttype(nq)                    # UInt128 (BitIntegers)
    T_ch = getchunkedinttype(nq)             # NTupleUInt{2,UInt64}

    make_obs(T) = (o = VectorPauliSum(nq, T[], Float64[]); push!(paulis(o), symboltoint(T, [:Z], [div(nq, 2)])); push!(coefficients(o), 1.0); o)

    println("\nBitIntegers  (UInt128):")
    bench(() -> propagate(circ, make_obs(T_bi), thetas))

    println("NTupleUInt{2,UInt64}:")
    bench(() -> propagate(circ, make_obs(T_ch), thetas))
end

bench_section("propagate()  —  128 qubits, 2-layer hardware-efficient circuit")

let nq = 128
    Random.seed!(2)
    topo   = bricklayertopology(nq; periodic=false)
    circ   = hardwareefficientcircuit(nq, 2; topology=topo)
    thetas = randn(length(circ))

    T_bi = getinttype(nq)                    # BitIntegers UInt256
    T_ch = getchunkedinttype(nq)             # NTupleUInt{4,UInt64}

    make_obs(T) = (o = VectorPauliSum(nq, T[], Float64[]); push!(paulis(o), symboltoint(T, [:Z], [div(nq, 2)])); push!(coefficients(o), 1.0); o)

    println("\nBitIntegers  (UInt256):")
    bench(() -> propagate(circ, make_obs(T_bi), thetas))

    println("NTupleUInt{4,UInt64}:")
    bench(() -> propagate(circ, make_obs(T_ch), thetas))
end

# ── 2. countweight() ──────────────────────────────────────────────────────────

bench_section("countweight()  —  10 000 random Pauli strings")

let nq = 128, n = 10_000
    T_bi = getinttype(nq)
    T_ch = getchunkedinttype(nq)

    Random.seed!(3)
    ps_bi = make_random_pstrs(T_bi, n; nq=nq)
    ps_ch = make_random_pstrs(T_ch, n; nq=nq)

    println("\nBitIntegers  (UInt256):")
    bench(() -> sum(countweight(p) for p in ps_bi))

    println("NTupleUInt{4,UInt64}:")
    bench(() -> sum(countweight(p) for p in ps_ch))
end

# ── 3. pauliprod() ────────────────────────────────────────────────────────────

bench_section("pauliprod()  —  10 000 random pairs")

let nq = 128, n = 10_000
    T_bi = getinttype(nq)
    T_ch = getchunkedinttype(nq)

    Random.seed!(4)
    as_bi = make_random_pstrs(T_bi, n; nq=nq)
    bs_bi = make_random_pstrs(T_bi, n; nq=nq)
    as_ch = make_random_pstrs(T_ch, n; nq=nq)
    bs_ch = make_random_pstrs(T_ch, n; nq=nq)

    println("\nBitIntegers  (UInt256):")
    bench(() -> begin
        for i in eachindex(as_bi)
            pauliprod(as_bi[i], bs_bi[i])
        end
    end)

    println("NTupleUInt{4,UInt64}:")
    bench(() -> begin
        for i in eachindex(as_ch)
            pauliprod(as_ch[i], bs_ch[i])
        end
    end)
end

# ── 4. sorting ────────────────────────────────────────────────────────────────

bench_section("sort()  —  10 000 random Pauli strings")

let nq = 128, n = 10_000
    T_bi = getinttype(nq)
    T_ch = getchunkedinttype(nq)

    Random.seed!(5)
    ps_bi = make_random_pstrs(T_bi, n; nq=nq)
    ps_ch = make_random_pstrs(T_ch, n; nq=nq)

    println("\nBitIntegers  (UInt256):")
    bench(() -> sort(ps_bi))

    println("NTupleUInt{4,UInt64}:")
    bench(() -> sort(ps_ch))
end

println()
println("Done.")