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
using BenchmarkTools
using Random

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

# ── 1. propagate() ─────────────────────────────────────────────────────────────

bench_section("propagate()  —  64 qubits, 2-layer hardware-efficient circuit")

let nq = 64
    Random.seed!(1)
    topo   = bricklayertopology(nq; periodic=false)
    circ   = hardwareefficientcircuit(nq, 2; topology=topo)
    thetas = randn(length(circ))

    T_bi = getinttype(nq)                    # UInt128 (BitIntegers)
    T_ch = getchunkedinttype(nq)             # NTupleUInt{2,UInt64}

    obs_bi = VectorPauliSum(nq, T_bi[], Float64[])
    push!(paulis(obs_bi), symboltoint(T_bi, [:Z], [div(nq, 2)]))
    push!(coefficients(obs_bi), 1.0)

    obs_ch = VectorPauliSum(nq, T_ch[], Float64[])
    push!(paulis(obs_ch), symboltoint(T_ch, [:Z], [div(nq, 2)]))
    push!(coefficients(obs_ch), 1.0)

    println("\nBitIntegers  (UInt128):")
    @btime propagate($circ, $obs_bi, $thetas)

    println("NTupleUInt{2,UInt64}:")
    @btime propagate($circ, $obs_ch, $thetas)
end

bench_section("propagate()  —  128 qubits, 2-layer hardware-efficient circuit")

let nq = 128
    Random.seed!(2)
    topo   = bricklayertopology(nq; periodic=false)
    circ   = hardwareefficientcircuit(nq, 2; topology=topo)
    thetas = randn(length(circ))

    T_bi = getinttype(nq)                    # BitIntegers UInt256
    T_ch = getchunkedinttype(nq)             # NTupleUInt{4,UInt64}

    obs_bi = VectorPauliSum(nq, T_bi[], Float64[])
    push!(paulis(obs_bi), symboltoint(T_bi, [:Z], [div(nq, 2)]))
    push!(coefficients(obs_bi), 1.0)

    obs_ch = VectorPauliSum(nq, T_ch[], Float64[])
    push!(paulis(obs_ch), symboltoint(T_ch, [:Z], [div(nq, 2)]))
    push!(coefficients(obs_ch), 1.0)

    println("\nBitIntegers  (UInt256):")
    @btime propagate($circ, $obs_bi, $thetas)

    println("NTupleUInt{4,UInt64}:")
    @btime propagate($circ, $obs_ch, $thetas)
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
    @btime sum(countweight(p) for p in $ps_bi)

    println("NTupleUInt{4,UInt64}:")
    @btime sum(countweight(p) for p in $ps_ch)
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
    @btime for i in eachindex($as_bi)
        pauliprod($as_bi[i], $bs_bi[i])
    end

    println("NTupleUInt{4,UInt64}:")
    @btime for i in eachindex($as_ch)
        pauliprod($as_ch[i], $bs_ch[i])
    end
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
    @btime sort($ps_bi)

    println("NTupleUInt{4,UInt64}:")
    @btime sort($ps_ch)
end

println()
println("Done.")