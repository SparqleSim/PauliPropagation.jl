# test/test_ntuple_pauli_string_cuda.jl
#
# End-to-end GPU integration tests for NTupleUInt.
#
# Run on a CUDA-capable machine with:
#
#   julia --project=benchmarks test/test_ntuple_pauli_string_cuda.jl
#
# First-time setup (once):
#
#   julia --project=benchmarks -e '
#       using Pkg
#       Pkg.develop(PackageSpec(path="."))
#       Pkg.instantiate()
#   '
#
# Or via the JuliaGPU Buildkite pipeline (.buildkite/pipeline.yml).
#
# These tests verify that:
#   1. NTupleUInt arrays are correctly moved to the GPU via cu()
#   2. Round-trip collect() reproduces the original CPU data exactly
#   3. Pauli accessor functions (getpauli, countweight, commutes) give
#      identical results whether evaluated on CPU or after a GPU round-trip
#   4. The 256-qubit smoke test passes on GPU (the primary motivating case)
#
# The tests deliberately avoid calling CUDA kernels directly: correctness of
# the on-device compute paths will be validated separately once GPU propagation
# benchmarks are added.  What we confirm here is that the data transport layer
# (CuArray + NTupleUInt bits-are-plain-bits property) is sound.

using Test

# Make sure the local checkout takes priority over any installed version.
let dir = joinpath(@__DIR__, "..")
    dir ∉ LOAD_PATH && push!(LOAD_PATH, dir)
end

using PauliPropagation
using CUDA

import PauliPropagation: _setpaulibits, _getpaulibits, _bitcommutes, _countbitweight

# ── helpers ──────────────────────────────────────────────────────────────────

pauli_to_bits = Dict(:I => 0, :X => 1, :Y => 2, :Z => 3)

function cpu_encode(::Type{T}, paulis::Vector{Symbol}) where {T<:NTupleUInt}
    pstr = zero(T)
    for (i, p) in enumerate(paulis)
        pstr = _setpaulibits(pstr, pauli_to_bits[p], i)
    end
    return pstr
end

# ── top-level CUDA guard ──────────────────────────────────────────────────────

if !CUDA.functional()
    @warn "CUDA not functional — skipping GPU integration tests."
    exit(0)
end

println("CUDA device: ", CUDA.name(CUDA.device()))
println("CUDA runtime version: ", CUDA.runtime_version())

# ── tests ─────────────────────────────────────────────────────────────────────

@testset "NTupleUInt CUDA integration" begin

    # ── 64-qubit VectorPauliSum round-trip ───────────────────────────────────
    @testset "64-qubit VectorPauliSum GPU round-trip (UInt32 words)" begin
        nq = 64
        TT = getchunkedinttype(nq; word=UInt32)   # NTupleUInt{4, UInt32}
        vpsum = VectorPauliSum(Float64, nq, TT)

        # Build a handful of terms on CPU.
        test_paulis = [
            ([:X, :Z], [1, 3]),
            ([:Y, :Y], [2, 4]),
            ([:X, :X, :X], [1, 32, 64]),
            ([:Z, :Z, :Z, :Z], [1, 16, 32, 64]),
        ]
        for (syms, qinds) in test_paulis
            push!(paulis(vpsum), symboltoint(TT, syms, qinds))
            push!(coefficients(vpsum), randn())
        end

        # Move to GPU.
        cu_vpsum = cu(vpsum)
        @test cu_vpsum isa VectorPauliSum
        @test nqubits(cu_vpsum) == nq
        @test length(paulis(cu_vpsum)) == length(test_paulis)

        # Round-trip back to CPU.
        cpu_vpsum = collect(cu_vpsum)
        @test paulis(cpu_vpsum) == paulis(vpsum)
        @test coefficients(cpu_vpsum) ≈ coefficients(vpsum)
    end

    # ── 256-qubit smoke test (the primary motivating use case) ───────────────
    @testset "256-qubit VectorPauliSum GPU round-trip (UInt32 words)" begin
        nq = 256
        TT = getchunkedinttype(nq; word=UInt32)   # NTupleUInt{16, UInt32}
        vpsum = VectorPauliSum(Float64, nq, TT)

        # Identity string and a string with Paulis at the boundary qubits.
        id_term  = zero(TT)
        far_term = cpu_encode(TT, vcat([:X], fill(:I, nq - 2), [:Z]))  # q1=X, q256=Z

        push!(paulis(vpsum), id_term);  push!(coefficients(vpsum), 1.0)
        push!(paulis(vpsum), far_term); push!(coefficients(vpsum), -1.0)

        cu_vpsum = cu(vpsum)
        @test length(paulis(cu_vpsum)) == 2

        cpu_vpsum = collect(cu_vpsum)
        @test paulis(cpu_vpsum)[1] == id_term
        @test paulis(cpu_vpsum)[2] == far_term
    end

    # ── getpauli / countweight consistency after GPU round-trip ──────────────
    @testset "Pauli accessor consistency after GPU round-trip" begin
        nq = 128
        TT = getchunkedinttype(nq; word=UInt32)   # NTupleUInt{8, UInt32}
        vpsum = VectorPauliSum(Float64, nq, TT)

        pauli_seq = [:X, :Y, :Z, :X, :Z]
        qinds_seq = [1, 10, 50, 100, 128]
        term = symboltoint(TT, pauli_seq, qinds_seq)

        push!(paulis(vpsum), term)
        push!(coefficients(vpsum), 1.0)

        cpu_term  = paulis(vpsum)[1]
        cpu_vpsum = collect(cu(vpsum))
        rt_term   = paulis(cpu_vpsum)[1]

        # The round-tripped bits must be identical.
        @test rt_term == cpu_term

        # Pauli accessors must agree.
        for (sym, qind) in zip(pauli_seq, qinds_seq)
            @test getpauli(rt_term, qind) == getpauli(cpu_term, qind)
        end

        # Weight should be 5.
        @test countweight(rt_term) == 5
        @test countweight(rt_term) == countweight(cpu_term)
    end

    # ── commutation check survives GPU round-trip ────────────────────────────
    @testset "Commutation result unchanged after GPU round-trip" begin
        nq = 64
        TT = getchunkedinttype(nq; word=UInt32)

        pX_all = symboltoint(TT, fill(:X, nq), collect(1:nq))
        pZ_all = symboltoint(TT, fill(:Z, nq), collect(1:nq))

        vpsum = VectorPauliSum(Float64, nq, TT)
        push!(paulis(vpsum), pX_all); push!(coefficients(vpsum), 1.0)
        push!(paulis(vpsum), pZ_all); push!(coefficients(vpsum), 1.0)

        cpu_vpsum = collect(cu(vpsum))
        rt_X = paulis(cpu_vpsum)[1]
        rt_Z = paulis(cpu_vpsum)[2]

        # X⊗n and Z⊗n anticommute iff n is odd; here n=64 (even) → commute.
        @test _bitcommutes(pX_all, pZ_all) == _bitcommutes(rt_X, rt_Z)
    end

    # ── UInt64-word variant ──────────────────────────────────────────────────
    @testset "128-qubit VectorPauliSum GPU round-trip (UInt64 words)" begin
        nq = 128
        TT = getchunkedinttype(nq; word=UInt64)   # NTupleUInt{4, UInt64}
        vpsum = VectorPauliSum(Float64, nq, TT)

        term = symboltoint(TT, [:X, :Y, :Z], [1, 64, 128])
        push!(paulis(vpsum), term)
        push!(coefficients(vpsum), 2.5)

        cpu_vpsum = collect(cu(vpsum))
        @test paulis(cpu_vpsum)[1] == term
        @test coefficients(cpu_vpsum)[1] ≈ 2.5
    end

end

println("\nAll NTupleUInt CUDA integration tests passed.")