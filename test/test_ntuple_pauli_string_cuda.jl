# test/test_ntuple_pauli_string_cuda.jl
#
# End-to-end GPU integration tests for NTupleUInt and NTupleUInt.
#
# Run on a CUDA-capable machine with:
#
#   julia --project=. test/test_ntuple_pauli_string_cuda.jl
#
# Or via the JuliaGPU Buildkite pipeline (.buildkite/pipeline.yml).
#
# Tests verify:
#   1. NTupleUInt arrays move to GPU via cu() and round-trip correctly.
#   2. NTupleUInt arrays move to GPU via cu() and round-trip correctly.
#   3. Pauli accessors (getpauli, countweight, commutes) give identical
#      results on CPU and after a GPU round-trip — for both types.
#   4. The 256-qubit smoke test passes for both types on GPU.

using Test
using Random

let dir = joinpath(@__DIR__, "..")
    dir ∉ LOAD_PATH && push!(LOAD_PATH, dir)
end

using PauliPropagation

# Load CUDA — skip gracefully if not installed.
const _cuda_available = !isnothing(Base.find_package("CUDA"))
if _cuda_available
    import CUDA: CUDA, cu, CuArray
end

if !_cuda_available
    @warn "CUDA.jl not installed — skipping GPU tests."
else

import PauliPropagation: _setpaulibits, _getpaulibits, _bitcommutes, _countbitweight

pauli_to_bits = Dict(:I => 0, :X => 1, :Y => 2, :Z => 3)

function cpu_encode(::Type{T}, paulis::Vector{Symbol}) where {T}
    pstr = zero(T)
    for (i, p) in enumerate(paulis)
        pstr = _setpaulibits(pstr, pauli_to_bits[p], i)
    end
    return pstr
end

if !CUDA.functional()
    @warn "CUDA not functional — skipping GPU integration tests."
    exit(0)
end

println("CUDA device: ", CUDA.name(CUDA.device()))
println("CUDA runtime version: ", CUDA.runtime_version())


# ── NTupleUInt GPU tests (existing) ──────────────────────────────────────────

@testset "NTupleUInt CUDA integration" begin

    @testset "64-qubit VectorPauliSum GPU round-trip (UInt32 words)" begin
        nq = 64
        TT = getchunkedinttype(nq; word=UInt32)
        vpsum = VectorPauliSum(Float64, nq, TT)

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

        cu_vpsum = cu(vpsum)
        @test cu_vpsum isa VectorPauliSum
        @test nqubits(cu_vpsum) == nq
        @test length(paulis(cu_vpsum)) == length(test_paulis)

        cpu_vpsum = collect(cu_vpsum)
        @test paulis(cpu_vpsum) == paulis(vpsum)
        @test coefficients(cpu_vpsum) ≈ coefficients(vpsum)
    end

    @testset "256-qubit VectorPauliSum GPU round-trip (UInt32 words)" begin
        nq = 256
        TT = getchunkedinttype(nq; word=UInt32)
        vpsum = VectorPauliSum(Float64, nq, TT)

        id_term  = zero(TT)
        far_term = cpu_encode(TT, vcat([:X], fill(:I, nq - 2), [:Z]))

        push!(paulis(vpsum), id_term);  push!(coefficients(vpsum), 1.0)
        push!(paulis(vpsum), far_term); push!(coefficients(vpsum), -1.0)

        cpu_vpsum = collect(cu(vpsum))
        @test paulis(cpu_vpsum)[1] == id_term
        @test paulis(cpu_vpsum)[2] == far_term
    end

    @testset "Pauli accessor consistency after GPU round-trip" begin
        nq = 128
        TT = getchunkedinttype(nq; word=UInt32)
        vpsum = VectorPauliSum(Float64, nq, TT)

        pauli_seq = [:X, :Y, :Z, :X, :Z]
        qinds_seq = [1, 10, 50, 100, 128]
        term = symboltoint(TT, pauli_seq, qinds_seq)

        push!(paulis(vpsum), term)
        push!(coefficients(vpsum), 1.0)

        cpu_term  = paulis(vpsum)[1]
        rt_term   = paulis(collect(cu(vpsum)))[1]

        @test rt_term == cpu_term
        for (sym, qind) in zip(pauli_seq, qinds_seq)
            @test getpauli(rt_term, qind) == getpauli(cpu_term, qind)
        end
        @test countweight(rt_term) == 5
    end

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

        @test _bitcommutes(pX_all, pZ_all) == _bitcommutes(rt_X, rt_Z)
    end

    @testset "128-qubit VectorPauliSum GPU round-trip (UInt64 words)" begin
        nq = 128
        TT = getchunkedinttype(nq; word=UInt64)
        vpsum = VectorPauliSum(Float64, nq, TT)

        term = symboltoint(TT, [:X, :Y, :Z], [1, 64, 128])
        push!(paulis(vpsum), term)
        push!(coefficients(vpsum), 2.5)

        cpu_vpsum = collect(cu(vpsum))
        @test paulis(cpu_vpsum)[1] == term
        @test coefficients(cpu_vpsum)[1] ≈ 2.5
    end

end


# ── NTupleUInt GPU tests ───────────────────────────────────────────────────────
# NTupleUInt <: Unsigned so getinttype automatically returns it for >32 qubits.
# No opt-in or separate API needed.

@testset "NTupleUInt CUDA integration" begin

    @testset "isbitstype on device" begin
        @test isbitstype(NTupleUInt{8, UInt64})
        @test isbitstype(NTupleUInt{16, UInt32})
        @test getinttype(256)              === NTupleUInt{8,  UInt64}
        @test getinttype(256; word=UInt32) === NTupleUInt{16, UInt32}
    end

    @testset "256-qubit VectorPauliSum GPU round-trip (UInt64 words)" begin
        nq  = 256
        TM  = getinttype(nq)   # NTupleUInt{8, UInt64} — automatic, no opt-in
        vps = VectorPauliSum(Float64, nq, TM)

        id_term  = zero(TM)
        far_term = let p = zero(TM)
            p = setpauli(p, 1, 1)
            setpauli(p, 2, nq)
        end

        push!(paulis(vps), id_term);  push!(coefficients(vps), 1.0)
        push!(paulis(vps), far_term); push!(coefficients(vps), -1.0)

        rt = collect(cu(vps))
        @test paulis(rt)[1] == id_term
        @test paulis(rt)[2] == far_term
        @test coefficients(rt) ≈ [1.0, -1.0]
    end

    @testset "256-qubit VectorPauliSum GPU round-trip (UInt32 words)" begin
        nq  = 256
        TM  = getinttype(nq; word=UInt32)   # NTupleUInt{16, UInt32}
        vps = VectorPauliSum(Float64, nq, TM)

        far_term = let p = zero(TM)
            p = setpauli(p, 1, 1)
            setpauli(p, 3, nq)
        end
        push!(paulis(vps), far_term); push!(coefficients(vps), 0.5)

        rt = collect(cu(vps))
        @test paulis(rt)[1] == far_term
        @test coefficients(rt)[1] ≈ 0.5
    end

    @testset "Broadcast kernels match CPU (countweight, commutes, xor)" begin
        nq = 256
        TM = getinttype(nq)
        Random.seed!(2026)
        batch = 512

        pa = [zero(TM) for _ in 1:batch]
        pb = [zero(TM) for _ in 1:batch]
        for k in 1:batch
            for _ in 1:5
                pa[k] = setpauli(pa[k], rand(0:3), rand(1:nq))
                pb[k] = setpauli(pb[k], rand(0:3), rand(1:nq))
            end
        end

        weights_cpu = Int[countweight(p) for p in pa]
        commute_cpu = Bool[commutes(pa[k], pb[k]) for k in 1:batch]
        xor_cpu     = TM[pa[k] ⊻ pb[k] for k in 1:batch]

        gpu_a = CuArray(pa)
        gpu_b = CuArray(pb)
        @test isbitstype(eltype(gpu_a))

        @test Array(countweight.(gpu_a)) == weights_cpu
        @test Array(commutes.(gpu_a, gpu_b)) == commute_cpu
        @test Array(gpu_a .⊻ gpu_b) == xor_cpu
    end

    @testset "propagate() on GPU VectorPauliSum (64 qubits)" begin
        # 64 qubits → NTupleUInt{2, UInt64}: small enough for CI, exercises full GPU path.
        nq = 64
        TM = getinttype(nq)
        @test TM <: NTupleUInt
        @test isbitstype(TM)

        Random.seed!(42)
        topo   = bricklayertopology(nq; periodic=false)
        circ   = hardwareefficientcircuit(nq, 2; topology=topo)
        thetas = randn(length(circ))

        # CPU reference via VectorPauliSum
        vps_cpu = VectorPauliSum(Float64, nq, TM)
        push!(paulis(vps_cpu), symboltoint(TM, [:Z], [div(nq, 2)]))
        push!(coefficients(vps_cpu), 1.0)
        cpu_result = overlapwithzero(propagate(circ, vps_cpu, thetas))

        # GPU path: same data moved to device, propagate, collect back
        gpu_result = overlapwithzero(collect(propagate(circ, cu(vps_cpu), thetas)))

        # GPU operates in Float32; allow Float32 precision tolerance (~1e-6)
        @test cpu_result ≈ gpu_result rtol=1e-5
        @info "GPU propagate() 64q passed" device=string(CUDA.device())
    end

end

end