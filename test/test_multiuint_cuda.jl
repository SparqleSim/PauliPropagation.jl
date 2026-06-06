# GPU correctness test for MultiUInt-backed Pauli ops at 256 qubits.
#
# This is the partial-reward target of issue #145: prove that MultiUInt
# (a) lives in GPU registers (`isbitstype`, fixed shape) and
# (b) supports the bitwise contract PauliPropagation's CUDA kernels need.
#
# We test the bit-level operations that don't depend on any CPU-side
# global state: countweight, commutes, getpauli, and the XOR product
# `a ⊻ b` (the unsigned Pauli multiplication). The signed `pauliprod`
# uses a CPU-side `impowers` Vector lookup and is intentionally out of
# scope here; lifting that to GPU is a separate, downstream change.
#
# Skipped automatically when CUDA isn't functional, so the test file is
# safe to run in CI environments without a GPU.

using Test

@testset "MultiUInt on CUDA — 256-qubit Pauli ops" begin
    try
        @eval using CUDA
    catch
        @info "CUDA.jl unavailable — skipping GPU test."
        return
    end
    if !CUDA.functional()
        @info "CUDA not functional on this host — skipping GPU test."
        return
    end

    using PauliPropagation
    using PauliPropagation: MultiUInt

    nq = 256
    T = getinttype(nq)
    @test T === MultiUInt{8, UInt64}
    @test isbitstype(T)

    # Build a batch of random Pauli strings on a 256-qubit register.
    using Random
    Random.seed!(2026)
    batch_size = 1024
    paulis_a = T[zero(T) for _ in 1:batch_size]
    paulis_b = T[zero(T) for _ in 1:batch_size]
    for k in 1:batch_size
        for _ in 1:5
            paulis_a[k] = setpauli(paulis_a[k], rand(0:3), rand(1:nq))
            paulis_b[k] = setpauli(paulis_b[k], rand(0:3), rand(1:nq))
        end
    end

    # CPU reference results.
    weights_cpu = Int[countweight(p) for p in paulis_a]
    commute_cpu = Bool[commutes(paulis_a[k], paulis_b[k]) for k in 1:batch_size]
    xor_cpu = T[paulis_a[k] ⊻ paulis_b[k] for k in 1:batch_size]   # unsigned Pauli product

    # Move to GPU.
    paulis_a_gpu = CuArray(paulis_a)
    paulis_b_gpu = CuArray(paulis_b)
    @test isbitstype(eltype(paulis_a_gpu))

    # Each broadcast compiles into a CUDA kernel that operates on the
    # MultiUInt elements in registers. If MultiUInt weren't truly bits-
    # like on the device, these would either fail to compile or hit
    # ILLEGAL_ADDRESS at runtime.
    weights_gpu = Array(countweight.(paulis_a_gpu))
    commute_gpu = Array(commutes.(paulis_a_gpu, paulis_b_gpu))
    xor_gpu = Array(paulis_a_gpu .⊻ paulis_b_gpu)

    @test weights_gpu == weights_cpu
    @test commute_gpu == commute_cpu
    @test xor_gpu == xor_cpu

    @info "GPU 256-qubit test passed" device=string(CUDA.device()) batch=batch_size
end
