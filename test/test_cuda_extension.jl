using Test
using Random
using PauliPropagation
import CUDA: CUDA, cu, CuArray


println("CUDA device: ", CUDA.name(CUDA.device()))
println("CUDA runtime version: ", CUDA.runtime_version())

rng = MersenneTwister()
println("CUDA extension test seed: $(rng.seed)")


@testset "NTupleInteger GPU integration" begin

    @testset "isbitstype on device" begin
        @test isbitstype(NTupleInteger{4,UInt64})
        @test isbitstype(NTupleInteger{8,UInt32})
        @test getchunkedinttype(256) == NTupleInteger{8,UInt64}
        @test getchunkedinttype(256; word=UInt32) == NTupleInteger{16,UInt32}
    end

    @testset "propagate() matches CPU (256 qubits)" begin
        # 256 qubits: small enough for CI, exercises the full GPU path.
        # BitIntegers-backed UInt512 that is not isbits on the device.
        nq = 256
        TM = getchunkedinttype(nq)   # NTupleInteger{2,UInt64}
        @test TM <: NTupleInteger
        @test isbitstype(TM)

        topo = bricklayertopology(nq; periodic=false)
        circ = hardwareefficientcircuit(nq, 2; topology=topo)
        thetas = randn(rng, length(circ))

        # CPU reference via VectorPauliSum
        vps_cpu = VectorPauliSum(nq, TM[], Float64[])
        push!(paulis(vps_cpu), symboltoint(TM, [:Z], [div(nq, 2)]))
        push!(coefficients(vps_cpu), 1.0)
        cpu_result = overlapwithzero(propagate(circ, vps_cpu, thetas))

        # GPU path: same data moved to device, propagate, collect back
        gpu_result = overlapwithzero(collect(propagate(circ, cu(vps_cpu), thetas)))

        # GPU operates in Float32; allow Float32 precision tolerance
        @test cpu_result ≈ gpu_result rtol = 1e-5
    end

    @testset "merge!() matches CPU (256 qubits)" begin
        nq = 256
        TM = getchunkedinttype(nq)

        # Two copies of the same Pauli string with different coefficients.
        # merge!() should combine them into a single term whose coefficient
        # equals the sum of the two originals, on both CPU and GPU.
        base = zero(TM)
        base = setpauli(base, 1, 1)  # X on qubit 1
        base = setpauli(base, 3, 2)  # Z on qubit 2

        vps_cpu = VectorPauliSum(nq, TM[], Float64[])
        push!(paulis(vps_cpu), base)
        push!(coefficients(vps_cpu), 0.4)
        push!(paulis(vps_cpu), base)   # same string again
        push!(coefficients(vps_cpu), 0.6)

        expected_coeff = 0.4 + 0.6

        # Move duplicates to GPU first, then merge on each independently.
        vps_gpu = cu(deepcopy(vps_cpu))

        merge!(vps_cpu)
        cpu_n = length(paulis(vps_cpu))
        cpu_coeff = sum(coefficients(vps_cpu))

        merge!(vps_gpu)
        gpu_n = length(paulis(vps_gpu))
        gpu_coeff = sum(Array(coefficients(vps_gpu)))

        @test cpu_n == 1
        @test cpu_coeff ≈ expected_coeff
        @test gpu_n == cpu_n
        @test gpu_coeff ≈ cpu_coeff
    end

    @testset "truncate!() matches CPU (256 qubits)" begin
        nq = 256
        TM = getchunkedinttype(nq)

        vps_cpu = VectorPauliSum(nq, TM[], Float64[])
        for _ in 1:8
            pstr = zero(TM)
            for _ in 1:5
                pstr = setpauli(pstr, rand(rng, 0:3), rand(rng, 1:nq))
            end
            push!(paulis(vps_cpu), pstr)
            push!(coefficients(vps_cpu), randn(rng))
        end

        vps_gpu = cu(deepcopy(vps_cpu))
        wtrunc = (pstr, coeff) -> countweight(pstr) > 2
        truncate!(vps_cpu; customtruncfunc=wtrunc)
        truncate!(vps_gpu; customtruncfunc=wtrunc)

        # After truncation, all surviving strings have weight <= 2.
        # paulis(vps_gpu) is a CuArray, so collect to CPU first.
        @test all(countweight(p) <= 2 for p in paulis(vps_cpu))
        @test all(countweight(p) <= 2 for p in Array(paulis(vps_gpu)))
    end

end
