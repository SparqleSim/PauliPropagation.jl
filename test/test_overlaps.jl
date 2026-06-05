# Test File for stateoverlap.jl
# 
# This file contains unit tests for the following functionalities:
# - Edge cases for overlapwithcomputational.
# - TODO(YT): add tests for other functions in stateoverlap.jl

using Test

const test_cases = [
    "twoZs" => (4, (2, 4), 1.0),
    "oneZ" => (5, (2), -1.0),
    "noZ" => (4, (3), 1.0),
    "noones" => (4, (), 1.0),
]

@testset "Parameterized Tests stateoverlaps" begin

    for (name, (nq, indices, expected)) in test_cases
        @testset "$name" begin

            # SubTest the case where the PauliString is the identity
            pstr = PauliString(nq, :I, nq)
            @test overlapwithcomputational(pstr, indices) == 1.0

        end

        @testset "$name" begin

            # SubTest the case where the PauliString is a Z operator
            pstr = PauliString(nq, [:Z, :Z], [2, 4])
            @test overlapwithcomputational(pstr, indices) == expected

        end

        @testset "$name" begin

            # SubTest the case where the PauliString contains an X/Y operator
            pstr = PauliString(nq, [:X, :Z, :Z], [1, 2, 4])
            @test overlapwithcomputational(pstr, indices) == 0.0

            pstr = PauliString(nq, [:Y], [4])
            @test overlapwithcomputational(pstr, indices) == 0.0

        end
    end

end

@testset "Test scalarproduct" begin

    # Test the scalar product of two Pauli strings
    pstr1 = PauliString(4, [:I, :Z], [1, 2], 1.0)
    pstr2 = PauliString(4, [:X, :Y], [2, 3], -2.0)
    pstr3 = PauliString(4, [:Z, :X], [3, 4], 0.2)

    @test scalarproduct(pstr1, pstr2) == 0.0
    @test scalarproduct(pstr1, pstr3) == 0.0
    @test scalarproduct(pstr2, pstr3) == 0.0

    orig_psum = PauliSum([pstr1, pstr2])
    vector_psum = VectorPauliSum(orig_psum)
    wrapped_psum = wrapcoefficients(orig_psum, PauliFreqTracker)
    for psum in [orig_psum, wrapped_psum, vector_psum]
        @test scalarproduct(psum, pstr3) == scalarproduct(pstr3, psum) == 0.0
        @test scalarproduct(psum, pstr1) == scalarproduct(pstr1, psum) == 1.0
        @test scalarproduct(psum, pstr2) == scalarproduct(pstr2, psum) == 4.0
    end

    for psum1 in [orig_psum, wrapped_psum, vector_psum]
        for psum2 in [orig_psum, wrapped_psum, vector_psum]
            @test scalarproduct(psum1, psum2) == 5.0
        end
    end
end


@testset "Test overlapwithpaulisum" begin

    nq = 3

    psum = PauliSum(nq)
    add!(psum, :I, 1, 0.2)
    add!(psum, :X, 2, 0.3)
    add!(psum, :Z, 3, 0.4)
    add!(psum, [:Z, :Z], [1, 2], 0.5)

    vecpsum = VectorPauliSum(psum)

    rho = PauliSum(PauliString(3, :I, 1, 1 / 2^nq))
    vecrho = VectorPauliSum(rho)
    @test overlapwithpaulisum(rho, psum) == overlapwithmaxmixed(psum) == 0.2
    @test overlapwithpaulisum(rho, vecpsum) == overlapwithmaxmixed(vecpsum) == 0.2
    @test overlapwithpaulisum(vecrho, psum) == overlapwithpaulisum(vecrho, vecpsum) == 0.2


    @test overlapwithplus(rho) == 1 / 2^nq
    @test overlapwithplus(vecrho) == 1 / 2^nq

    which = rand(0:1, nq)
    rho = prod(PauliSum([PauliString(nq, :I, qind, 1 / 2), PauliString(nq, :Z, qind, which[qind] == 0 ? 1 / 2 : -1 / 2)]) for qind in 1:nq)
    @test overlapwithpaulisum(rho, psum) == overlapwithcomputational(psum, findall(which .== 1))

    rho = prod(PauliSum([PauliString(nq, :I, qind, 1 / 2), PauliString(nq, :Z, qind, 1 / 2)]) for qind in 1:nq)
    @test overlapwithpaulisum(rho, psum) == overlapwithzero(psum) == 1.1





end