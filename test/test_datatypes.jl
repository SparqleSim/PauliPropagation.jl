## Test the datatypes module with all possible constructors and adders

using Test

function createpaulistring(nq)
    # Create a random PauliString with nq qubits

    symbols = rand([:I, :X, :Y, :Z], nq)
    qinds = shuffle(1:nq)[1:nq]
    coeff = randn()

    return PauliString(nq, symbols, qinds, coeff)

end

function createpaulisum(nq)

    return PauliSum(createpaulistring(nq))

end

function createvectorpaulisum(nq)

    return VectorPauliSum(createpaulistring(nq))

end

@testset "Add to PauliSum" begin
    nq = 65

    for (creatorfunc, PS) in zip(
        (createpaulisum, createvectorpaulisum),
        (PauliSum, VectorPauliSum)
    )
        psum = creatorfunc(nq)
        pstr = createpaulistring(nq)
        pstr_temp = psum + pstr
        @test isa(pstr_temp, PS)
        pstr_temp = pstr + pstr
        # currently this always converts to PauliSum
        @test isa(pstr_temp, PauliSum)
        add!(psum, pstr)
        @test getcoeff(psum, pstr.term) == pstr.coeff

        symbol = rand([:I, :X, :Y, :Z])
        qind = rand(1:nq)
        coeff = randn()
        add!(psum, symbol, qind, coeff)
        @test getcoeff(psum, symbol, qind) == coeff

        symbols = rand([:I, :X, :Y, :Z], min(nq, 4))
        qinds = shuffle(1:nq)[1:min(nq, 4)]
        coeff = randn()
        psum2 = creatorfunc(nq)
        add!(psum2, symbols, qinds, coeff)
        @test getcoeff(psum2, symbols, qinds) == coeff
    end
end

@testset "PauliString Tests" begin
    nq = 7
    symbol = rand([:I, :X, :Y, :Z])
    qind = rand(1:nq)
    coeff = randn()
    pstr = PauliString(nq, symbol, qind, coeff)
    @test pstr.coeff == coeff
    @test pstr.nqubits == nq

    symbols = rand([:I, :X, :Y, :Z], min(nq, 4))
    qinds = shuffle(1:nq)[1:min(nq, 4)]
    coeff = randn()
    pstr = PauliString(nq, symbols, qinds, coeff)

    for (ii, qind) in enumerate(qinds)
        @test getpauli(pstr.term, qind) == symboltoint(symbols[ii])
    end
    @test getpauli(pstr.term, qinds) == symboltoint(symbols)
    @test paulitype(pstr) == getinttype(nq) == UInt16


    nq = 17
    psum = PauliSum(nq)
    @test length(psum) == length(psum.terms) == 0
    @test coefftype(psum) == Float64
    @test paulitype(psum) == getinttype(nq)
    @test paulitype(psum) == PauliPropagation.UInt40

    pstr = createpaulistring(7)
    wrapped_pstr = wrapcoefficients(pstr, PauliFreqTracker)
    @test coefftype(wrapped_pstr) <: PauliFreqTracker
    @test tonumber(wrapped_pstr.coeff) == tonumber(pstr.coeff) == pstr.coeff
end

@testset "PauliSum/VectorPauliSum Conversions" begin
    nq = rand(1:100)

    # Subtest for subtracting PauliSum
    @testset "PauliSum to VectorPauliSum" begin    
        pstr = createpaulistring(nq)
        psum = PauliSum(pstr)
        vpsum = VectorPauliSum(psum)
        vpsum_from_str = VectorPauliSum(pstr)
        @test vpsum == psum # checks Pauli by Pauli
        @test vpsum_from_str == psum 
        @test vpsum_from_str == vpsum

    # Subtest for subtracting PauliSum
        # VectorPauliSum to PauliSums with Merging

        pstr2 = createpaulistring(nq)
        vpsum = VectorPauliSum([pstr, pstr2, pstr])
        psum = PauliSum(vpsum)
        @test vpsum == psum
    end

end

# Test overloading methods for PauliSum
@testset "PauliSum Tests" begin

    # Subtest for subtracting PauliSum
    @testset "Substract PauliSums" begin

        psum1 = PauliSum(3)
        add!(psum1, [:I, :I, :Y], 1:3, 1.0)
        add!(psum1, :I, 1, 1.5)
        vecpsum = VectorPauliSum(psum1)
        psum2 = PauliSum(PauliString(3, :I, 1, 1.5))
        result_psum = psum1 - psum2
        result_vecpsum = vecpsum - psum2

        expected_psum = PauliSum(PauliString(3, [:I, :I, :Y], 1:3, 1.0))
        @test result_psum == expected_psum
        @test result_psum ≈ expected_psum
        @test result_vecpsum == expected_psum
        @test result_vecpsum ≈ expected_psum

        complex_psum = PauliSum(3)
        for (pstr, coeff) in result_psum
            add!(complex_psum, pstr, coeff)
        end
        @test result_psum ≈ complex_psum
        @test result_vecpsum ≈ complex_psum
    end

    @testset "+ PauliSum" begin
        psum1 = PauliSum(3)
        add!(psum1, [:I, :I, :Y], 1:3, 1.0)
        add!(psum1, :I, 1, 1.5)
        psum2 = PauliSum(PauliString(3, :I, 1, 1.5))

        # test out-of-place
        psum3 = deepcopy(psum1)
        psum4 = deepcopy(psum2)

        result_psum = psum1 + psum2
        expected_psum = PauliSum(PauliString(3, [:I, :I, :Y], 1:3, 1.0))
        add!(expected_psum, :I, 1, 3.0)
        @test result_psum == expected_psum
        @test psum1 == psum3
        @test psum2 == psum4
    end

    @testset "- PauliSum" begin
        psum1 = PauliSum(PauliString(3, [:I, :I, :Y], 1:3, 1im))
        add!(psum1, :I, 1, 1.5im)
        psum2 = PauliSum(PauliString(3, :I, 1, 1.5im))

        # test out-of-place
        psum3 = deepcopy(psum1)
        psum4 = deepcopy(psum2)

        result_psum = psum1 - psum2
        expected_psum = PauliSum(PauliString(3, [:I, :I, :Y], 1:3, 1im))
        @test result_psum == expected_psum
        @test psum1 == psum3
        @test psum2 == psum4
    end

    @testset "* PauliSum" begin
        c = 2
        psum1 = PauliSum(3)
        add!(psum1, [:I, :I, :Y], 1:3, 1.0)
        add!(psum1, :I, 1, 1.5)

        # test out-of-place
        psum2 = deepcopy(psum1)

        result_psum = psum1 * c
        expected_psum = PauliSum(PauliString(3, [:I, :I, :Y], 1:3, 2.0))
        add!(expected_psum, :I, 1, 3.0)
        @test result_psum == expected_psum
        @test psum1 == psum2

        c = 2.0 + 1im
        psum2 = PauliSum(PauliString(3, :I, 1, 1.5im))
        result_psum = psum2 * c
        expected_psum = PauliSum(PauliString(3, :I, 1, 3im - 1.5))
        @test result_psum == expected_psum
    end

    @testset "/ PauliSum" begin
        c = 2
        psum1 = PauliSum(3)
        add!(psum1, [:I, :I, :Y], 1:3, 1.0)
        add!(psum1, :I, 1, 1.5)

        # test out-of-place
        psum2 = deepcopy(psum1)

        result_psum = psum1 / c
        expected_psum = PauliSum(PauliString(3, [:I, :I, :Y], 1:3, 0.5))
        add!(expected_psum, :I, 1, 0.75)
        @test result_psum == expected_psum
        @test psum1 == psum2
    end

end

@testset "* for two PauliStrings" begin

    # Test for single-qubit pauli string
    @testset "Single Qubit PauliString" begin
        nq = 1
        pstr1 = PauliString(nq, :X, 1)
        pstr2 = PauliString(nq, :Y, 1)
        result = pstr1 * pstr2
        expected_result = PauliString(nq, :Z, 1, 1im)
        @test result == expected_result
    end

    # Test for multi-qubit pauli string
    @testset "Multi Qubit PauliString" begin
        nq = 3
        pstr1 = PauliString(nq, [:X, :Y], [1, 2], 2.0)
        pstr2 = PauliString(nq, [:Y, :Z], [2, 3])
        result = pstr1 * pstr2

        expected_result = PauliString(nq, [:X, :Z], [1, 3], 2.0 + 0im)

        @test result == expected_result
    end
end

@testset "PauliSum * PauliString" begin

    @testset "Multiply with Identity" begin
        nq = 3
        psum = PauliSum(PauliString(nq, :I, 1, 1.5))
        pstr = PauliString(nq, [:I, :I], [1, 2], 2.0)
        result = psum * pstr
        expected_result = PauliSum(PauliString(nq, :I, 2, 3 + 0im))
        @test result == expected_result
    end

    @testset "Multiply with PauliString" begin
        nq = 3
        psum = PauliSum(nq)
        add!(psum, [:X, :Y], 2:3, 1.5)
        add!(psum, :Y, 3, 1.0)
        pstr = PauliString(nq, [:Y, :Z], [1, 2], 2.0)
        result = psum * pstr

        expected_result = PauliSum(nq)
        add!(expected_result, [:Y, :Z, :Y], [1, 2, 3], 2.0)
        expected_result = expected_result + PauliString(
            nq, [:Y, :Y, :Y], [1, 2, 3], -3im
        )

        @test result == expected_result
    end
end


@testset "PauliString * PauliSum" begin

    @testset "Multiply with Identity Complex Coefficients" begin
        nq = 3
        psum = PauliSum(PauliString(nq, [:I, :I, :I], 1:3, 1.5im))
        pstr = PauliString(nq, [:I, :I], [1, 2], 2)
        result = psum * pstr
        expected_result = PauliSum(PauliString(nq, [:I, :I, :I], 1:3, 3im))
        @test result == expected_result
    end

    @testset "Multiply with PauliString" begin
        nq = 3
        pstr = PauliString(nq, [:Y, :Z], [1, 2], 2.0)
        psum = PauliSum([PauliString(nq, [:X, :Y], 2:3, 1.5), PauliString(nq, :Y, 3, 1.0)])
        result = pstr * psum

        expected_result = PauliSum(ComplexF64, nq)
        add!(expected_result, [:Y, :Z, :Y], [1, 2, 3], 2.0 + 0im)
        add!(expected_result, [:Y, :Y, :Y], [1, 2, 3], 3im)

        @test result == expected_result
    end
end
