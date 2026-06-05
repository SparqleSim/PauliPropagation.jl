## Test the utils for PauliAlgebra/datatypes.jl module.

using Test

import PauliPropagation: pauli_symbols

@testset "Identity" begin

    for nq in [1, 4, 7, 17, 33, 64, 127, 1024]
        created_id = identitypauli(nq)
        manual_id = PauliPropagation.getinttype(nq)(0)
        @test created_id == manual_id == 0
        @test typeof(created_id) == typeof(manual_id)
        @test typeof(identitypauli(typeof(created_id))) == typeof(created_id)
        @test typeof(identitylike(created_id)) == typeof(created_id)
    end
end

@testset "Int to Symbol" begin

    # Test the conversion from integer to symbol
    for (i, symbol) in enumerate(pauli_symbols)
        @test inttosymbol(i - 1) == symbol # i-1 because integers are 0-indexed
    end

    # Test the bounds error for converting integers outside [0, 3]
    @test_throws ArgumentError inttosymbol(4)
    @test_throws ArgumentError inttosymbol(-1)

    # Test multiqubit conversion. Note that the order of the symbols is reversed
    # compared to the bitstring to int conversion.
    @test inttosymbol(1, 2) == [:X, :I]
    @test inttosymbol(4, 2) == [:I, :X]
end

@testset "Symbol to Int" begin

    # Test the conversion from a symbol to integer
    for i in 0:3
        @test symboltoint(inttosymbol(i)) == i
    end

    # Test 2-qubit conversion
    nqubits = 2
    for i in 4:15
        @test symboltoint(inttosymbol(i, nqubits)) == i
    end

end

@testset "Get Pauli" begin

    # For single qubit pauli, the integer representation is the same.
    for i in 0:3
        @test getpauli(i, 1) == i
    end

    pstr = PauliString(3, [:X], [2])
    # Test for `pstr` of `PauliString` type.
    @test getpauli(pstr.term, 1) == 0
    @test getpauli(pstr.term, 2) == 1

end

@testset "Get Paulis" begin

    # Test consistency with integer representation
    rand_int = 13
    @test getpauli(rand_int, [1, 2]) == 13

    # Test with `PauliStringType`
    nqubits = 4
    pstr = PauliString(nqubits, :X, 2)
    inds = [1, 3]
    @test getpauli(pstr.term, inds) == 0
    inds = [3, 1]
    @test getpauli(pstr.term, inds) == 0
    inds = [1, 2]
    @test getpauli(pstr.term, inds) == 4

    # if the indices are not sorted, `getpauli` will reutrn a pauli
    # integer representation according to the unsorted indices
    inds = [2, 1]
    @test getpauli(pstr.term, inds) == 1

    # Test with inttosymbol
    inds = [2, 4]
    symbols = [:X, :Z]
    pstr = PauliString(nqubits, symbols, inds)

    @test inttosymbol(
        getpauli(pstr.term, inds), length(inds)
    ) == symbols

    # flip `qinds` will return reversed symbols
    @test inttosymbol(
        getpauli(pstr.term, [4, 2]), length(inds)
    ) == [symbols[2], symbols[1]]

end

@testset "Set Pauli" begin
    nqubits = 4

    # Test pstr with `PauliStringType`
    pstr = PauliString(nqubits, :X, 2)
    expected_pstr = PauliString(nqubits, [:Z, :X], [1, 2])
    @test setpauli(pstr.term, :Z, 1) == expected_pstr.term

    # Test target_pauli as integer
    @test setpauli(pstr.term, 3, 1) == expected_pstr.term

end


@testset "Get Pauli for a range of indices" begin
    nq = 12
    pstr = PauliString(nq, [:X, :Z], [3, 9])

    @test getpauli(pstr.term, range(2, 10)) == getpauli(pstr.term, range(2, 10))
    @test getpauli(pstr.term, range(2, 10)) == getpauli(pstr.term, 2, 10)
end

@testset "Set Pauli Continuous" begin
    nq = 12
    pstr = PauliString(nq, [:X, :Z], [3, 9])
    target_pstr = PauliString(nq, [:Y, :X], [4, 5])
    @test setpauli(pstr.term, target_pstr.term, 2, 10) == setpauli(pstr.term, target_pstr.term, range(2, 10))
end


@testset "Convert PauliString/PauliSum coeff types" begin

    @testset "PauliString" begin
        pstr = PauliString(2, :X, 1, 1 + 0im)
        @test convertcoefftype(Float64, pstr) == PauliString(2, :X, 1, 1.)

        pstr = PauliString(2, [:X, :Y], [1, 2], 1)
        @test convertcoefftype(ComplexF64, pstr) == PauliString(2, [:X, :Y], [1, 2], 1 + 0im)
    end

    @testset "PauliSum" begin
        psum = PauliSum(ComplexF64, 2)
        add!(psum, [:X, :Y], [1, 2], 2 + 0im)
        add!(psum, :Z, 1, 1 + 0im)
        expected_psum = PauliSum(2)
        add!(expected_psum, [:X, :Y], [1, 2], 2)
        add!(expected_psum, :Z, 1, 1)
        @test convertcoefftype(Float64, psum) == expected_psum
    end

end

@testset "Set Pauli for `PauliString` type" begin

    # nqubits = 4
    # pstr = PauliString(nqubits, [:X, :Z], [1, 4])

    # TODO: Implement setpauli() PauliString and then also getpauli()
    # Test pstr with `PauliString` type and target_pauli as `Integer`
    # transformed_pstr = setpauli(pstr, 2, [1, 3])
    # expected_pstr = PauliString(4, [:Y, :Z], [1, 4])
    # @test transformed_pstr == expected_pstr

    # Test pstr with `PauliString` type and target_pauli as `Symbol`
    # transformed_pstr = setpauli(pstr, [:Y, :X], [1, 3])
    # expected_pstr = PauliString(4, [:Y, :X, :Z], [1, 3, 4])
    # @test transformed_pstr == expected_pstr

    # Test pstr with `PauliStringType` and target_pauli as `Integer`
    # transformed_pstr = setpauli(pstr.term, symboltoint([:Y, :X]), [1, 3])
    # @test transformed_pstr == expected_pstr.term

    # expected_pstr = PauliString(4, [:Y, :X, :Z], [3, 1, 4])
    # transformed_pstr = setpauli(pstr.term, [:Y, :X], [3, 1])
    # @test transformed_pstr == expected_pstr.term
end


