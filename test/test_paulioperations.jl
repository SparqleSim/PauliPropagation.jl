using Test

@testset "pauliprod Tests" begin
    """Test the product of PauliStrings and PauliSums."""

    # Test with PauliStrings
    @testset "PauliString Product" begin
        # X * Y = iZ
        nq = 2
        qind = 1
        pstr1 = PauliString(nq, :X, qind, 1)
        pstr2 = PauliString(nq, :Y, qind, 1.5)
        pstr3 = pauliprod(pstr1, pstr2)
        @test pstr3 == PauliString(nq, :Z, qind, pstr1.coeff * pstr2.coeff * 1im)
        @test pauliprod(pstr2, pstr1).coeff == -1 * pstr3.coeff

        # Z * I = Z
        nq = 5
        qind = 4
        pstr1 = PauliString(nq, :Z, qind, 1.2 + 0.2im)
        pstr2 = PauliString(nq, :I, qind, 1im)
        pstr3 = pauliprod(pstr1, pstr2)
        @test pstr3 == PauliString(nq, :Z, qind, pstr1.coeff * pstr2.coeff)
        @test pauliprod(pstr2, pstr1).coeff == pstr3.coeff

        # Z * I = Z
        nq = 7
        qind = 7
        pstr1 = PauliString(nq, :I, qind, -1im)
        pstr2 = PauliString(nq, :X, qind, 0)
        pstr3 = pauliprod(pstr1, pstr2)
        @test pstr3 == PauliString(nq, :X, qind, pstr1.coeff * pstr2.coeff)
        @test pauliprod(pstr2, pstr1).coeff == pstr3.coeff


        # X * X = I
        nq = 13
        qind = 2
        pstr1 = PauliString(nq, :X, qind, 1im + 0.5)
        pstr2 = PauliString(nq, :X, qind, -0.2)
        pstr3 = pauliprod(pstr1, pstr2)
        @test pstr3 == PauliString(nq, :I, qind, pstr1.coeff * pstr2.coeff)
        @test pauliprod(pstr2, pstr1).coeff == pstr3.coeff


        # XY * ZY = -iYI
        nq = 17
        qinds = [3, 14]
        pstr1 = PauliString(nq, [:X, :Y], qinds, 0.3)
        pstr2 = PauliString(nq, [:Z, :Y], qinds, 5)
        pstr3 = pauliprod(pstr1, pstr2)
        @test pstr3 == PauliString(nq, [:Y, :I], qinds, pstr1.coeff * pstr2.coeff * -1im * 1)
        @test pauliprod(pstr2, pstr1).coeff == -1 * pstr3.coeff


        # I * I = I
        nq = 32
        qind = 16
        pstr1 = PauliString(nq, :I, qind, 2.1)
        pstr2 = PauliString(nq, :I, qind, 3im + 0.1)
        pstr3 = pauliprod(pstr1, pstr2)
        @test pstr3 == PauliString(nq, :I, qind, pstr1.coeff * pstr2.coeff)
        @test pauliprod(pstr2, pstr1).coeff == pstr3.coeff

        # XY * ZY = -iYI
        nq = 33
        qinds = [1, 29]
        pstr1 = PauliString(nq, [:X, :I], qinds, 2im)
        pstr2 = PauliString(nq, [:I, :Y], qinds, -3)
        pstr3 = pauliprod(pstr1, pstr2)
        @test pstr3 == PauliString(nq, [:X, :Y], qinds, pstr1.coeff * pstr2.coeff)
        @test pauliprod(pstr2, pstr1).coeff == pstr3.coeff

        # ZX * XZ = YY
        nq = 65
        qinds = [62, 31]
        pstr1 = PauliString(nq, [:Z, :X], qinds, 1im)
        pstr2 = PauliString(nq, [:X, :Z], qinds, π)
        pstr3 = pauliprod(pstr1, pstr2)
        @test pstr3 == PauliString(nq, [:Y, :Y], qinds, pstr1.coeff * pstr2.coeff)
        @test pauliprod(pstr2, pstr1).coeff == pstr3.coeff
    end

    @testset "PauliSums" begin
        nq = 3
        psum1 = PauliSum(nq)
        add!(psum1, [:Z, :X], [1, 2])

        psum2 = PauliSum(nq)
        add!(psum2, [:Y, :Y], [1, 3])
        add!(psum2, [:X, :Y], [1, 2])

        # Psum * I = Psum
        psum_identity = PauliSum(nq)
        add!(psum_identity, [:I], [1], 1.0)
        @test pauliprod(psum2, psum_identity) == psum2

        # Expected product
        psum_expected = PauliSum(nq, Dict{getinttype(psum1.nqubits),ComplexF64}())
        add!(psum_expected, [:Y, :Z], [1, 2], -1.)
        add!(psum_expected, [:X, :X, :Y], [1, 2, 3], -1im)
        @test pauliprod(psum1, psum2) == psum_expected

        # Test real coefficients conversion
        psum2 = PauliSum(nq)
        add!(psum2, [:Y, :Y], [1, 2], 2.0)
        psum_expected = PauliSum(nq)
        add!(psum_expected, [:X, :Z], [1, 2], 2.0)
        @test pauliprod(psum1, psum2) == psum_expected
    end

    @testset "Commutators" begin
        # [X, Y] = iZ
        nq = 1
        pstr1 = PauliString(nq, :X, 1, 1.0)
        pstr2 = PauliString(nq, :Y, 1, 1.0)
        @test commutator(pstr1, pstr2) == PauliString(nq, :Z, 1, 2im)

        # [P, I] = 0
        pstr_identity = PauliString(nq, :I, 1, 1.0)
        for p in (:I, :X, :Y, :Z)
            pstr = PauliString(nq, p, 1, 1.0)
            @test commutator(pstr, pstr_identity) == PauliString(nq, :I, 1, 0im)
        end

        # [XX, ZY] = 0
        nq = 2
        pstr1 = PauliString(nq, [:X, :X], [1, 2], 1.0)
        pstr2 = PauliString(nq, [:Z, :Y], [1, 2], 1.0)
        @test commutator(pstr1, pstr2) == PauliString(nq, :I, 1, 0im)

        # [XY, YI] = -iYI
        nq = 2
        pstr1 = PauliString(nq, [:X, :Y], [1, 2], 1.0)
        pstr2 = PauliString(nq, [:Y, :I], [1, 2], 1.0)
        @test commutator(pstr1, pstr2) == PauliString(nq, [:Z, :Y], [1, 2], 2im)
    end

    @testset "VectorPauliSum commutator" begin
        nq = 3
        # VectorPauliSum commutator verify with PauliSum commutator
        psum1 = PauliSum(nq)
        add!(psum1, [:X, :Y], [1, 2], 1.5)
        psum2 = PauliSum(nq)
        add!(psum2, [:Y, :I], [1, 2], -2.0)

        expected = commutator(psum1, psum2)
        com_vec = commutator(VectorPauliSum(psum1), VectorPauliSum(psum2))
        @test PauliSum(com_vec) == expected           
        
        vpsum = VectorPauliSum(3, [1, 2], [1., 1.])
        vpsum2 = VectorPauliSum(3, [7], [0.25])
        expected = commutator(PauliSum(vpsum), PauliSum(vpsum2))
        com_vec = commutator(vpsum, vpsum2)
        @test PauliSum(com_vec) == expected

    end    
end


@testset "Count Paulis" begin
    """Test counting X, Y, Z in PauliStrings."""

    # Verify counting functions for X, Y, Z, X|Y, Y|Z and overlall weight
    function _count_verification(pstr, expected_counts)
        countx(pstr) == expected_counts.X
        county(pstr) == expected_counts.Y
        countz(pstr) == expected_counts.Z
        countxy(pstr) == expected_counts.XY
        countyz(pstr) == expected_counts.YZ
        countweight(pstr) == (
            expected_counts.X + expected_counts.Y + expected_counts.Z
        ) || error("countweight failed")
    end

    nq = 6
    @testset "Counting Identity" begin
        pstr = PauliString(nq, [:I], 1)
        expected_counts = (X=0, Y=0, Z=0, XY=0, YZ=0)
        _count_verification(pstr, expected_counts)
    end

    @testset "Counting Single Pauli X" begin
        pstr = PauliString(nq, [:X, :X], [2, 3])
        expected_counts = (X=2, Y=0, Z=0, XY=2, YZ=0)
        _count_verification(pstr, expected_counts)
    end

    @testset "Counting Single Pauli Y" begin
        pstr = PauliString(nq, [:Y, :Y], [1, 5])
        expected_counts = (X=0, Y=2, Z=0, XY=2, YZ=2)
        _count_verification(pstr, expected_counts)
    end

    @testset "Counting Single Pauli Z" begin
        pstr = PauliString(nq, [:Z, :Z], [4, 6])
        expected_counts = (X=0, Y=0, Z=2, XY=0, YZ=2)
        _count_verification(pstr, expected_counts)
    end

    @testset "Counting Mixed Paulis" begin
        pstr = PauliString(nq, [:X, :Y, :I, :Z, :X, :Y], collect(1:nq))
        expected_counts = (X=2, Y=2, Z=1, XY=4, YZ=3)
        _count_verification(pstr, expected_counts)
    end
end

@testset "LinearAlgebra" begin
    for nq in 1:10
        λ = rand()
        @test tr(λ * PauliString(nq, :I, rand(1:nq))) == λ * 2.0^nq
        @test tr(λ * PauliString(nq, :X, rand(1:nq))) == 0.0
        @test tr(λ * PauliString(nq, :Y, rand(1:nq))) == 0.0
        @test tr(λ * PauliString(nq, :Z, rand(1:nq))) == 0.0

        @test trace(λ * PauliString(nq, :I, rand(1:nq))) == λ * 2.0^nq
        @test trace(λ * PauliString(nq, :X, rand(1:nq))) == 0.0
        @test trace(λ * PauliString(nq, :Y, rand(1:nq))) == 0.0
        @test trace(λ * PauliString(nq, :Z, rand(1:nq))) == 0.0
    end

    nq = 42
    letters = [rand([:X, :Y, :Z]) for _ in 1:nq]
    pstr = PauliString(nq, letters, 1:nq)
    @test tr(pstr) == 0.0
    @test trace(pstr) == 0.0

    psum = pstr + PauliString(nq, :I, 1) / 137.
    @test tr(psum) == 2.0^nq / 137.
    @test trace(psum) == 2.0^nq / 137.
end
