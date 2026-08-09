using Test
using PauliPropagation

@testset "In-place propagate" begin

    nq = 2
    # it can be layer-dependent
    for nl in 1:4

        pstr = PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq))

        # contains clifford gates as well
        circ = hardwareefficientcircuit(nq, nl)

        nparams = countparameters(circ)
        thetas = randn(nparams)

        for PS in (PauliSum, VectorPauliSum)
            psum = PS(pstr)
            psum_evolved = propagate!(circ, psum, thetas; min_abs_coeff=0)
            psum_evolved_outofplace = propagate(circ, PS(pstr), thetas; min_abs_coeff=0)

            @test psum_evolved === psum
            @test psum_evolved == psum_evolved_outofplace
        end
    end

end


@testset "truncate" begin

    nq = 4
    pstrs = [
        PauliString(nq, [:X, :Y], [1, 2], 0.5),   # survives
        PauliString(nq, [:Z], [3], 0.05),         # truncated (below threshold)
        PauliString(nq, [:X, :Z], [2, 4], -0.2),  # survives
        PauliString(nq, [:Y], [1], 0.099),        # truncated (just below threshold)
        PauliString(nq, [:Z, :Y], [1, 3], 0.1),   # survives (== threshold)
    ]

    for PS in (PauliSum, VectorPauliSum)
        psum = PS(pstrs)
        psum_truncated_outofplace = truncate(psum; min_abs_coeff=0.1)
        psum_truncated_inplace = truncate!(psum; min_abs_coeff=0.1)

        @test psum_truncated_inplace === psum
        @test psum_truncated_inplace == psum_truncated_outofplace

        @test length(psum_truncated_inplace) == 3
        @test getcoeff(psum_truncated_inplace, [:X, :Y], [1, 2]) ≈ 0.5
        @test getcoeff(psum_truncated_inplace, [:X, :Z], [2, 4]) ≈ -0.2
        @test getcoeff(psum_truncated_inplace, [:Z, :Y], [1, 3]) ≈ 0.1
        @test getcoeff(psum_truncated_inplace, [:Z], [3]) == 0.0
        @test getcoeff(psum_truncated_inplace, [:Y], [1]) == 0.0
    end

end


@testset "merge" begin
    nq = 1
    pstrs = [PauliString(nq, :X, 1, 0.5), PauliString(nq, :Y, 1, 0.3), PauliString(nq, :Y, 1, 0.4), PauliString(nq, :X, 1, -0.2)]

    for PS in (PauliSum, VectorPauliSum)
        psum = PS(pstrs)
        psum_merged_outofplace = merge(psum)
        psum_merged_inplace = merge!(psum)

        @test psum_merged_inplace === psum
        @test psum_merged_inplace == psum_merged_outofplace

        @test length(psum_merged_inplace) == 2
        @test getcoeff(psum_merged_inplace, :X, 1) ≈ 0.3
        @test getcoeff(psum_merged_inplace, :Y, 1) ≈ 0.7
    end

end
