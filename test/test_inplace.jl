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
    function random_pauli_sum(PS, nq, nterms)
        pstrs = [PauliString(nq, rand([:I, :X, :Y, :Z], nq), 1:nq, randn()) for _ in 1:nterms]
        return PS(pstrs)
    end

    for PS in (PauliSum, VectorPauliSum)
        psum = random_pauli_sum(PS, 4, 10)
        psum_truncated_outofplace = truncate(psum; min_abs_coeff=0.1)
        psum_truncated_inplace = truncate!(psum; min_abs_coeff=0.1)

        @test psum_truncated_inplace === psum
        @test psum_truncated_inplace == psum_truncated_outofplace
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
    end

end
