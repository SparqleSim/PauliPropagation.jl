# Test File for symmetries.jl
using Test
using PauliPropagation: _periodicshiftup

function get_psum(nq)
    """
    Create a PauliSum with terms that can be merged by translational symmetries.
    """
    input_psum = PauliSum(nq)
    add!(input_psum, :Z, 3)
    add!(input_psum, :Z, 6)
    add!(input_psum, :Z, 5)
    add!(input_psum, [:X], [5], 0.5)
    add!(input_psum, [:X, :Z], [2, 5])
    add!(input_psum, [:Z, :X], [2, 5], 0.5)
    add!(input_psum, [:X, :Y, :Z], [1, 3, 6])
    add!(input_psum, [:Y, :X, :Z], [1, 2, 4])

    return input_psum
end


@testset "Translation 1d merging" begin
    nq = 6
    input_psum = get_psum(nq)

    expected_psum = PauliSum(nq)
    add!(expected_psum, :Z, 1, 3)
    add!(expected_psum, [:Z, :X], [1, 4], 1.5)
    add!(expected_psum, [:X], [1], 0.5)
    add!(expected_psum, [:Y, :X, :Z], [1, 2, 4], 1)
    add!(expected_psum, [:Z, :X, :Y], [1, 2, 4], 1)

    merged_psum = translationmerge(input_psum)
    @test merged_psum == expected_psum

    merged_vecpsum = translationmerge(VectorPauliSum(input_psum))
    @test PauliSum(merged_vecpsum) == expected_psum

end

@testset "Full shiftup 2D translation merging" begin
    nx, ny = 3, 2
    nq = nx * ny
    input_psum = get_psum(nq)

    expected_psum = PauliSum(nq)
    add!(expected_psum, :Z, 1, 3)
    add!(expected_psum, [:Z, :X], [1, 4], 1.5)
    add!(expected_psum, [:X], [1], 0.5)
    add!(expected_psum, [:Y, :X, :Z], [1, 2, 4], 2)

    # test merging with shiftup
    @test translationmerge(input_psum, nx, ny) == expected_psum

    merged_vecpsum = translationmerge(VectorPauliSum(input_psum), nx, ny)
    @test PauliSum(merged_vecpsum) == expected_psum

end


@testset "translationmerge does not mutate its input" begin
    nq = 6
    vpsum = VectorPauliSum(get_psum(nq))
    original = deepcopy(vpsum)

    translationmerge(vpsum)

    @test PauliSum(vpsum) == PauliSum(original)
end

@testset "translationmerge accepts a propagation cache" begin
    nq = 6
    nx, ny = 3, 2
    input_psum = get_psum(nq)

    for psum in (input_psum, VectorPauliSum(input_psum))
        @test PauliSum(translationmerge(PropagationCache(psum))) == PauliSum(translationmerge(psum))
        @test PauliSum(translationmerge(PropagationCache(psum), nx, ny)) == PauliSum(translationmerge(psum, nx, ny))
    end
end

@testset "translationmerge thread=false matches thread=true" begin
    nq = 6
    vpsum = VectorPauliSum(get_psum(nq))

    merged_thread = translationmerge(vpsum; thread=true)
    merged_nothread = translationmerge(vpsum; thread=false)

    @test PauliSum(merged_thread) == PauliSum(merged_nothread)
end

@testset "translationmerge grid dimension mismatch" begin
    nq = 6
    input_psum = get_psum(nq)

    # nx * ny must equal nqubits(psum)
    @test_throws ArgumentError translationmerge(input_psum, 2, 4)
    @test_throws ArgumentError translationmerge(input_psum, 4, 1)
    @test_throws ArgumentError translationmerge(VectorPauliSum(input_psum), 2, 4)

    # sanity check: matching dimensions do not throw
    @test translationmerge(input_psum, 2, 3) isa PauliSum
end

@testset "symmetrymerge takes the map first and works on every backend" begin
    nq = 6
    input_psum = get_psum(nq)
    expected_psum = translationmerge(input_psum)
    mapfunc = pstr -> PauliPropagation._translatetolowestinteger(pstr, nq)

    @test symmetrymerge(mapfunc, input_psum) == expected_psum
    @test PauliSum(symmetrymerge(mapfunc, VectorPauliSum(input_psum))) == expected_psum
    @test PauliSum(symmetrymerge(mapfunc, MultiPauliSum(input_psum, 4))) == expected_psum
    @test PauliSum(symmetrymerge(mapfunc, MultiPauliSum(VectorPauliSum(input_psum), 4))) == expected_psum
    @test PauliSum(symmetrymerge(mapfunc, PropagationCache(VectorPauliSum(input_psum)))) == expected_psum
end

@testset "translationmerge! merges in place and returns its input" begin
    nq = 6
    nx, ny = 3, 2
    expected_1d = translationmerge(get_psum(nq))
    expected_2d = translationmerge(get_psum(nq), nx, ny)

    for tosum in (identity, VectorPauliSum, psum -> MultiPauliSum(psum, 4))
        psum_1d = tosum(get_psum(nq))
        @test translationmerge!(psum_1d) === psum_1d
        @test PauliSum(psum_1d) == expected_1d

        psum_2d = tosum(get_psum(nq))
        @test translationmerge!(psum_2d, nx, ny) === psum_2d
        @test PauliSum(psum_2d) == expected_2d
    end

    prop_cache = PropagationCache(VectorPauliSum(get_psum(nq)))
    @test translationmerge!(prop_cache) === prop_cache
    @test PauliSum(prop_cache) == expected_1d
end

@testset "translationmerge! on a propagated cache" begin
    # propagate! leaves a full sorted prefix behind, which remapping the terms must invalidate
    # so that merge! deduplicates instead of taking the sorted-tail path
    nq = 6
    circuit = [PauliRotation([:X, :X], [1, 2]), PauliRotation([:Y, :Y], [3, 4])]
    thetas = [0.3, 0.7]
    expected = translationmerge!(propagate(circuit, get_psum(nq), thetas; min_abs_coeff=0.0))

    for psum in (get_psum(nq), VectorPauliSum(get_psum(nq)), MultiPauliSum(get_psum(nq), 4))
        prop_cache = PropagationCache(psum)
        propagate!(circuit, prop_cache, thetas; min_abs_coeff=0.0)
        translationmerge!(prop_cache)

        @test PauliSum(prop_cache) == expected
        @test length(prop_cache) == length(expected)
    end
end
