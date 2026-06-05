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
