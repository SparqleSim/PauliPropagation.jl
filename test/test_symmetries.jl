# Test File for symmetries.jl
using Test
using Random: randperm
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

@testset "Out-of-place merging does not mutate input" begin
    nq = 6
    nx, ny = 3, 2

    # `symmetrymerge` builds a PropagationCache, which wraps the VectorPauliSum
    # without copying it. The out-of-place methods must copy first, or they
    # silently corrupt the caller's sum (merged terms, unresized arrays).
    for mergecall in (psum -> translationmerge(psum),
                      psum -> translationmerge(psum, nx, ny))

        input_vecpsum = VectorPauliSum(get_psum(nq))
        reference_terms = copy(input_vecpsum.terms)
        reference_coeffs = copy(input_vecpsum.coeffs)

        merged_vecpsum = mergecall(input_vecpsum)

        # input is untouched, down to its length
        @test input_vecpsum.terms == reference_terms
        @test input_vecpsum.coeffs == reference_coeffs
        @test PauliSum(input_vecpsum) == get_psum(nq)

        # output shares no storage with the input
        @test merged_vecpsum.terms !== input_vecpsum.terms
        @test merged_vecpsum.coeffs !== input_vecpsum.coeffs

        # merging twice from the same input gives the same result
        @test PauliSum(merged_vecpsum) == PauliSum(mergecall(input_vecpsum))
    end
end

@testset "Reflection 1d merging" begin
    nq = 6
    input_psum = PauliSum(nq)
    add!(input_psum, :Z, 1)
    add!(input_psum, :Z, 6)                     # mirror image of Z on site 1
    add!(input_psum, [:X, :Y], [2, 3])
    add!(input_psum, [:Y, :X], [4, 5], 0.5)     # mirror image of XY on sites 2, 3
    add!(input_psum, [:X, :Y], [4, 5])          # not a mirror image (order matters)

    # representatives are the lowest integers, i.e. the images on the lower sites
    expected_psum = PauliSum(nq)
    add!(expected_psum, :Z, 1, 2)
    add!(expected_psum, [:X, :Y], [2, 3], 1.5)
    add!(expected_psum, [:Y, :X], [2, 3])

    @test reflectionmerge(input_psum) == expected_psum
    @test PauliSum(reflectionmerge(VectorPauliSum(input_psum))) == expected_psum
    @test PauliSum(reflectionmerge!(VectorPauliSum(input_psum))) == expected_psum
    @test PauliSum(reflectionmerge!(PropagationCache(VectorPauliSum(input_psum)))) == expected_psum
end

@testset "Reflection 2d merging" begin
    # 3x2 grid, sites numbered row by row:
    #   1 2 3
    #   4 5 6
    nx, ny = 3, 2
    nq = nx * ny
    input_psum = PauliSum(nq)
    add!(input_psum, :Z, 1)             # corner
    add!(input_psum, :Z, 3)             # x-mirror of site 1
    add!(input_psum, :Z, 4)             # y-mirror of site 1
    add!(input_psum, :Z, 6)             # x- and y-mirror of site 1
    add!(input_psum, :Z, 2)             # top middle: only y-mirror is site 5
    add!(input_psum, [:X, :Y], [1, 2])  # -> ZYX-type images: [Y,X] on (2,3), [X,Y] on (4,5), [Y,X] on (5,6)
    add!(input_psum, [:Y, :X], [5, 6], 2.0)

    # both mirrors (default): full point group of the rectangle
    expected_both = PauliSum(nq)
    add!(expected_both, :Z, 1, 4)
    add!(expected_both, :Z, 2)
    add!(expected_both, [:X, :Y], [1, 2], 3.0)
    @test reflectionmerge(input_psum, nx, ny) == expected_both
    @test PauliSum(reflectionmerge(VectorPauliSum(input_psum), nx, ny)) == expected_both
    @test PauliSum(reflectionmerge!(VectorPauliSum(input_psum), nx, ny)) == expected_both
    @test PauliSum(reflectionmerge!(PropagationCache(VectorPauliSum(input_psum)), nx, ny)) == expected_both

    # x-mirror only: site 1 ~ 3, 4 ~ 6, but 1 !~ 4
    expected_x = PauliSum(nq)
    add!(expected_x, :Z, 1, 2)
    add!(expected_x, :Z, 4, 2)
    add!(expected_x, :Z, 2)
    add!(expected_x, [:X, :Y], [1, 2])
    add!(expected_x, [:X, :Y], [4, 5], 2.0)
    @test reflectionmerge(input_psum, nx, ny; axes=:x) == expected_x
    @test reflectionmerge(input_psum, nx, ny; axes=(:x,)) == expected_x

    # y-mirror only: site 1 ~ 4, 3 ~ 6, 2 ~ 5, but 1 !~ 3
    expected_y = PauliSum(nq)
    add!(expected_y, :Z, 1, 2)
    add!(expected_y, :Z, 3, 2)
    add!(expected_y, :Z, 2)
    add!(expected_y, [:X, :Y], [1, 2])
    add!(expected_y, [:Y, :X], [2, 3], 2.0)
    @test reflectionmerge(input_psum, nx, ny; axes=:y) == expected_y

    # validation
    @test_throws ArgumentError reflectionmerge(input_psum, 2, 2)
    @test_throws ArgumentError reflectionmerge!(VectorPauliSum(input_psum), 2, 2)
    @test_throws ArgumentError reflectionmerge(input_psum, nx, ny; axes=:z)
    @test_throws ArgumentError reflectionmerge(input_psum, nx, ny; axes=())
end

@testset "Permutation merging" begin
    nq = 4

    pstr1 = PauliString(4, [:X, :Y, :Z], [1, 2, 3])
    pstr2 = PauliString(4, [:Y, :Z, :X], [2, 3, 4])
    pstr3 = PauliString(4, [:Z, :X, :Y], [1, 3, 4])
    psum = PauliSum([pstr1, pstr2, pstr3])
    rep_perm = permutationmerge(psum)
    
    expected = PauliSum(PauliString(4, [:X, :Y, :Z], [1, 2, 3], 3.0))
    @test rep_perm == expected
end

@testset "Permutation merging (all-to-all)" begin
    nq = 6
    input_psum = PauliSum(nq)
    # three strings with one X, one Y and one Z each, on different sites and in different orders
    add!(input_psum, [:X, :Y, :Z], [1, 2, 3])
    add!(input_psum, [:Z, :Y, :X], [4, 5, 6], 0.5)
    add!(input_psum, [:Y, :Z, :X], [1, 4, 6], 0.25)
    # two strings with two Z each
    add!(input_psum, [:Z, :Z], [1, 6])
    add!(input_psum, [:Z, :Z], [3, 4], 2.0)
    # a string with an X only, and one with a Y only: not equivalent to each other
    add!(input_psum, :X, 5)
    add!(input_psum, :Y, 5)

    expected_psum = PauliSum(nq)
    add!(expected_psum, [:X, :Y, :Z], [1, 2, 3], 1.75)
    add!(expected_psum, [:Z, :Z], [1, 2], 3.0)
    add!(expected_psum, :X, 1)
    add!(expected_psum, :Y, 1)

    @test permutationmerge(input_psum) == expected_psum
    @test PauliSum(permutationmerge(VectorPauliSum(input_psum))) == expected_psum
    @test PauliSum(permutationmerge!(VectorPauliSum(input_psum))) == expected_psum
    @test PauliSum(permutationmerge!(PropagationCache(VectorPauliSum(input_psum)))) == expected_psum

    # permutation symmetry contains translation and reflection symmetry
    psum = get_psum(6)
    @test permutationmerge(translationmerge(psum)) == permutationmerge(psum)
    @test permutationmerge(reflectionmerge(psum, 3, 2)) == permutationmerge(psum)
end

@testset "Out-of-place reflection/permutation merging does not mutate input" begin
    nq = 6
    for mergecall in (psum -> reflectionmerge(psum),
                      psum -> reflectionmerge(psum, 3, 2),
                      psum -> permutationmerge(psum))
        input_vecpsum = VectorPauliSum(get_psum(nq))
        reference_terms = copy(input_vecpsum.terms)
        reference_coeffs = copy(input_vecpsum.coeffs)

        merged_vecpsum = mergecall(input_vecpsum)

        @test input_vecpsum.terms == reference_terms
        @test input_vecpsum.coeffs == reference_coeffs
        @test merged_vecpsum.terms !== input_vecpsum.terms
    end
end


@testset "thread=false matches thread=true" begin
    nq = 6
    vpsum = VectorPauliSum(get_psum(nq))

    # every public entry point accepts `thread` and gives the same result either way
    @test PauliSum(translationmerge(vpsum; thread=false)) == PauliSum(translationmerge(vpsum; thread=true))
    @test PauliSum(translationmerge(vpsum, 3, 2; thread=false)) == PauliSum(translationmerge(vpsum, 3, 2; thread=true))
    @test PauliSum(reflectionmerge(vpsum; thread=false)) == PauliSum(reflectionmerge(vpsum; thread=true))
    @test PauliSum(reflectionmerge(vpsum, 3, 2; thread=false)) == PauliSum(reflectionmerge(vpsum, 3, 2; thread=true))
    @test PauliSum(reflectionmerge(vpsum, 3, 2; axes=:x, thread=false)) == PauliSum(reflectionmerge(vpsum, 3, 2; axes=:x, thread=true))
    @test PauliSum(permutationmerge(vpsum; thread=false)) == PauliSum(permutationmerge(vpsum; thread=true))

    # in-place versions too
    @test PauliSum(translationmerge!(deepcopy(vpsum); thread=false)) == PauliSum(translationmerge!(deepcopy(vpsum); thread=true))
    @test PauliSum(reflectionmerge!(deepcopy(vpsum); thread=false)) == PauliSum(reflectionmerge!(deepcopy(vpsum); thread=true))
    @test PauliSum(reflectionmerge!(deepcopy(vpsum), 3, 2; thread=false)) == PauliSum(reflectionmerge!(deepcopy(vpsum), 3, 2; thread=true))
    @test PauliSum(permutationmerge!(deepcopy(vpsum); thread=false)) == PauliSum(permutationmerge!(deepcopy(vpsum); thread=true))
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

@testset "Symmetry merging on a propagated cache" begin
    # `propagate!` leaves the cache with a full sorted-prefix marker,
    # and `symmetrymerge!`'s in-place term remapping used to leave it stale, so `merge!`
    # took the sorted-tail fast path and silently skipped deduplication.
    nq = 6
    circuit = [PauliRotation([:X, :X], [1, 2]), PauliRotation([:Y, :Y], [3, 4])]
    thetas = [0.3, 0.7]

    for mergecall! in (permutationmerge!, translationmerge!, reflectionmerge!)
        psum = get_psum(nq)
        prop_cache = PropagationCache(VectorPauliSum(psum))
        propagate!(circuit, prop_cache, thetas; min_abs_coeff=0.0)
        mergecall!(prop_cache)

        # reference through the dict-backed path, which has no cache invariants
        expected = mergecall!(propagate(circuit, psum, thetas; min_abs_coeff=0.0))

        merged_psum = PauliSum(nq)
        for (pstr, coeff) in zip(PauliPropagation.activeterms(prop_cache), PauliPropagation.activecoeffs(prop_cache))
            add!(merged_psum, pstr, coeff)
        end
        @test merged_psum == expected
        # in particular the duplicates were actually merged away
        @test length(prop_cache) == length(expected)
    end
end

@testset "Block-wise permutation merging" begin
    nq = 6
    T = getinttype(nq)
    allbits = (T(1) << (2 * nq)) - T(1)
    blocks = ((1, 2), (3, 3), (4, 6))

    for _ in 1:100
        pstr = rand(T) & allbits
        # a single block is the full permutation canonical form
        @test PauliPropagation._permutationcanonicalform(pstr, ((1, nq),)) == PauliPropagation._permutationcanonicalform(pstr)
        canonical = PauliPropagation._permutationcanonicalform(pstr, blocks)
        # idempotent, and invariant under permutations within the blocks
        @test PauliPropagation._permutationcanonicalform(canonical, blocks) == canonical
        perm = vcat(randperm(2), [3], 3 .+ randperm(3))
        @test PauliPropagation._permutationcanonicalform(getpauli(pstr, perm), blocks) == canonical
        # the site counts per block are preserved
        for (lo, hi) in blocks
            @test countweight(getpauli(canonical, lo:hi)) == countweight(getpauli(pstr, lo:hi))
        end
    end

    # merging: sites 1,2 equivalent, site 3 on its own, sites 4,5,6 equivalent
    psum = PauliSum(nq)
    add!(psum, [:X, :Z], [1, 4])
    add!(psum, [:X, :Z], [2, 5], 0.5)
    add!(psum, [:X, :Z], [3, 6])
    expected = PauliSum(nq)
    add!(expected, [:X, :Z], [1, 4], 1.5)
    add!(expected, [:X, :Z], [3, 4])
    @test permutationmerge(psum, blocks) == expected
    @test PauliSum(permutationmerge(VectorPauliSum(psum), blocks)) == expected
    @test PauliSum(permutationmerge!(VectorPauliSum(psum), blocks)) == expected
    @test PauliSum(permutationmerge!(PropagationCache(VectorPauliSum(psum)), blocks)) == expected
    @test PauliSum(permutationmerge!(VectorPauliSum(psum), blocks; thread=false)) == expected
    @test permutationmerge(psum, ((1, nq),)) == permutationmerge(psum)

    # validation of the block structure
    @test_throws ArgumentError permutationmerge(psum, ((1, 3), (5, 6)))   # gap
    @test_throws ArgumentError permutationmerge(psum, ((1, 4), (3, 6)))   # overlap
    @test_throws ArgumentError permutationmerge(psum, ((1, 5),))          # does not cover
    @test_throws ArgumentError permutationmerge(psum, ((2, 6),))          # does not start at 1

    # residual blocks of the lexicographic all-to-all sweep
    @test residualpermutationblocks(2, 4, 6) == ((1, 1), (2, 2), (3, 4), (5, 6))
    @test residualpermutationblocks(1, 2, 6) == ((1, 0), (1, 1), (2, 2), (3, 6))
    @test residualpermutationblocks(5, 6, 6) == ((1, 4), (5, 5), (6, 6), (7, 6))
    @test_throws ArgumentError residualpermutationblocks(3, 3, 6)
    @test_throws ArgumentError residualpermutationblocks(0, 3, 6)
end

@testset "Residual-subsymmetry sweep is exact" begin
    # a commuting all-to-all block applied gate by gate with a residual merge after every
    # gate must agree with applying the whole block and merging under S_N once
    nq = 6
    psum = PauliSum(nq)
    for i in 1:nq
        add!(psum, :X, i)
    end
    add!(psum, [:Z, :Y], [2, 5], 0.3)
    # the block must be permutation symmetric: the same angle on every pair
    pairs = [(i, j) for i in 1:nq for j in i+1:nq]
    gates = Dict(p => PauliRotation([:Y, :Y], [p...]) for p in pairs)
    thetas = Dict(p => 0.17 for p in pairs)

    plain = PropagationCache(VectorPauliSum(psum))
    propagate!([gates[p] for p in pairs], plain, [thetas[p] for p in pairs]; min_abs_coeff=0.0)
    permutationmerge!(plain)

    residual = PropagationCache(VectorPauliSum(psum))
    for i in 1:nq-1, j in i+1:nq
        propagate!([gates[(i, j)]], residual, [thetas[(i, j)]]; min_abs_coeff=0.0)
        permutationmerge!(residual, residualpermutationblocks(i, j, nq))
    end
    permutationmerge!(residual)

    tosum(c) = (s = PauliSum(nq); for (t, co) in zip(PauliPropagation.activeterms(c), PauliPropagation.activecoeffs(c)); add!(s, t, co); end; s)
    a, b = tosum(plain), tosum(residual)
    # compare coefficient by coefficient over the union of keys: a coefficient that cancels
    # to exactly zero on one path may survive as ~1e-17 on the other (summation order)
    allkeys = union(Set(paulis(a)), Set(paulis(b)))
    @test all(isapprox(getcoeff(a, p), getcoeff(b, p); atol=1e-10) for p in allkeys)
    @test count(p -> abs(getcoeff(a, p)) > 1e-10, allkeys) == count(p -> abs(getcoeff(b, p)) > 1e-10, allkeys)
end
