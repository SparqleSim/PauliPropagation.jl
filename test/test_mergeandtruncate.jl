using PauliPropagation
using Test
using Random

const MAT_PB = PauliPropagation.PropagationBase

# Terms 1 and 2 demonstrate why truncation has to follow merging:
# 0.60 - 0.55 cancels below the threshold, while 0.06 + 0.06 rises above it.
const MAT_TERMS = UInt8[0x01, 0x01, 0x02, 0x02, 0x03]
const MAT_COEFFS = Float64[0.60, -0.55, 0.06, 0.06, 1.0]
const MAT_THRESHOLD = 0.10

mat_truncfunc(term, coefficient) = abs(coefficient) < MAT_THRESHOLD

function mat_vector_sum()
    return VectorPauliSum(2, copy(MAT_TERMS), copy(MAT_COEFFS))
end

function mat_expected_sum()
    return PauliSum(2, Dict{UInt8,Float64}(0x02 => 0.12, 0x03 => 1.0))
end

@testset "mergeandtruncate!" begin
    expected = mat_expected_sum()

    @testset "term sums" begin
        # Dictionary sums are already merged, while vector sums can carry duplicate terms.
        dict_sum = PauliSum(mat_vector_sum())
        vector_sum = mat_vector_sum()
        multi_dict_sum = MultiPauliSum(deepcopy(dict_sum), 2)
        multi_vector_sum = MultiPauliSum(mat_vector_sum(), 2)

        for term_sum in (dict_sum, vector_sum, multi_dict_sum, multi_vector_sum)
            returned = MAT_PB.mergeandtruncate!(mat_truncfunc, term_sum; thread=false)
            @test returned === term_sum
            @test PauliSum(term_sum) ≈ expected
        end
    end

    @testset "propagation caches" begin
        for term_sum in (
            PauliSum(mat_vector_sum()),
            mat_vector_sum(),
            MultiPauliSum(PauliSum(mat_vector_sum()), 2),
            MultiPauliSum(mat_vector_sum(), 2),
        )
            prop_cache = PropagationCache(term_sum)
            returned = MAT_PB.mergeandtruncate!(mat_truncfunc, prop_cache; thread=false)
            @test returned === prop_cache
            @test PauliSum(MAT_PB.activesum(prop_cache)) ≈ expected
        end

        # A dictionary cache can hold colliding contributions across its main and auxiliary sums.
        dict_cache = PropagationCache(PauliSum(2, Dict{UInt8,Float64}(0x01 => 0.60, 0x02 => 0.06)))
        add!(MAT_PB.auxsum(dict_cache), 0x01, -0.55)
        add!(MAT_PB.auxsum(dict_cache), 0x02, 0.06)
        add!(MAT_PB.auxsum(dict_cache), 0x03, 1.0)
        MAT_PB.mergeandtruncate!(mat_truncfunc, dict_cache; thread=false)
        @test MAT_PB.activesum(dict_cache) ≈ expected
    end

    @testset "already merged and empty inputs" begin
        # Even with no tail to merge, a preceding operation may have made a coefficient truncatable.
        sorted_sum = VectorPauliSum(2, UInt8[0x01, 0x02], [0.01, 1.0], 2)
        sorted_cache = PropagationCache(sorted_sum)
        MAT_PB.mergeandtruncate!(mat_truncfunc, sorted_cache; thread=false)
        @test MAT_PB.sortedprefix(MAT_PB.mainsum(sorted_cache)) == 1
        @test PauliSum(MAT_PB.activesum(sorted_cache)) == PauliSum(2, Dict{UInt8,Float64}(0x02 => 1.0))

        for empty_sum in (
            PauliSum(2),
            VectorPauliSum(2),
            MultiPauliSum(PauliSum(2), 2),
            MultiPauliSum(VectorPauliSum(2), 2),
        )
            @test isempty(MAT_PB.mergeandtruncate!(mat_truncfunc, empty_sum; thread=false))
        end
    end

    @testset "parallel grouped reduction" begin
        n_groups = 20_000
        grouped_terms = repeat(UInt32.(1:n_groups), inner=2)
        grouped_coefficients = Vector{Float64}(undef, 2n_groups)
        for group in 1:n_groups
            if isodd(group)
                grouped_coefficients[2group-1] = 0.06
                grouped_coefficients[2group] = 0.06
            else
                grouped_coefficients[2group-1] = 0.60
                grouped_coefficients[2group] = -0.55
            end
        end

        grouped_cache = PropagationCache(VectorPauliSum(16, grouped_terms, grouped_coefficients))
        MAT_PB.mergeandtruncate!(mat_truncfunc, grouped_cache; thread=true)

        @test MAT_PB.terms(grouped_cache) == UInt32.(1:2:n_groups)
        @test all(coefficient -> coefficient ≈ 0.12, coefficients(grouped_cache))
        @test MAT_PB.sortedprefix(MAT_PB.mainsum(grouped_cache)) == length(grouped_cache)
    end

    @testset "XOR merge integration" begin
        xor_mask = UInt8(0x03)

        # No appended tail must still run truncation over the settled head.
        tail_cache = PropagationCache(VectorPauliSum(2, UInt8[0x01], [0.01], 1))
        MAT_PB.xormergeandtruncate!(mat_truncfunc, tail_cache, xor_mask; thread=false)
        @test isempty(tail_cache)

        sorted_tail_cache = PropagationCache(VectorPauliSum(2, UInt8[0x01], [0.01], 1))
        MAT_PB._sortedtailmergeandtruncate!(mat_truncfunc, sorted_tail_cache; thread=false)
        @test isempty(sorted_tail_cache)

        box_cache = PropagationCache(VectorPauliSum(2, UInt8[0x01], [0.01], 1))
        empty_box = VectorPauliSum(2)
        MAT_PB._xormergeandtruncatebox!(MAT_PB.ArrayStorage(), mat_truncfunc, box_cache, empty_box, xor_mask)
        @test isempty(box_cache)
        @test isempty(empty_box)

        # A branched multi sum merges and truncates through the boxes it left behind.
        multi_cache = PropagationCache(MultiPauliSum(VectorPauliSum(2, UInt8[0x01, 0x02], [0.5, 0.06], 2), 2))
        branch(term, coefficient) = MAT_PB.Branch(coefficient, coefficient)
        MAT_PB.xorbranch!(branch, multi_cache, xor_mask; thread=false)
        MAT_PB.xormergeandtruncate!(mat_truncfunc, multi_cache, xor_mask; thread=false)
        @test PauliSum(MAT_PB.activesum(multi_cache)) ≈ PauliSum(2, Dict{UInt8,Float64}(0x01 => 0.56, 0x02 => 0.56))
    end
end

# The tail merge `_mergeandtruncatebysorting!` takes elsewhere than the CPU is written in array
# kernels, so it runs on `Vector`s when called directly and can be held against the CPU route
# here. The two have to agree bit for bit: both add the contributions of a term in the order they
# are stored, head before tail.

# a sorted, duplicate-free head of `n` terms followed by the tail a rotation appends, `term ⊻ gate_mask`
# of every head term, so that a tail term repeats a head term whenever both images are in the head
function mat_rotatedcache(nq, n, gate_mask)
    rng = MersenneTwister(hash((nq, n, gate_mask)))
    TT = PauliPropagation.getinttype(nq)
    pstrs = sort!(collect(Set(rand(rng, TT, 3n) .& ((TT(1) << (2nq)) - TT(1)))))[1:n]

    prop_cache = PropagationCache(VectorPauliSum(nq, pstrs, rand(rng, n)))
    merge!(prop_cache)

    resize!(prop_cache, 2n)
    terms, coeffs = paulis(MAT_PB.mainsum(prop_cache)), coefficients(MAT_PB.mainsum(prop_cache))
    for ii in 1:n
        terms[n+ii] = terms[ii] ⊻ gate_mask
        coeffs[n+ii] = coeffs[ii] * sin(0.3)
        coeffs[ii] *= cos(0.3)
    end
    MAT_PB.setactivesize!(prop_cache, 2n)

    return prop_cache
end

# a sorted, duplicate-free head followed by a tail given explicitly, which may repeat head terms
function mat_headtailcache(nq, head_terms, head_coeffs, tail_terms, tail_coeffs)
    n, n_tail = length(head_terms), length(tail_terms)
    prop_cache = PropagationCache(VectorPauliSum(nq, copy(head_terms), copy(head_coeffs), n))
    resize!(prop_cache, n + n_tail)
    paulis(MAT_PB.mainsum(prop_cache))[n+1:n+n_tail] .= tail_terms
    coefficients(MAT_PB.mainsum(prop_cache))[n+1:n+n_tail] .= tail_coeffs
    MAT_PB.setactivesize!(prop_cache, n + n_tail)
    return prop_cache
end

# the array route of `_mergeandtruncatebysorting!`, taken on a `Vector` cache although it is on the CPU
function mat_arraytailmergeandtruncate!(truncfunc, prop_cache; thread)
    n_sorted, n_total = MAT_PB._sortedandactive(prop_cache)
    MAT_PB._ranktailmerge!(prop_cache, n_sorted, n_total; thread)
    MAT_PB._deduplicateflagged!(prop_cache; thread)
    MAT_PB.setsortedprefix!(MAT_PB.mainsum(prop_cache), MAT_PB.activesize(prop_cache))
    return MAT_PB._truncate!(truncfunc, prop_cache; thread)
end

function mat_activepairs(prop_cache)
    term_sum = MAT_PB.extractsum!(prop_cache)
    return paulis(term_sum), coefficients(term_sum)
end

@testset "array tail merge agrees with the CPU route" begin
    U = PauliPropagation.getinttype(20)
    W = PauliPropagation.getinttype(100)

    # a tail of 50000 is above 3 * _MIN_ELEMS_PER_TASK, where the kernels are split across tasks
    for (nq, gate_mask) in ((20, U(0b11) << 8), (20, U(0b11) | (U(0b11) << 30)), (100, W(0b11) << 126)),
        n in (500, 50000), thread in (false, true)

        ref_cache = mat_rotatedcache(nq, n, gate_mask)
        MAT_PB.mergeandtruncate!(mat_truncfunc, ref_cache; thread)

        array_cache = mat_rotatedcache(nq, n, gate_mask)
        returned = mat_arraytailmergeandtruncate!(mat_truncfunc, array_cache; thread)
        @test returned === array_cache
        @test MAT_PB.sortedprefix(MAT_PB.mainsum(array_cache)) == MAT_PB.activesize(array_cache)

        ref_terms, ref_coeffs = mat_activepairs(ref_cache)
        array_terms, array_coeffs = mat_activepairs(array_cache)
        @test array_terms == ref_terms
        @test array_coeffs == ref_coeffs
    end

    # ties across the boundary: the tail repeats head terms in a shuffled order, with the fixture's
    # coefficients so that merging both cancels a term below the threshold and lifts a pair above it
    rng = MersenneTwister(2024)
    head_terms = sort!(collect(Set(rand(rng, UInt32(0):UInt32(2^20 - 1), 3000))))[1:1000]
    head_coeffs = fill(0.60, 1000)
    tail_terms = shuffle(rng, head_terms)[1:600]
    tail_coeffs = [isodd(ii) ? -0.55 : 0.06 for ii in 1:600]
    head_coeffs[findfirst(==(tail_terms[2]), head_terms)] = 0.06

    for thread in (false, true)
        ref_cache = mat_headtailcache(20, head_terms, head_coeffs, tail_terms, tail_coeffs)
        MAT_PB.mergeandtruncate!(mat_truncfunc, ref_cache; thread)

        array_cache = mat_headtailcache(20, head_terms, head_coeffs, tail_terms, tail_coeffs)
        mat_arraytailmergeandtruncate!(mat_truncfunc, array_cache; thread)
        @test MAT_PB.sortedprefix(MAT_PB.mainsum(array_cache)) == MAT_PB.activesize(array_cache)
        @test issorted(MAT_PB.terms(array_cache))

        ref_terms, ref_coeffs = mat_activepairs(ref_cache)
        array_terms, array_coeffs = mat_activepairs(array_cache)
        @test array_terms == ref_terms
        @test array_coeffs == ref_coeffs
    end
end

@testset "array tail merge on its own" begin
    rng = MersenneTwister(2025)
    randomterms(lo, hi, n) = rand(rng, UInt32(lo):UInt32(hi), n)
    sortedhead(lo, hi, n) = sort!(collect(Set(randomterms(lo, hi, 3n))))[1:n]

    # a tail entirely below the head, entirely above it, equal in every term, and four times as long
    cases = [
        (sortedhead(2^20, 2^21 - 1, 400), randomterms(0, 2^19 - 1, 100)),
        (sortedhead(0, 2^19 - 1, 400), randomterms(2^20, 2^21 - 1, 100)),
        ([UInt32(77)], fill(UInt32(77), 200)),
        (sortedhead(0, 2^20 - 1, 100), randomterms(0, 2^20 - 1, 400)),
    ]

    for (head_terms, tail_terms) in cases, thread in (false, true)
        n, n_tail = length(head_terms), length(tail_terms)
        n_total = n + n_tail
        head_coeffs, tail_coeffs = rand(rng, n), rand(rng, n_tail)
        prop_cache = mat_headtailcache(20, head_terms, head_coeffs, tail_terms, tail_coeffs)
        input_terms, input_coeffs = copy(MAT_PB.terms(prop_cache)), copy(coefficients(prop_cache))

        MAT_PB._ranktailmerge!(prop_cache, n, n_total; thread)

        # the merge alone leaves the active range sorted, with the same terms and coefficients
        @test MAT_PB.activesize(prop_cache) == n_total
        merged_terms, merged_coeffs = MAT_PB.terms(prop_cache), coefficients(prop_cache)
        @test issorted(merged_terms)
        @test sort(collect(zip(merged_terms, merged_coeffs))) == sort(collect(zip(input_terms, input_coeffs)))

        # the sorted tail is parked above the active range of the output, and the input arrays,
        # now the auxiliary ones, are untouched
        @test length(MAT_PB.mainsum(prop_cache)) >= n_total + n_tail
        @test paulis(MAT_PB.mainsum(prop_cache))[n_total+1:n_total+n_tail] == sort(tail_terms)
        @test paulis(MAT_PB.auxsum(prop_cache))[1:n_total] == input_terms
        @test coefficients(MAT_PB.auxsum(prop_cache))[1:n_total] == input_coeffs
    end

    # a head term goes before an equal tail term
    prop_cache = mat_headtailcache(20, UInt32[1, 2, 3], [1.0, 2.0, 3.0], UInt32[3, 1, 2], [30.0, 10.0, 20.0])
    MAT_PB._ranktailmerge!(prop_cache, 3, 6; thread=false)
    @test MAT_PB.terms(prop_cache) == UInt32[1, 1, 2, 2, 3, 3]
    @test coefficients(prop_cache) == [1.0, 10.0, 2.0, 20.0, 3.0, 30.0]

    # the gate at its boundaries: an empty head, no tail, the smallest array, the longest tail
    min_total = MAT_PB._RANKTAILMERGE_MIN_TOTAL
    @test !MAT_PB._ranktailmergepays(0, min_total)
    @test !MAT_PB._ranktailmergepays(min_total, min_total)
    @test !MAT_PB._ranktailmergepays(min_total ÷ 2, min_total - 1)
    @test MAT_PB._ranktailmergepays(min_total ÷ 2, min_total)
    @test !MAT_PB._ranktailmergepays(min_total ÷ 2 - 1, min_total)
    @test MAT_PB._ranktailmergepays(min_total, 2min_total)
end
