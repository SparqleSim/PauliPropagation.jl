using PauliPropagation
using Test

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
        MAT_PB._sortedtailmerge!(mat_truncfunc, sorted_tail_cache; thread=false)
        @test isempty(sorted_tail_cache)

        box_cache = PropagationCache(VectorPauliSum(2, UInt8[0x01], [0.01], 1))
        empty_box = VectorPauliSum(2)
        MAT_PB._xorsortedboxmerge!(mat_truncfunc, box_cache, empty_box, xor_mask; thread=false)
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
