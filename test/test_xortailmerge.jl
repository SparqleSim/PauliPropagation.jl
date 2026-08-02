using PauliPropagation
using Test
using Random

const PB = PauliPropagation.PropagationBase

# a sorted, deduplicated sum followed by the tail a rotation appends: `pstr ⊻ gate_mask` for every
# term, in the order of the terms they came from
function _rotatedcache(nq, n, gate_mask)
    rng = MersenneTwister(hash((nq, n, gate_mask)))
    TT = PauliPropagation.getinttype(nq)
    pstrs = sort!(collect(Set(rand(rng, TT, 3n) .& ((TT(1) << (2nq)) - TT(1)))))[1:n]

    prop_cache = PropagationCache(VectorPauliSum(nq, pstrs, rand(rng, n)))
    merge!(prop_cache)

    resize!(prop_cache, 2n)
    terms, coeffs = paulis(mainsum(prop_cache)), coefficients(mainsum(prop_cache))
    for ii in 1:n
        terms[n+ii] = terms[ii] ⊻ gate_mask
        coeffs[n+ii] = coeffs[ii] * sin(0.3)
        coeffs[ii] *= cos(0.3)
    end
    PB.setactivesize!(prop_cache, 2n)

    return prop_cache
end

# `xorsortedtailmerge!` only changes how the tail is sorted, so it has to agree with
# `sortedtailmerge!` on the same cache down to the last coefficient
function _agreeswithsortedtailmerge(nq, n, gate_mask, xor_mask, sorted_before, thread)
    xor_cache = _rotatedcache(nq, n, gate_mask)
    PB.xorsortedtailmerge!(xor_cache, xor_mask, sorted_before; thread)

    ref_cache = _rotatedcache(nq, n, gate_mask)
    PB.sortedtailmerge!(ref_cache; thread)

    xor_sum, ref_sum = PB.extractsum!(xor_cache), PB.extractsum!(ref_cache)
    return paulis(xor_sum) == paulis(ref_sum) && coefficients(xor_sum) == coefficients(ref_sum)
end

@testset "xorsortedtailmerge!" begin
    U = PauliPropagation.getinttype(20)
    W = PauliPropagation.getinttype(100)

    gate_masks = [
        (20, U(0b11)),                     # one qubit at the bottom of the string
        (20, U(0b11) << 8),                # one qubit in the middle
        (20, U(0b11) << 38),               # one qubit at the very top
        (20, U(0b11) | (U(0b11) << 30)),   # two qubits far apart, so two groups
        (100, W(0b11) << 126),             # a group straddling the word boundary
        (100, W(0b11) | (W(0b11) << 150)), # two groups, one of them beyond the first word
    ]

    # without this the cases below would compare the fallback against itself and test nothing
    @test all(PB._xorplan(gate_mask, PauliPropagation.getinttype(nq)[]) !== nothing for (nq, gate_mask) in gate_masks)

    # a tail of 50000 is above 3 * _MIN_ELEMS_PER_TASK, where a pass is split across tasks and runs
    # of terms straddle the chunk boundaries
    for (nq, gate_mask) in gate_masks, n in (500, 50000), thread in (false, true)
        @test _agreeswithsortedtailmerge(nq, n, gate_mask, gate_mask, true, thread)
    end

    # falls back to the generic merge, and still merges correctly, when the tail was not appended in
    # parent order, when the mask is spread over more groups than there are passes, and when it is
    # not a term at all
    gate_mask = U(0b11) << 8
    @test _agreeswithsortedtailmerge(20, 500, gate_mask, gate_mask, false, true)
    @test _agreeswithsortedtailmerge(20, 500, gate_mask, U(0b101010101), true, true)
    @test _agreeswithsortedtailmerge(20, 500, gate_mask, nothing, true, true)
end
