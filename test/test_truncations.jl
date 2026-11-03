using Random
using Test

function brickcircuit(seed)
    nq = 8
    nl = 4
    pstr = PauliString(nq, :Z, round(Int, nq / 2))

    topo = bricklayertopology(nq; periodic=false)
    circ = hardwareefficientcircuit(nq, nl; topology=topo)

    Random.seed!(seed)
    thetas = randn(length(circ))

    return circ, pstr, thetas
end

@testset "Truncate damping coefficients Tests" begin
    """Test the truncations by damped coefficients."""
    seed = 42
    circ, pstr, thetas = brickcircuit(seed)

    W = Inf
    min_abs_coeff = 0.0
    evolved_p = propagate(
        circ, pstr, thetas;
        max_weight=W, min_abs_coeff=min_abs_coeff
    )
    expected_expval = overlapwithzero(evolved_p)

    gamma = 0.0
    truncategamma = (pstr, coeff) -> truncatedampingcoeff(
        pstr, coeff, gamma, min_abs_coeff
    )
    evolved_p = propagate(
        circ, pstr, thetas;
        max_weight=W, min_abs_coeff=min_abs_coeff,
        customtruncfunc=truncategamma
    )
    # \gamma=0 == zero dissipation
    @test isapprox(overlapwithzero(evolved_p), expected_expval)

    gamma = 0.01
    min_abs_coeff = 1e-5
    truncategamma = (pstr, coeff) -> truncatedampingcoeff(
        pstr, coeff, gamma, min_abs_coeff
    )
    evolved_p = propagate(
        circ, pstr, thetas;
        max_weight=W, min_abs_coeff=min_abs_coeff,
        customtruncfunc=truncategamma
    )
    # \gamma=0.1 \approx zero dissipation
    #TODO: is there another way to test dissipation?
    @test isapprox(overlapwithzero(evolved_p), expected_expval, rtol=1e-3)

end

@testset "maxabscoeff Tests" begin
    """Test maxabscoeff() on Dict- and Array-based term sums."""
    nq = 4
    paulis = (:X, :Y, :Z)
    qinds = (1, 2, 3)
    coeffs = (0.1, -0.7, 0.3)

    psum = PauliSum(nq)
    for (pauli, qind, coeff) in zip(paulis, qinds, coeffs)
        add!(psum, pauli, qind, coeff)
    end

    @test maxabscoeff(psum) ≈ maximum(abs, coeffs)

    vpsum = VectorPauliSum(psum)
    @test maxabscoeff(vpsum) ≈ maximum(abs, coeffs)

    # complex coefficients: the reduction must stay real-valued internally
    complex_coeffs = (0.1 + 0.2im, -0.7 - 0.3im, 0.3im)

    cpsum = PauliSum(ComplexF64, nq)
    for (pauli, qind, coeff) in zip(paulis, qinds, complex_coeffs)
        add!(cpsum, pauli, qind, coeff)
    end

    @test maxabscoeff(cpsum) ≈ maximum(abs, complex_coeffs)

    cvpsum = VectorPauliSum(cpsum)
    @test maxabscoeff(cvpsum) ≈ maximum(abs, complex_coeffs)
end

@testset "Truncate relative coefficient Tests" begin
    """Test the min_rel_coeff truncation, comparing Dict- and Array-based propagation."""
    seed = 42
    circ, pstr, thetas = brickcircuit(seed)

    min_abs_coeff = 1e-8

    # min_rel_coeff=0.0 does not add any truncation beyond min_abs_coeff
    expected_p = propagate(circ, pstr, thetas; min_abs_coeff=min_abs_coeff)
    zero_rel_p = propagate(circ, pstr, thetas; min_abs_coeff=min_abs_coeff, min_rel_coeff=0.0)
    @test length(zero_rel_p) == length(expected_p)
    @test isapprox(overlapwithzero(zero_rel_p), overlapwithzero(expected_p))

    # increasing min_rel_coeff can only truncate more aggressively,
    # and Dict- and Array-based propagation must agree
    prev_nterms = length(expected_p)
    for min_rel_coeff in (1e-4, 1e-3, 1e-2)
        dnum = propagate(circ, pstr, thetas; min_abs_coeff=min_abs_coeff, min_rel_coeff=min_rel_coeff)
        dvec = propagate(circ, VectorPauliSum(pstr), thetas; min_abs_coeff=min_abs_coeff, min_rel_coeff=min_rel_coeff)

        @test length(dnum) == length(dvec)
        @test isapprox(overlapwithzero(dnum), overlapwithzero(dvec))

        @test length(dnum) <= prev_nterms
        prev_nterms = length(dnum)
    end
end

@testset "buildtruncfunc" begin
    psum = PauliSum(1)
    add!(psum, :X, 1, 1.0)
    add!(psum, :Y, 1, 0.1)

    truncfunc = buildtruncfunc(psum; min_abs_coeff=0.0, min_rel_coeff=0.2)
    @test !truncfunc(symboltoint(UInt8, :X, 1), 1.0)
    @test truncfunc(symboltoint(UInt8, :Y, 1), 0.1)
end

@testset "Top-K truncation Tests" begin
    """Test max_terms, whose threshold is found by radix select on the coefficient magnitudes, never by sorting."""
    nq = 20

    # well-separated magnitudes land in buckets of their own, so every tolerance is exact
    psum = PauliSum(nq)
    for qind in 1:nq
        add!(psum, :Z, qind, 2.0^(1 - qind))
    end
    @test topkabscoeff(psum, 5) == 2.0^-4
    @test topkabscoeff(VectorPauliSum(psum), 5) == 2.0^-4
    @test topkabscoeff(psum, nq) == 0.0
    @test_throws ArgumentError topkabscoeff(psum, 0)
    @test_throws ArgumentError topkabscoeff(psum, 5; rtol=-0.1)
    for pauli_sum in (psum, VectorPauliSum(psum))
        truncated = truncate(pauli_sum; min_abs_coeff=0.0, max_terms=5)
        @test length(truncated) == 5
        @test sort(collect(PB.coefficients(truncated)); rev=true) == 2.0 .^ (0:-1:-4)
        @test length(truncate(pauli_sum; min_abs_coeff=0.0, max_terms=nq)) == nq
    end

    # a smooth spectrum: the kept count lands within rtol of K, and the kept terms are the largest ones
    rng = MersenneTwister(1)
    n = 20_000
    K = 2000
    magnitudes = exp.(randn(rng, n))
    vpsum = VectorPauliSum(nq, UInt64.(1:n), magnitudes .* rand(rng, (-1, 1), n))
    @test topkabscoeff(vpsum, K; thread=true) == topkabscoeff(vpsum, K; thread=false)
    for rtol in (0.05, 0.01, 0.001, 0.0)
        truncated = truncate(vpsum; min_abs_coeff=0.0, max_terms=K, max_terms_rtol=rtol)
        n_kept = length(truncated)
        @test abs(n_kept - K) <= rtol * K
        @test minimum(abs, PB.coefficients(truncated)) >= partialsort(magnitudes, n_kept; rev=true)
    end

    # tied magnitudes are kept or dropped together, whichever count is closer to K
    tied = VectorPauliSum(nq, UInt64.(1:200), [fill(1.0, 100); fill(-0.5, 100)])
    @test length(truncate(tied; min_abs_coeff=0.0, max_terms=150, max_terms_rtol=0.0)) == 200
    @test length(truncate(tied; min_abs_coeff=0.0, max_terms=120, max_terms_rtol=0.0)) == 100

    # during propagation, Dict- and Array-based propagation must agree
    circ, pstr, thetas = brickcircuit(42)
    n_untruncated = length(propagate(circ, pstr, thetas; min_abs_coeff=0.0))
    for max_terms in (50, 200)
        dnum = propagate(circ, pstr, thetas; min_abs_coeff=0.0, max_terms)
        dvec = propagate(circ, VectorPauliSum(pstr), thetas; min_abs_coeff=0.0, max_terms)

        @test length(dnum) == length(dvec)
        @test isapprox(overlapwithzero(dnum), overlapwithzero(dvec))
        @test length(dnum) <= 1.05 * max_terms + 1
        @test length(dnum) < n_untruncated
    end
end
