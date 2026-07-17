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