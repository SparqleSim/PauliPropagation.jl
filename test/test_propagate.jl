using Random


@testset "Test propagation with known value" begin

    # this has been cross-validated with other libraries
    expected_value = 0.21720058439757214

    nq = 8
    nl = 4
    W = Inf
    min_abs_coeff = 0.0
    max_freq = Inf

    # Dict-based PauliSum propagation
    pstr = PauliString(nq, :Z, round(Int, nq / 2))

    topo = bricklayertopology(nq; periodic=false)
    circ = hardwareefficientcircuit(nq, nl; topology=topo)

    m = countparameters(circ)

    Random.seed!(42)
    thetas = randn(m)

    dnum = propagate(circ, pstr, thetas; max_weight=W, min_abs_coeff=min_abs_coeff)

    @test overlapwithzero(dnum) ≈ expected_value

    # Vector-based VectorPauliSum propagation

    dvec = propagate(circ, VectorPauliSum(pstr), thetas; max_weight=W, min_abs_coeff=min_abs_coeff)

    @test overlapwithzero(dvec) ≈ expected_value



    ## Hybrid propagation with PathProperties
    pstr = PauliString(nq, :Z, round(Int, nq / 2))

    wrapped_pstr = wrapcoefficients(pstr, PauliFreqTracker)

    topo = bricklayertopology(nq; periodic=false)
    circ = hardwareefficientcircuit(nq, nl; topology=topo)

    m = countparameters(circ)

    Random.seed!(42)
    thetas = randn(m)

    dhyb = propagate(circ, wrapped_pstr, thetas; max_weight=W, max_freq=max_freq, min_abs_coeff=min_abs_coeff)

    @test overlapwithzero(dhyb) ≈ expected_value


    ## Surrogate propagation with NodePathProperties
    pstr = PauliString(nq, :Z, round(Int, nq / 2))

    wrapped_pstr = wrapcoefficients(pstr, NodePathProperties)

    topo = bricklayertopology(nq; periodic=false)
    circ = hardwareefficientcircuit(nq, nl; topology=topo)

    m = countparameters(circ)

    Random.seed!(42)
    thetas = randn(m)

    dsym = propagate(circ, wrapped_pstr; max_weight=W, max_freq=max_freq)
    zerofilter!(dsym)  # Filter the nodes that you find relevant
    evaluate!(dsym, thetas)

    @test overlapwithzero(dsym) ≈ expected_value
end


@testset "Test propagate equivalence" begin
    nq = 6
    nl = 3

    topology = rectangletopology(2, 3; periodic=true)
    circ = efficientsu2circuit(nq, nl; topology=topology)

    nparams = countparameters(circ)

    ## Test weight truncation
    for max_weight in (0, 1, 3, 6)
        thetas = randn(nparams)

        pstr = PauliString(nq, rand([:I, :X, :Y, :Z]), rand(1:nq))
        dnum = propagate(circ, pstr, thetas; min_abs_coeff=0, max_weight=max_weight)

        dvec = propagate(circ, VectorPauliSum(pstr), thetas; min_abs_coeff=0, max_weight=max_weight)

        wrapped_pstr = wrapcoefficients(pstr, PauliFreqTracker)
        dhyb = propagate(circ, wrapped_pstr, thetas; min_abs_coeff=0, max_weight=max_weight)

        surrogate_pstr = wrapcoefficients(pstr, NodePathProperties)
        dsym = propagate(circ, surrogate_pstr; max_weight=max_weight)
        evaluate!(dsym, thetas)

        @test overlapwithzero(dnum) ≈ overlapwithzero(dvec) ≈ overlapwithzero(dhyb) ≈ overlapwithzero(dsym)
        @test overlapwithplus(dnum) ≈ overlapwithplus(dvec) ≈ overlapwithplus(dhyb) ≈ overlapwithplus(dsym)
    end

    # Test frequency truncation
    for max_freq in (0, 2, 5, 10)
        thetas = randn(nparams)

        pstr = PauliString(nq, rand([:I, :X, :Y, :Z]), rand(1:nq))

        wrapped_pstr = wrapcoefficients(pstr, PauliFreqTracker)
        dhyb = propagate(circ, wrapped_pstr, thetas; min_abs_coeff=0, max_freq=max_freq)

        surrogate_pstr = wrapcoefficients(pstr, NodePathProperties)
        dsym = propagate(circ, surrogate_pstr; max_freq=max_freq)
        evaluate!(dsym, thetas)

        @test overlapwithzero(dhyb) ≈ overlapwithzero(dsym)
        @test overlapwithplus(dhyb) ≈ overlapwithplus(dsym)
    end

    # Test max sins/ small-angle truncation
    for max_sins in (0, 2, 5, 10)
        thetas = randn(nparams)

        pstr = PauliString(nq, rand([:I, :X, :Y, :Z]), rand(1:nq))

        wrapped_pstr = wrapcoefficients(pstr, PauliFreqTracker)
        dhyb = propagate(circ, wrapped_pstr, thetas; min_abs_coeff=0, max_sins=max_sins)

        surrogate_pstr = wrapcoefficients(pstr, NodePathProperties)
        dsym = propagate(circ, surrogate_pstr; max_sins=max_sins)
        evaluate!(dsym, thetas)

        @test overlapwithzero(dhyb) ≈ overlapwithzero(dsym)
        @test overlapwithplus(dhyb) ≈ overlapwithplus(dsym)
    end


    # Test min_abs_coeff truncation
    nl = 5
    circ = efficientsu2circuit(nq, nl; topology=topology)
    nparams = countparameters(circ)

    for min_abs_coeff in (1e-3, 1e-2, 1e-1)
        thetas = randn(nparams)

        pstr = PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq))
        dnum = propagate(circ, pstr, thetas; min_abs_coeff=min_abs_coeff)

        dvec = propagate(circ, VectorPauliSum(pstr), thetas; min_abs_coeff=min_abs_coeff)

        wrapped_pstr = wrapcoefficients(pstr, PauliFreqTracker)
        dhyb = propagate(circ, wrapped_pstr, thetas; min_abs_coeff=min_abs_coeff)

        @test overlapwithzero(dnum) ≈ overlapwithzero(dvec) ≈ overlapwithzero(dhyb)
        @test overlapwithplus(dnum) ≈ overlapwithplus(dvec) ≈ overlapwithplus(dhyb)
    end

end

@testset "Test improper truncations" begin

    pstr = PauliString(4, :Z, 2)
    gate = PauliRotation(:X, 2, randn())

    @test_throws ArgumentError propagate!(gate, pstr; max_freq=rand(1:10))
    @test_throws ArgumentError propagate!(gate, pstr; max_sins=rand(1:10))
    @test_throws ArgumentError propagate!(gate, pstr; max_freq=rand(1:10), max_sins=rand(1:10))

    # a PathProperties type that does not track freq or sins
    struct MyPathProperties <: PathProperties
        coeff::Float64
    end

    wpstr = wrapcoefficients(pstr, MyPathProperties)
    @test_throws ArgumentError propagate(gate, wpstr; max_freq=rand(1:10))
    @test_throws ArgumentError propagate(gate, wpstr; max_sins=rand(1:10))
    @test_throws ArgumentError propagate(gate, wpstr; max_freq=rand(1:10), max_sins=rand(1:10))

end

@testset "Test automatic conversion" begin
    # when `max_freq` or `max_sins` are specified and the coefficients are not PathProperties
    # then we automatically convert the coefficients to PauliFreqTracker
    # this should yield the same results as the manual conversion

    nq = 4
    nl = 4
    pstr = PauliString(nq, :Z, 2)
    circuit = tfitrottercircuit(nq, nl)
    thetas = randn(countparameters(circuit))

    out_psum = propagate(circuit, pstr, thetas; max_freq=4, max_sins=2)

    # convert to PathProperties
    wpstr = wrapcoefficients(pstr, PauliFreqTracker)
    out_psum2 = propagate(circuit, wpstr, thetas; max_freq=4, max_sins=2)
    upwraped_psum = unwrapcoefficients(out_psum2)
    @test out_psum ≈ upwraped_psum

    # same with initial PauliSum
    psum = PauliSum(pstr)
    out_psum3 = propagate(circuit, psum, thetas; max_freq=4, max_sins=2)
    @test out_psum3 ≈ upwraped_psum

end
