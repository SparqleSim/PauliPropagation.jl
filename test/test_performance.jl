using Random
using Test
using LinearAlgebra

using PauliPropagation.Performance

@testset "Performance module is opt-in and does not change stock propagate" begin
    # `propagate`/`propagate!` with `fused=false` should exactly replicate main library behavior. 
    nq = 6
    nl = 3
    topo = bricklayertopology(nq; periodic=false)
    circuit = hardwareefficientcircuit(nq, nl; topology=topo)

    Random.seed!(1)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 3)
    min_abs_coeff = 1e-4

    stock_dict = propagate(circuit, pstr, thetas; min_abs_coeff)
    stock_vec = propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff)

    @test Performance.propagate(circuit, pstr, thetas; min_abs_coeff, fused=false) == stock_dict
    @test Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff, fused=false) == stock_vec

    # calling propagate with fused=true elsewhere must not retroactively change what a plain,
    # fused-less propagate call returns
    Performance.propagate(circuit, pstr, thetas; min_abs_coeff, fused=true)
    @test propagate(circuit, pstr, thetas; min_abs_coeff) == stock_dict
end

@testset "Performance.mcpropagate matches Performance.propagate exactly below the resampling threshold" begin
    # With max_size effectively infinite, applymergetruncateresample! never resamples, so
    # mcpropagate's per-gate step reduces to the same applymergetruncate! call propagate uses --
    nq = 6
    nl = 3
    topo = bricklayertopology(nq; periodic=false)
    circuit = hardwareefficientcircuit(nq, nl; topology=topo)

    Random.seed!(1)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 3)
    min_abs_coeff = 1e-4

    fused_vec = Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff, fused=true)
    mc_vec = Performance.mcpropagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff, fused=true, max_size=10^9)

    @test length(mc_vec) == length(fused_vec)
    for (term, coeff) in zip(paulis(mc_vec), coefficients(mc_vec))
        @test coeff == getcoeff(fused_vec, term)
    end
end

@testset "fused Dict, fused Vector and stock propagation agree exactly without coefficient truncation" begin
    # truncation by max_weight should not affect any results during propagation
    nq = 6
    topo = rectangletopology(2, 3; periodic=true)

    for nl in (1, 3, 5)
        circuit = efficientsu2circuit(nq, nl; topology=topo)
        Random.seed!(10 + nl)
        thetas = randn(countparameters(circuit))
        pstr = PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq))

        for max_weight in (2, 4, nq)
            stock = propagate(circuit, pstr, thetas; min_abs_coeff=0.0, max_weight)
            dict_fused = Performance.propagate(circuit, pstr, thetas; min_abs_coeff=0.0, max_weight, fused=true)
            vec_fused_sum = Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff=0.0, max_weight, fused=true)
            vec_fused = PauliSum(nq, Dict(zip(paulis(vec_fused_sum), coefficients(vec_fused_sum))))

            @test dict_fused == stock
            @test vec_fused == stock
        end
    end
end

@testset "fused Vector on wide Pauli strings matches stock exactly" begin
    # from 96 qubits on, a Pauli string is wider than a machine word and the fused rotations read
    # only the bytes a gate touches. The long-range rotation is too spread out for that, so it also
    # covers the fall-back to the whole string.
    nq = 100
    topo = bricklayertopology(nq; periodic=false)

    for nl in (2, 3)
        circuit = hardwareefficientcircuit(nq, nl; topology=topo)
        push!(circuit, PauliRotation([:X, :Y, :Z], [1, 40, 90]))

        Random.seed!(30 + nl)
        thetas = randn(countparameters(circuit))
        pstr = PauliString(nq, :Z, 50)

        for max_weight in (3, 4)
            stock = propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff=0.0, max_weight)
            fused = Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff=0.0, max_weight, fused=true)
            unthreaded = Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff=0.0, max_weight, fused=true, thread=false)

            stock_dict = Dict(zip(paulis(stock), coefficients(stock)))
            fused_dict = Dict(zip(paulis(fused), coefficients(fused)))
            @test stock_dict == fused_dict
            @test fused == unthreaded
        end
    end
end

@testset "fused Dict and fused Vector agree with each other and with stock within a small tolerance under coefficient truncation" begin
    # min_abs_coeff makes propagation results differ, but it should not be by much.
    nq = 8
    topo = bricklayertopology(nq; periodic=false)
    tol = 3e-2

    for nl in (3, 5)
        circuit = hardwareefficientcircuit(nq, nl; topology=topo)

        Random.seed!(42)
        thetas = randn(countparameters(circuit))
        pstr = PauliString(nq, :Z, 3)
        min_abs_coeff = 1e-4

        stock = propagate(circuit, pstr, thetas; min_abs_coeff)
        dict_fused = Performance.propagate(circuit, pstr, thetas; min_abs_coeff, fused=true)
        vec_fused_sum = Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff, fused=true)
        vec_fused = PauliSum(nq, Dict(zip(paulis(vec_fused_sum), coefficients(vec_fused_sum))))

        relnorm(a, b) = norm(a - b) / norm(a)

        @test relnorm(stock, dict_fused) < tol
        @test relnorm(stock, vec_fused) < tol
        @test relnorm(dict_fused, vec_fused) < tol

        @test isapprox(overlapwithzero(stock), overlapwithzero(dict_fused); atol=tol)
        @test isapprox(overlapwithzero(stock), overlapwithzero(vec_fused); atol=tol)
    end
end

function _randomimaginarycircuit(nq, ngates; seed)
    Random.seed!(seed)
    symbs = [:X, :Y, :Z]
    gates = ImaginaryPauliRotation[]
    for _ in 1:ngates
        support = rand(1:min(3, nq))
        gate_generator = rand(symbs, support)
        gate_inds = shuffle(1:nq)[1:support]
        push!(gates, ImaginaryPauliRotation(gate_generator, gate_inds))
    end
    taus = randn(ngates) .* 0.2
    return gates, taus
end

@testset "fused Vector ImaginaryPauliRotation matches stock exactly without coefficient truncation" begin
    # truncation by max_weight should not affect any results during propagation, same as for PauliRotation.
    # normalize_coeffs assumes a density-matrix-like start (trace 1), so start from the identity string.
    nq = 6

    for nl in (1, 3, 5)
        gates, taus = _randomimaginarycircuit(nq, nl * nq; seed=20 + nl)
        pstr = PauliString(nq, :I, 1)

        for max_weight in (2, 4, nq)
            stock = propagate(gates, VectorPauliSum(pstr), taus; heisenberg=false, min_abs_coeff=0.0, max_weight)
            fused_sum = Performance.propagate(gates, VectorPauliSum(pstr), taus; heisenberg=false, min_abs_coeff=0.0, max_weight, fused=true)

            stock_dict = Dict(zip(paulis(stock), coefficients(stock)))
            fused_dict = Dict(zip(paulis(fused_sum), coefficients(fused_sum)))
            @test stock_dict == fused_dict
        end
    end
end

@testset "fused Vector ImaginaryPauliRotation agrees with stock within a small tolerance under coefficient truncation" begin
    # min_abs_coeff makes propagation results differ, but it should not be by much.
    nq = 8
    tol = 3e-2

    for nl in (3, 5)
        gates, taus = _randomimaginarycircuit(nq, nl * nq; seed=42)
        pstr = PauliString(nq, :I, 1)
        min_abs_coeff = 1e-4

        stock = propagate(gates, VectorPauliSum(pstr), taus; heisenberg=false, min_abs_coeff)
        fused_sum = Performance.propagate(gates, VectorPauliSum(pstr), taus; heisenberg=false, min_abs_coeff, fused=true)

        stock_dict = Dict(zip(paulis(stock), coefficients(stock)))
        fused_dict = Dict(zip(paulis(fused_sum), coefficients(fused_sum)))
        allkeys = union(keys(stock_dict), keys(fused_dict))
        stock_vec = [get(stock_dict, k, 0.0) for k in allkeys]
        fused_vec = [get(fused_dict, k, 0.0) for k in allkeys]

        @test norm(stock_vec - fused_vec) / norm(stock_vec) < tol
    end
end

function _localimaginarycircuit(nq, nlayers; topology=bricklayertopology(nq; periodic=false), seed)
    Random.seed!(seed)
    gates = ImaginaryPauliRotation[]
    for _ in 1:nlayers
        for ii in 1:nq
            push!(gates, ImaginaryPauliRotation(:X, ii))
            push!(gates, ImaginaryPauliRotation(:Z, ii))
        end
        for pair in topology
            push!(gates, ImaginaryPauliRotation([:Y, :Y], pair))
        end
    end
    taus = randn(length(gates)) .* 0.3
    return gates, taus
end

@testset "fused Vector ImaginaryPauliRotation: thread=false matches thread=true on a multi-task-sized propagation" begin
    # Unlike PauliRotation, ImaginaryPauliRotation branches on *commutation*, so a densely-connected
    # random circuit (as in _randomimaginarycircuit) blows up combinatorially; use a local,
    # topology-constrained circuit instead to reach the multi-task threshold in a controlled way.
    nq = 8
    gates, taus = _localimaginarycircuit(nq, 2; seed=9)
    pstr = PauliString(nq, :I, 1)
    min_abs_coeff = 1e-6

    d_thread = Performance.propagate(gates, VectorPauliSum(pstr), taus; heisenberg=false, min_abs_coeff, fused=true, thread=true)
    d_nothread = Performance.propagate(gates, VectorPauliSum(pstr), taus; heisenberg=false, min_abs_coeff, fused=true, thread=false)

    @test length(d_thread) > 16384  # sanity check that this circuit actually exercises multiple tasks
    @test d_thread == d_nothread
end

@testset "fused Vector ImaginaryPauliRotation on wide Pauli strings matches stock exactly" begin
    nq = 100
    gates, taus = _localimaginarycircuit(nq, 1; seed=31)
    pstr = PauliString(nq, :I, 1)
    max_weight = 2

    stock = propagate(gates, VectorPauliSum(pstr), taus; heisenberg=false, min_abs_coeff=0.0, max_weight)
    fused = Performance.propagate(gates, VectorPauliSum(pstr), taus; heisenberg=false, min_abs_coeff=0.0, max_weight, fused=true)

    stock_dict = Dict(zip(paulis(stock), coefficients(stock)))
    fused_dict = Dict(zip(paulis(fused), coefficients(fused)))
    @test length(stock_dict) > 1024  # sanity check that this reaches the radix-sorted tail
    @test stock_dict == fused_dict
end

@testset "fused Dict PauliNoise matches stock exactly" begin
    # PauliNoise should not be affected by the fuse and Dict internals usage
    nq = 6

    Random.seed!(2)
    pstr = PauliString(nq, rand([:X, :Y, :Z], nq), 1:nq)
    noise_circuit = [DepolarizingNoise(qind, 0.03 + 0.01 * qind) for qind in 1:nq]
    min_abs_coeff = 1e-8

    stock = propagate(noise_circuit, pstr; min_abs_coeff)
    dict_fused = Performance.propagate(noise_circuit, pstr; min_abs_coeff, fused=true)

    @test dict_fused == stock
end

@testset "fused Vector PauliNoise matches stock exactly" begin
    # PauliNoise never branches or merges, so fused=true must reproduce stock coefficients exactly,
    # even under truncation.
    nq = 6

    Random.seed!(2)
    pstr = PauliString(nq, rand([:X, :Y, :Z], nq), 1:nq)
    noise_circuit = [DepolarizingNoise(qind, 0.03 + 0.01 * qind) for qind in 1:nq]
    min_abs_coeff = 1e-8

    stock = propagate(noise_circuit, pstr; min_abs_coeff)
    vec_fused_sum = Performance.propagate(noise_circuit, VectorPauliSum(pstr); min_abs_coeff, fused=true)
    vec_fused = PauliSum(nq, Dict(zip(paulis(vec_fused_sum), coefficients(vec_fused_sum))))

    @test vec_fused == stock
end

@testset "fused Vector: thread=false matches thread=true on a multi-task-sized propagation" begin
    # AK.TaskPartitioner only splits into multiple tasks once the active range is large enough, so
    # a small circuit would silently skip the multi-task code path. This one grows past 10^5 terms.
    nq = 14
    topo = bricklayertopology(nq; periodic=false)
    circuit = hardwareefficientcircuit(nq, 6; topology=topo)

    Random.seed!(9)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 3)
    min_abs_coeff = 1e-4

    d_thread = Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff, fused=true, thread=true)
    d_nothread = Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff, fused=true, thread=false)

    @test length(d_thread) > 1024  # sanity check that this circuit actually exercises multiple tasks
    @test d_thread == d_nothread
    @test overlapwithzero(d_thread) == overlapwithzero(d_nothread)
end

@testset "fused Vector: a cache with no room to spare matches one with room to spare" begin
    # A cache that starts exactly full makes nearly every gate split its walk and grow.
    # It has to land where a cache that never grows does.
    nq = 14
    topo = bricklayertopology(nq; periodic=false)
    circuit = hardwareefficientcircuit(nq, 6; topology=topo)

    Random.seed!(9)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 3)
    min_abs_coeff = 1e-4

    tight = PropagationCache(VectorPauliSum(pstr))
    roomy = PropagationCache(VectorPauliSum(pstr))
    resize!(roomy, 10^6)

    Performance.propagate!(circuit, tight, thetas; min_abs_coeff, fused=true, thread=false)
    Performance.propagate!(circuit, roomy, thetas; min_abs_coeff, fused=true, thread=false)

    @test length(tight) > 1024  # sanity check that the walks really do run out of room
    @test capacity(roomy) == 10^6  # sanity check that the other one never grew
    @test PauliSum(tight) == PauliSum(roomy)
end

@testset "fused Vector PauliNoise: a compacting walk matches an out-of-place one" begin
    # The single-task noise walk keeps the surviving terms where they lie and closes the gaps the
    # truncated ones leave. It has to agree with the multi-task walk, which writes them elsewhere,
    # and with stock, on a sum long enough to be split and with enough of it dropped to move terms.
    nq = 14
    topo = bricklayertopology(nq; periodic=false)
    circuit = hardwareefficientcircuit(nq, 6; topology=topo)

    Random.seed!(9)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 3)

    grown = Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff=1e-5, fused=true, thread=false)
    @test length(grown) > 2 * 16384  # sanity check that the noise walk below is split across tasks

    noise_circuit = [DepolarizingNoise(qind, 0.02 + 0.01 * qind) for qind in 1:nq]
    min_abs_coeff = 1e-3

    asdict(psum) = Dict(zip(paulis(psum), coefficients(psum)))
    stock = asdict(propagate(noise_circuit, deepcopy(grown); min_abs_coeff))
    nothread = Performance.propagate(noise_circuit, deepcopy(grown); min_abs_coeff, fused=true, thread=false)
    threaded = Performance.propagate(noise_circuit, deepcopy(grown); min_abs_coeff, fused=true, thread=true)

    @test length(nothread) < length(grown) ÷ 2  # sanity check that terms really are dropped
    @test asdict(nothread) == stock
    @test asdict(threaded) == stock
end