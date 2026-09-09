using Random


@testset "Test @countpaulis" begin

    nq = 6
    nl = 3
    min_abs_coeff = 1e-8

    pstr = PauliString(nq, :Z, 3)
    circ = hardwareefficientcircuit(nq, nl; topology=bricklayertopology(nq; periodic=false))

    Random.seed!(42)
    thetas = randn(countparameters(circ))

    counts = @countpaulis propagate(circ, pstr, thetas; min_abs_coeff)

    # one count per gate, and the last one is the size of the propagated sum
    @test length(counts) == length(circ)
    @test counts[end] == length(propagate(circ, pstr, thetas; min_abs_coeff))

    # the counts are exactly what applying the circuit gate by gate produces
    psum = PauliSum(pstr)
    manual_counts = Int[]
    theta_iterator = Iterators.Stateful(reverse(thetas))
    for gate in reverse(circ)
        if isa(gate, ParametrizedGate)
            propagate!(gate, psum, popfirst!(theta_iterator); min_abs_coeff)
        else
            propagate!(gate, psum; min_abs_coeff)
        end
        push!(manual_counts, length(psum))
    end
    @test manual_counts == counts

    # the vector backend takes the same path through the circuit
    @test (@countpaulis propagate(circ, VectorPauliSum(pstr), thetas; min_abs_coeff)) == counts

    # in-place propagation is counted too, and keeps the propagated sum
    psum = PauliSum(pstr)
    @test (@countpaulis propagate!(circ, psum, thetas; min_abs_coeff)) == counts
    @test length(psum) == counts[end]

    # an instrumented assignment still assigns, in the scope the macro was written in
    assigned_counts = @countpaulis propagated = propagate(circ, pstr, thetas; min_abs_coeff)
    @test assigned_counts == counts
    @test length(propagated) == counts[end]

    # the Schrödinger picture is counted the same way
    @test length(@countpaulis propagate(circ, pstr, thetas; min_abs_coeff, heisenberg=false)) == length(circ)

    # Monte Carlo propagation and sampling go through the same gate loop
    @test length(@countpaulis mcpropagate(circ, VectorPauliSum(pstr), thetas; max_size=100)) == length(circ)
    @test length(@countpaulis mcsample(circ, pstr, thetas)) == length(circ)

    # a truncation that keeps nothing leaves empty sums behind, not missing counts
    empty_counts = @countpaulis propagate(circ, pstr, thetas; min_abs_coeff=1e10)
    @test length(empty_counts) == length(circ)
    @test all(iszero, empty_counts)

    # nothing to count
    @test (@countpaulis 1 + 1) == Int[]

end

@testset "Test @peakpaulis" begin

    nq = 6
    nl = 3
    min_abs_coeff = 1e-8

    pstr = PauliString(nq, :Z, 3)
    circ = hardwareefficientcircuit(nq, nl; topology=bricklayertopology(nq; periodic=false))

    Random.seed!(42)
    thetas = randn(countparameters(circ))

    counts = @countpaulis propagate(circ, pstr, thetas; min_abs_coeff)

    @test (@peakpaulis propagate(circ, pstr, thetas; min_abs_coeff)) == maximum(counts)

    # the peak runs over every propagation inside the expression
    function twopropagations(circ, thetas)
        short_circ = circ[1:2]
        propagate(short_circ, pstr, thetas[1:countparameters(short_circ)]; min_abs_coeff)
        return propagate(circ, pstr, thetas; min_abs_coeff)
    end
    @test (@peakpaulis twopropagations(circ, thetas)) == maximum(counts)

    # gradients propagate forwards and then backwards
    @test (@peakpaulis rewindgradient(circ, VectorPauliSum(pstr), thetas, overlapwithzero; min_abs_coeff)) >= maximum(counts)

    # nothing propagated
    @test (@peakpaulis 1 + 1) == 0

    # counters stack, so the outer one sees what the inner one counts
    @test (@peakpaulis (@countpaulis propagate(circ, pstr, thetas; min_abs_coeff))) == maximum(counts)

    # a destructuring assignment is instrumented without swallowing its left-hand side
    function tworeturns(circ, thetas)
        return propagate(circ, pstr, thetas; min_abs_coeff), :marker
    end
    peak = @peakpaulis propagated, marker = tworeturns(circ, thetas)
    @test peak == maximum(counts)
    @test length(propagated) == counts[end]
    @test marker == :marker

    # the same inside a function, where the assignment is to a fresh local
    function peakandsize(circ, thetas)
        peak = @peakpaulis psum = propagate(circ, pstr, thetas; min_abs_coeff)
        return peak, length(psum)
    end
    @test peakandsize(circ, thetas) == (maximum(counts), counts[end])

    # a counter is uninstalled again even when the expression throws
    @test_throws ErrorException @peakpaulis error("propagation failed")
    @test_throws ErrorException @peakpaulis failed = error("propagation failed")
    @test (@peakpaulis 1 + 1) == 0

end
