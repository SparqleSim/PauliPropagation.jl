using Random
using Test

# A hashed term sum is a storage of its own: it holds terms as a PauliSum does, and the gates of the library propagate a
# sum held in it through the default cache, with the partner of every term found in place.

@testset "HashedTermSum holds terms like a PauliSum" begin
    nq = 6
    rng = MersenneTwister(2)
    TT = UInt16
    hsum = PB.HashedTermSum{TT,Float64}(nq)
    @test length(hsum) == 0 && isempty(collect(hsum)) && nqubits(hsum) == nq
    @test PB.StorageType(hsum) == PB.HashedStorage()

    # more terms than a fresh sum has slots for, some of them twice, so that the table grows with its entries in it
    reference = PauliSum(nq)
    added = TT[]
    for _ in 1:200
        term = rand(rng, TT(0):TT(4^nq - 1))
        coeff = randn(rng)
        add!(hsum, term, coeff)
        add!(reference, term, coeff)
        push!(added, term)
    end
    @test length(hsum) == length(reference)
    @test all(getcoeff(hsum, term) == getcoeff(reference, term) for term in added)
    @test getcoeff(hsum, TT(4^nq - 1) ⊻ added[1]) == getcoeff(reference, TT(4^nq - 1) ⊻ added[1])
    @test Set(PB.terms(hsum)) == Set(PB.terms(reference))
    @test PauliSum(nq, Dict(collect(hsum))) == reference
    @test collect(coefficients(hsum)) == [coeff for (_, coeff) in hsum]

    # a copy holds the same terms
    copied = copy!(similar(hsum), hsum)
    @test length(copied) == length(hsum) && all(getcoeff(copied, term) == coeff for (term, coeff) in hsum)

    # a term set, deleted, and added again after its deletion takes its entry back
    term = added[1]
    set!(hsum, term, 2.5)
    @test getcoeff(hsum, term) == 2.5
    delete!(hsum, term)
    @test getcoeff(hsum, term) == 0.0 && length(hsum) == length(reference) - 1
    add!(hsum, term, 1.5)
    @test getcoeff(hsum, term) == 1.5 && length(hsum) == length(reference)

    # coefficients mapped and truncated in place
    mult!(hsum, 2.0)
    @test getcoeff(hsum, term) == 3.0
    truncate!(hsum; min_abs_coeff=1.0)
    @test all(abs(coeff) >= 1.0 for (_, coeff) in hsum)
    @test length(hsum) == count(coeff -> abs(coeff) >= 1.0, collect(coefficients(copied)) .* 2) + (abs(getcoeff(copied, term)) * 2 < 1.0)
    @test maxabscoeff(hsum) == maximum(abs, collect(coefficients(hsum)))

    # emptied, with slots for many terms, it holds nothing and takes terms again
    empty!(hsum; sizehint=1000)
    @test length(hsum) == 0
    add!(hsum, term, 1.0)
    @test getcoeff(hsum, term) == 1.0 && length(hsum) == 1
end

@testset "propagate! on a HashedTermSum agrees with a PauliSum" begin
    nq = 6
    rng = MersenneTwister(3)
    psum = PauliSum(nq)
    for _ in 1:10
        weight = rand(rng, 1:3)
        add!(psum, rand(rng, [:X, :Y, :Z], weight), randperm(rng, nq)[1:weight], randn(rng))
    end
    # rotations that branch, Clifford gates that map, and noise that rescales or branches into kept terms
    circuit = Gate[PauliRotation(:X, 1), PauliRotation([:Z, :Z], [1, 2]), CliffordGate(:CNOT, [2, 3]), PauliRotation(:Y, 3),
        CliffordGate(:H, [4]), PauliRotation([:X, :Y], [3, 4]), DepolarizingNoise(5), AmplitudeDampingNoise(2), PauliRotation(:Z, 5),
        PauliRotation([:Z, :Z], [5, 6]), PauliRotation(:X, 6), PauliRotation([:Y, :Y], [1, 6])]
    function parameterof(gate)
        if gate isa ParametrizedNoiseChannel
            return 0.1
        end
        return randn(rng)
    end
    params = [parameterof(gate) for gate in circuit if PB.isparametrized(gate)]

    for truncation in ((; min_abs_coeff=0.0), (; min_abs_coeff=1e-3), (; max_weight=3.0, min_abs_coeff=1e-4))
        reference = propagate(circuit, psum, params; truncation...)
        hsum = PB.HashedTermSum{paulitype(psum),Float64}(nq)
        for (term, coeff) in psum
            add!(hsum, term, coeff)
        end
        # the Pauli gates propagate a hashed sum through the default cache of Pauli strings
        cache = PP.PauliPropagationCache(hsum)
        propagate!(circuit, cache, params; truncation...)
        propagated = mainsum(cache)
        @test propagated isa PB.HashedTermSum
        @test length(propagated) == length(reference) > 1
        @test all(getcoeff(reference, term) == coeff for (term, coeff) in propagated)
    end
end

@testset "xorbranchmergeandtruncate! on a HashedTermSum handles every outcome of a rule" begin
    nq = 5
    rng = MersenneTwister(4)
    TT = UInt16
    mask = TT(0b0110)
    # a rule that leaves some terms alone, rescales some and branches the rest, so that a branch meets partners that are
    # present, absent after a truncation, and not there at all
    function rule(term, coeff)
        if iszero(term & TT(0b11000000))
            return PB.Unchanged()
        elseif isodd(term >> 6)
            return PB.Kept(0.5 * coeff)
        else
            return PB.Branch(0.6 * coeff, 0.4 * coeff)
        end
    end
    truncfunc(term, coeff) = abs(coeff) < 0.3

    reference = PropagationCache(PauliSum(nq))
    hashed = PropagationCache(PB.HashedTermSum{TT,Float64}(nq))
    for _ in 1:60
        term = rand(rng, TT(0):TT(4^nq - 1))
        coeff = randn(rng)
        add!(mainsum(reference), term, coeff)
        add!(mainsum(hashed), term, coeff)
    end
    for _ in 1:3
        PB.xorbranchmergeandtruncate!(rule, truncfunc, reference, mask)
        PB.xorbranchmergeandtruncate!(rule, truncfunc, hashed, mask)
        @test length(hashed) == length(reference) > 0
        @test all(getcoeff(mainsum(reference), term) == coeff for (term, coeff) in hashed)
    end
    # without a truncation, and a rule that returns something else
    PB.xorbranchmergeandtruncate!(rule, nothing, reference, mask)
    PB.xorbranchmergeandtruncate!(rule, nothing, hashed, mask)
    @test length(hashed) == length(reference)
    @test all(getcoeff(mainsum(reference), term) == coeff for (term, coeff) in hashed)
    @test_throws ArgumentError PB.xorbranchmergeandtruncate!((term, coeff) -> coeff, nothing, hashed, mask)
end

@testset "a HashedTermSum keyed by distinguishing bits propagates as a PauliSum does" begin
    nq = 36
    TT = UInt128
    rng = MersenneTwister(5)
    # the terms differ on the first `n_free` qubits only, so that their bits there tell the terms apart: gathered into
    # a key where they fit in 64 bits, hashed where they do not
    for n_free in (20, 35)
        free_bits = (one(TT) << (2 * n_free)) - one(TT)
        fixed = PauliString(nq, rand(rng, [:X, :Y, :Z], nq - n_free), n_free+1:nq).term
        circuit = [PauliRotation(rand(rng, [:X, :Y, :Z]), rand(rng, 1:n_free)) for _ in 1:8]
        append!(circuit, [PauliRotation([:Z, :Z], [q, q + 1]) for q in rand(rng, 1:n_free-1, 6)])
        angles = 0.3 * randn(rng, length(circuit))
        psum = PauliSum(nq)
        for _ in 1:100
            add!(psum, fixed ⊻ (rand(rng, TT) & free_bits), randn(rng))
        end
        # at quarter turns every term moves to its partner
        for (min_abs_coeff, circuit_angles) in ((3e-2, angles), (1e-2, angles), (1e-2, fill(pi / 2, length(angles))))
            hsum = PB.HashedTermSum{TT,Float64}(nq)
            PB._distinguishby!(hsum, free_bits)
            for (term, coeff) in psum
                add!(hsum, term, coeff)
            end
            propagated = mainsum(propagate!(circuit, PP.PauliPropagationCache(hsum), circuit_angles; min_abs_coeff))
            reference = propagate(circuit, psum, circuit_angles; min_abs_coeff)
            @test length(propagated) == length(reference) > 32
            @test all(getcoeff(reference, term) == coeff for (term, coeff) in propagated)
        end
    end
    # a broken promise is caught when the second term is added
    hsum = PB.HashedTermSum{TT,Float64}(nq)
    PB._distinguishby!(hsum, TT(0b11))
    add!(hsum, TT(0b01), 1.0)
    @test_throws ArgumentError add!(hsum, TT(0b101), 1.0)
end

@testset "the creation floor comes from the rule and the cutoff of the truncation" begin
    cache = PropagationCache(PauliSum(PauliString(4, :Z, 1, 1.0)))
    rule = PB.onlimbs(PP.ActingRotationRule, PP._branchmask(PauliRotation(:Y, 2), cache), cos(0.3), sin(0.3))
    @test PB._creationfloor(rule, 1e-3) ≈ 1e-3 / sin(0.3) && PB._creationfloor(rule, 1e-3) < 1e-3 / sin(0.3)
    @test PB._creationfloor(rule, 0.0) == 0.0
    @test PB._creationfloor(PP._branchrule(PauliRotation(:Y, 2), cache, 0.3), 1e-3) == 0.0
    @test PB._coefficientcutoff(buildtruncfunc(cache; min_abs_coeff=1e-5)) == 1e-5
    @test PB._coefficientcutoff(buildtruncfunc(cache; min_abs_coeff=1e-5, max_weight=3.0)) == 1e-5
    @test PB._coefficientcutoff(buildtruncfunc(cache; min_abs_coeff=1e-5, min_rel_coeff=0.5)) == 0.5
    @test PB._coefficientcutoff(nothing) == 0.0
    @test PB._coefficientcutoff((term, coeff) -> abs(coeff) < 0.1) == 0.0
end
