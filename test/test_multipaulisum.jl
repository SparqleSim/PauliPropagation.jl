# Test file for the MultiPauliSum type, which splits a term sum over work zones.
using Test
using Random

# largest deviation between two Pauli sums, over the terms of either
function maxdeviation(psum1, psum2)
    deviation = 0.0
    for pstr in union(keys(psum1.terms), keys(psum2.terms))
        deviation = max(deviation, abs(getcoeff(psum1, pstr) - getcoeff(psum2, pstr)))
    end
    return deviation
end

# every zone type must reproduce what the same circuit does to a plain PauliSum
function agreeswithpaulisum(circuit, pstr, thetas, n_zones; kwargs...)
    expected = propagate(circuit, PauliSum(pstr), thetas; kwargs...)

    for seed in (PauliSum(pstr), VectorPauliSum(pstr))
        msum = MultiPauliSum(seed, n_zones)
        propagate!(circuit, msum, thetas; kwargs...)
        got = PauliSum(msum)

        if length(got) != length(expected) || maxdeviation(expected, got) > 1e-12
            return false
        end
    end

    return true
end

nq = 6
topology = bricklayertopology(nq)

rotations = PauliRotation[]
rxlayer!(rotations, nq)
rzlayer!(rotations, nq)
rzzlayer!(rotations, topology)
append!(rotations, [PauliRotation([:X, :Y, :Z], [1, 3, 5]), PauliRotation([:Y, :Y], [2, 4])])

cliffords = Gate[CliffordGate(:CNOT, pair) for pair in topology]
append!(cliffords, [CliffordGate(:H, [qind]) for qind in 1:nq])
append!(cliffords, [TGate(qind) for qind in 1:nq])

noises = Gate[DepolarizingNoise(qind) for qind in 1:nq]
append!(noises, [DephasingNoise(qind) for qind in 1:nq])
append!(noises, [AmplitudeDampingNoise(qind) for qind in 1:nq])

mixed = vcat(Gate[], rotations, cliffords, noises, rotations)
frozen = Gate[freeze(gate, gate isa PauliRotation ? 0.3 : 0.05) for gate in vcat(Gate[], rotations, noises)]

pstr = PauliString(nq, [:Z, :Z], [2, 3])


@testset "MultiPauliSum propagation" begin
    Random.seed!(42)
    thetas = randn(countparameters(rotations))
    params = [gate isa ParametrizedNoiseChannel ? 0.05 : randn() for gate in mixed if gate isa ParametrizedGate]

    for n_zones in (1, 2, 3, 4, 7, 8)
        @test agreeswithpaulisum(rotations, pstr, thetas, n_zones; min_abs_coeff=0.0)
        @test agreeswithpaulisum(rotations, pstr, thetas, n_zones; min_abs_coeff=1e-3, max_weight=4)
        @test agreeswithpaulisum(rotations, pstr, thetas, n_zones; min_abs_coeff=1e-8, heisenberg=false)
        @test agreeswithpaulisum(cliffords, pstr, nothing, n_zones; min_abs_coeff=0.0)
        @test agreeswithpaulisum(frozen, pstr, nothing, n_zones; min_abs_coeff=1e-8)
        @test agreeswithpaulisum(mixed, pstr, params, n_zones; min_abs_coeff=1e-8)
        @test agreeswithpaulisum(mixed, pstr, params, n_zones; min_abs_coeff=1e-10, min_rel_coeff=1e-4)
        @test agreeswithpaulisum(mixed, pstr, params, n_zones; min_abs_coeff=1e-8, thread=false)
    end
end

@testset "MultiPauliSum through the Performance module" begin
    Random.seed!(42)
    thetas = randn(countparameters(rotations))

    expected = propagate(rotations, PauliSum(pstr), thetas; min_abs_coeff=1e-8)
    got = PauliPropagation.Performance.propagate(rotations, MultiPauliSum(VectorPauliSum(pstr), 4), thetas; min_abs_coeff=1e-8)

    @test length(PauliSum(got)) == length(expected)
    @test maxdeviation(expected, PauliSum(got)) < 1e-12
end

@testset "MultiPauliSum imaginary time evolution" begin
    circuit = [ImaginaryPauliRotation([:X], [qind]) for qind in 1:nq]
    append!(circuit, [ImaginaryPauliRotation([:Z, :Z], pair) for pair in topology])
    taus = fill(0.1, length(circuit))

    identity_pstr = PauliString(nq, :I, 1)

    expected = propagate(circuit, PauliSum(identity_pstr), taus; heisenberg=false, min_abs_coeff=1e-10)

    for n_zones in (1, 4), seed in (PauliSum(identity_pstr), VectorPauliSum(identity_pstr))
        msum = MultiPauliSum(seed, n_zones)
        propagate!(circuit, msum, taus; heisenberg=false, min_abs_coeff=1e-10)
        got = PauliSum(msum)

        @test length(got) == length(expected)
        @test maxdeviation(expected, got) < 1e-12
    end
end

@testset "MultiPauliSum truncation" begin
    Random.seed!(42)
    psum = propagate(rotations, PauliSum(pstr), randn(countparameters(rotations)); min_abs_coeff=1e-8)

    # min_rel_coeff is relative to the largest coefficient anywhere, not to each zone's own largest
    expected = truncate!(deepcopy(psum); min_rel_coeff=0.05, min_abs_coeff=0.0)
    for n_zones in (2, 4, 8), seed in (psum, VectorPauliSum(psum))
        msum = truncate!(MultiPauliSum(seed, n_zones); min_rel_coeff=0.05, min_abs_coeff=0.0)
        @test PauliSum(msum) == expected
    end

    @test PauliSum(truncate!(MultiPauliSum(psum, 4); max_weight=3)) == truncate!(deepcopy(psum); max_weight=3)
    @test PauliSum(filter((pauli, coeff) -> coeff > 0, MultiPauliSum(psum, 4))) == filter((pauli, coeff) -> coeff > 0, psum)
end

@testset "MultiPauliSum interface" begin
    vpsum = VectorPauliSum(nq)
    add!(vpsum, [:X, :Y], [1, 2], 0.5)
    add!(vpsum, [:Z], [3], 0.25)
    add!(vpsum, [:Y, :Y], [4, 5], -0.75)

    msum = MultiPauliSum(vpsum, 4)

    @test length(msum) == 3
    @test !isempty(msum)
    @test nqubits(msum) == nq
    @test nsites(msum) == nq
    @test termtype(msum) == paulitype(vpsum)
    @test coefftype(msum) == Float64
    @test sort(collect(terms(msum))) == sort(collect(paulis(vpsum)))
    @test sum(coeff for (_, coeff) in msum) == 0.0
    @test length(topaulistrings(msum)) == 3
    @test norm(msum) ≈ norm(vpsum)
    @test maxabscoeff(msum) == 0.75

    @test getcoeff(msum, [:X, :Y], [1, 2]) == 0.5
    @test getcoeff(msum, PauliString(nq, :Z, 3)) == 0.25
    @test getcoeff(msum, symboltoint(nq, [:Y, :Y], [4, 5])) == -0.75
    @test getmergedcoeff(msum, symboltoint(nq, [:Y, :Y], [4, 5])) == -0.75

    @test overlapwithzero(msum) == overlapwithzero(vpsum)
    @test overlapwithmaxmixed(msum) == 0.0
    @test scalarproduct(msum, vpsum) == scalarproduct(vpsum, vpsum)
    @test overlapwithpaulisum(msum, msum) == overlapwithpaulisum(vpsum, vpsum)

    @test PauliSum(msum) == PauliSum(vpsum)
    @test VectorPauliSum(msum) == vpsum
    @test msum == MultiPauliSum(vpsum, 4)
    @test msum == MultiPauliSum(PauliSum(vpsum), 8)

    @test coefftype(convertcoefftype(ComplexF64, msum)) == ComplexF64
    @test conj(mult!(convertcoefftype(ComplexF64, msum), im)) == mult!(convertcoefftype(ComplexF64, msum), -im)

    # a zone carries the very type it was seeded from, term type included
    @test paulitype(MultiPauliSum(VectorPauliSum(nq, UInt64[], Float64[]), 4)) == UInt64
    @test paulitype(MultiPauliSum(PauliSum(nq, Dict{UInt64,Float64}()), 4)) == UInt64

    add!(msum, [:X, :Y], [1, 2], 0.5)
    @test getcoeff(msum, [:X, :Y], [1, 2]) == 1.0

    set!(msum, symboltoint(nq, [:Z], [3]), 2.0)
    @test getcoeff(msum, [:Z], [3]) == 2.0

    mult!(msum, 2.0)
    @test getcoeff(msum, [:Z], [3]) == 4.0

    delete!(msum, symboltoint(nq, [:Z], [3]))
    @test getcoeff(msum, [:Z], [3]) == 0.0
    @test length(msum) == 2

    @test isempty(similar(msum))
    @test length(truncate!(msum; min_abs_coeff=1.6)) == 1
    @test isempty(empty!(msum))
end

@testset "MultiPauliSum constructors and conversions" begin
    psum = PauliSum(nq)
    add!(psum, [:X, :Y], [1, 2], 0.5)
    add!(psum, [:Z], [3], 0.25)
    vpsum = VectorPauliSum(psum)

    @test nzones(MultiPauliSum(psum)) == defaultnzones()
    @test_throws ArgumentError MultiPauliSum(psum, 0)

    # empty sums, on a number of qubits and optionally a coefficient type
    @test isempty(MultiPauliSum(nq)) && nqubits(MultiPauliSum(nq)) == nq
    @test nzones(MultiPauliSum(nq, 8)) == 8
    @test nzones(MultiPauliSum(nq, 6)) == 6
    @test coefftype(MultiPauliSum(ComplexF64, nq, 8)) == ComplexF64
    @test PauliSum(MultiPauliSum(PauliString(nq, :X, 1), 4)) == PauliSum(PauliString(nq, :X, 1))

    # the zone type follows what the sum was split from, and survives re-zoning
    for (seed, ZT) in ((psum, typeof(psum)), (vpsum, typeof(vpsum)))
        msum = MultiPauliSum(seed, 4)
        @test eltype(zones(msum)) == ZT
        @test eltype(zones(MultiPauliSum(msum, 8))) == ZT
        @test nzones(MultiPauliSum(msum, 8)) == 8
        @test PauliSum(MultiPauliSum(msum, 8)) == psum

        # gathering the zones back, directly and out of a propagation cache
        @test PauliSum(msum) == psum
        @test VectorPauliSum(msum) == vpsum
        @test convert(PauliSum, msum) == psum
        @test PauliSum(PropagationCache(deepcopy(msum))) == psum
    end
end

@testset "MultiPauliSum propagation cache" begin
    Random.seed!(42)
    vpsum = VectorPauliSum(PauliString(nq, [:Z, :Z], [2, 3]))
    msum = MultiPauliSum(vpsum, 4)
    prop_cache = PropagationCache(msum)

    @test length(prop_cache) == 1
    @test mainsum(prop_cache) === msum
    @test length(activesum(prop_cache)) == 1
    @test maxabscoeff(prop_cache) == 1.0
    @test numcoefftype(prop_cache) == Float64
    @test overlapwithzero(prop_cache) == overlapwithzero(vpsum)

    resize!(prop_cache, 400)
    @test capacity(prop_cache) >= 400

    propagate!(rotations, prop_cache, randn(countparameters(rotations)); min_abs_coeff=1e-8)

    # the cache hands back the very sum it was built from, sized to what it holds
    @test extractsum!(prop_cache) === msum
    @test length(msum) == length(PauliSum(msum))

    # out-of-place propagation leaves the sum it was given alone
    untouched = MultiPauliSum(vpsum, 4)
    propagated = propagate(rotations, untouched, randn(countparameters(rotations)); min_abs_coeff=1e-8)
    @test length(untouched) == 1
    @test length(propagated) > 1
end

@testset "MultiPauliSum zone assignment" begin
    Random.seed!(42)
    psum = propagate(rotations, PauliSum(pstr), randn(countparameters(rotations)); min_abs_coeff=1e-10)

    for n_zones in (2, 4, 6, 8), seed in (psum, VectorPauliSum(psum))
        msum = propagate(mixed, MultiPauliSum(seed, n_zones), [gate isa ParametrizedNoiseChannel ? 0.05 : 0.3 for gate in mixed if gate isa ParametrizedGate]; min_abs_coeff=1e-8)

        # every term sits in the zone that owns it, and no zone holds a term twice
        @test all(all(zoneof(msum, term) == zone_id for term in terms(zone)) for (zone_id, zone) in enumerate(msum.zones))
        @test all(length(zone) == length(unique(terms(zone))) for zone in msum.zones)

        # parities of fixed masks spread the terms evenly, whatever the sum looks like
        @test maximum(zonesizes(msum)) < 1.5 * length(msum) / nzones(msum)
    end

    # over a power-of-two number of zones the assignment is linear over GF(2), which is what lets a
    # gate that branches by a fixed mask send a whole zone to a single other zone
    terms_and_masks = collect(zip(rand(paulitype(psum), 100), rand(paulitype(psum), 100)))
    islinear(msum) = all(zoneof(msum, term ⊻ mask) - 1 == (zoneof(msum, term) - 1) ⊻ (zoneof(msum, mask) - 1)
                         for (term, mask) in terms_and_masks)

    @test isxorlinear(MultiPauliSum(psum, 8)) && islinear(MultiPauliSum(psum, 8))
    @test !isxorlinear(MultiPauliSum(psum, 6)) && !islinear(MultiPauliSum(psum, 6))
    @test zonemap(MultiPauliSum(psum, 8)) isa XorZoneMap
    @test zonemap(MultiPauliSum(psum, 6)) isa FoldedZoneMap
end
