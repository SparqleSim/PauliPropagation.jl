@testset "Term sum primitives" begin
    dict_sum = PauliSum(2, Dict{UInt8,Float64}(0x01 => 1.0, 0x02 => -2.0, 0x03 => 3.0))

    mapped_sum = map((term, coefficient) -> (zero(term), coefficient), dict_sum)
    @test length(mapped_sum) == 1
    @test getcoeff(mapped_sum, 0x00) == 2.0
    @test dict_sum == PauliSum(2, Dict{UInt8,Float64}(0x01 => 1.0, 0x02 => -2.0, 0x03 => 3.0))

    filter!((term, coefficient) -> coefficient > 0, dict_sum)
    @test length(dict_sum) == 2
    @test sort(collect(coefficients(dict_sum))) == [1.0, 3.0]
    @test mapreducecoeffs(abs, +, dict_sum) == 4.0
    @test mapreduce((term, coefficient) -> term * coefficient, +, dict_sum) == 10.0

    filtered_terms = filterterms(term -> term != 0x01, dict_sum)
    @test length(filtered_terms) == 1
    @test getcoeff(filtered_terms, 0x03) == 3.0

    vector_sum = VectorPauliSum(2, UInt8[0x03, 0x01, 0x02], [3.0, 1.0, 2.0])
    sortterms!(vector_sum)
    @test collect(PauliPropagation.PropagationBase.terms(vector_sum)) == UInt8[0x01, 0x02, 0x03]
    @test PauliPropagation.PropagationBase.sortedprefix(vector_sum) == 3
    @test getcoeff(vector_sum, 0x02) == 2.0

    duplicated_sum = VectorPauliSum(2, UInt8[0x03, 0x01, 0x02, 0x01], [3.0, 1.0, 2.0, 1.0])
    sortterms!(duplicated_sum)
    @test PauliPropagation.PropagationBase.sortedprefix(duplicated_sum) == 1
    sortterms!(duplicated_sum; rev=true)
    @test PauliPropagation.PropagationBase.sortedprefix(duplicated_sum) == 0

    mapterms!(term -> term ⊻ 0x03, vector_sum)
    mapcoeffs!(coefficient -> -coefficient, vector_sum)
    sortcoeffs!(vector_sum)
    @test collect(coefficients(vector_sum)) == [-3.0, -2.0, -1.0]

    filter!((term, coefficient) -> coefficient < -1, vector_sum)
    @test collect(coefficients(vector_sum)) == [-3.0, -2.0]
    @test mapreduce((term, coefficient) -> (term + 1) * coefficient, +, vector_sum) == -7.0

    filtercoeffs!(coefficient -> coefficient < -2, vector_sum)
    @test collect(coefficients(vector_sum)) == [-3.0]
end

@testset "Propagation cache primitives" begin
    cache = PropagationCache(VectorPauliSum(2, UInt8[0x01, 0x02, 0x03], [1.0, -2.0, 3.0]))
    resize!(cache, 8)

    cached_terms = PauliPropagation.PropagationBase.terms(mainsum(cache))
    cached_coefficients = coefficients(mainsum(cache))
    cached_terms[4:end] .= 0x00
    cached_coefficients[4:end] .= -99.0

    map!((term, coefficient) -> (term ⊻ 0x03, abs(coefficient)), cache)
    @test cached_terms[4:end] == fill(0x00, 5)
    @test cached_coefficients[4:end] == fill(-99.0, 5)

    filter!((term, coefficient) -> coefficient >= 2, cache)
    sortcoeffs!(cache; rev=true)

    @test length(cache) == 2
    @test collect(coefficients(cache)) == [3.0, 2.0]
    @test mapreducecoeffs(abs, +, cache) == 5.0
    @test mapreduce((term, coefficient) -> (term + 1) * coefficient, +, cache) == 7.0

    multi_cache = PropagationCache(MultiPauliSum(
        VectorPauliSum(2, UInt8[0x01, 0x02, 0x03], [1.0, -2.0, 3.0]), 2))

    mapterms!(term -> term ⊻ 0x03, multi_cache)
    mapcoeffs!(abs, multi_cache)
    filter!((term, coefficient) -> coefficient >= 2, multi_cache)

    @test length(multi_cache) == 2
    @test mapreduce((term, coefficient) -> (term + 1) * coefficient, +, multi_cache) == 7.0
    @test all(
        all(
            term -> PauliPropagation.PropagationBase.zoneof(multi_cache, term) == zone_id,
            PauliPropagation.PropagationBase.terms(zonecache),
        ) for (zone_id, zonecache) in enumerate(PauliPropagation.PropagationBase.zonecaches(multi_cache))
    )
end

@testset "Branching and filtering primitives" begin
    nq = 6
    rng = MersenneTwister(7)
    dict_sum = PauliSum(nq)
    for _ in 1:200
        add!(dict_sum, symboltoint(rand(rng, (:I, :X, :Y, :Z), nq)), randn(rng))
    end

    gate = PauliRotation([:X, :Y], [2, 5])
    theta = 0.7
    reference = propagate(gate, dict_sum, theta; min_abs_coeff=0.0)

    # a rule for `xorbranch!` that is the rotation: untouched, or kept and a new term
    gate_mask = symboltoint(paulitype(dict_sum), gate.symbols, gate.qinds)
    function rotate(pstr, coeff)
        commutes(gate_mask, pstr) && return nothing
        _, sign = PauliPropagation.paulirotationproduct(gate_mask, pstr)
        return (coeff * cos(theta), coeff * sin(theta) * sign)
    end

    # a rule that only rescales: every Pauli string with a Z on the first qubit
    rescale(pstr, coeff) = getpauli(pstr, 1) == 3 ? 0.5 * coeff : nothing
    rescaled_reference = mapcoeffs(identity, dict_sum)
    for (pstr, coeff) in dict_sum
        getpauli(pstr, 1) == 3 && set!(rescaled_reference, pstr, 0.5 * coeff)
    end

    # a coefficient map based on the pair and drops on `nothing`
    halve_or_drop(pstr, coeff) = coeff > 0 ? 0.5 * coeff : nothing

    for makesum in (identity, VectorPauliSum, psum -> MultiPauliSum(VectorPauliSum(psum), 4), psum -> MultiPauliSum(psum, 4))
        branched = PauliPropagation.PropagationBase.xorbranch(rotate, makesum(dict_sum), gate_mask)
        @test PauliSum(branched) ≈ reference

        # truncating in the merge agrees with truncating afterwards
        truncfunc(pstr, coeff) = abs(coeff) < 0.05
        truncated = PauliPropagation.PropagationBase.xorbranch(rotate, makesum(dict_sum), gate_mask; truncfunc)
        @test PauliSum(truncated) ≈ truncate(reference; min_abs_coeff=0.05)

        rescaled = PauliPropagation.PropagationBase.xorbranch(rescale, makesum(dict_sum), gate_mask)
        @test PauliSum(rescaled) ≈ rescaled_reference

        filtered = mapcoeffsbypair!(halve_or_drop, deepcopy(makesum(dict_sum)))
        @test length(filtered) == count(coeff > 0 for (_, coeff) in dict_sum)
        @test all(getcoeff(filtered, pstr) ≈ 0.5 * coeff for (pstr, coeff) in dict_sum if coeff > 0)
    end

    # on a sorted array sum, neither primitive disturbs the sorted prefix
    vector_sum = VectorPauliSum(dict_sum)
    sortterms!(vector_sum)
    prop_cache = PropagationCache(vector_sum)
    PauliPropagation.PropagationBase.setsortedprefix!(mainsum(prop_cache), length(prop_cache))

    PauliPropagation.PropagationBase.xorbranch!(rotate, prop_cache, gate_mask; thread=false)
    @test PauliPropagation.PropagationBase.sortedprefix(mainsum(prop_cache)) == length(prop_cache)
    @test issorted(PauliPropagation.PropagationBase.terms(prop_cache))

    mapcoeffsbypair!(halve_or_drop, prop_cache; thread=false)
    @test PauliPropagation.PropagationBase.sortedprefix(mainsum(prop_cache)) == length(prop_cache)
    @test issorted(PauliPropagation.PropagationBase.terms(prop_cache))

    # the kernels for arrays that are not on the CPU agree with the CPU kernels
    cpu_cache = PropagationCache(VectorPauliSum(dict_sum))
    portable_cache = PropagationCache(VectorPauliSum(dict_sum))
    PauliPropagation.PropagationBase._branchcpu!(rotate, cpu_cache, gate_mask; thread=false)
    PauliPropagation.PropagationBase._branchflagged!(rotate, portable_cache, gate_mask; thread=false)
    @test PauliPropagation.PropagationBase.terms(cpu_cache) == PauliPropagation.PropagationBase.terms(portable_cache)
    @test coefficients(cpu_cache) == coefficients(portable_cache)

    PauliPropagation.PropagationBase._mapcoeffsbypaircpu!(halve_or_drop, cpu_cache; thread=false)
    PauliPropagation.PropagationBase._mapcoeffsbypairflagged!(halve_or_drop, portable_cache; thread=false)
    @test PauliPropagation.PropagationBase.terms(cpu_cache) == PauliPropagation.PropagationBase.terms(portable_cache)
    @test coefficients(cpu_cache) == coefficients(portable_cache)
end

@testset "Gates written for every cache dispatch without ties" begin
    # a custom gate defined on the abstract Pauli cache runs on every storage, since the generic
    # `applytoall!` dispatches on the storage of the cache
    struct HalvingGate <: StaticGate end
    PauliPropagation.PropagationBase.applytoall!(::HalvingGate, prop_cache::PauliPropagation.AbstractPauliPropagationCache; kwargs...) =
        mapcoeffs!(coeff -> 0.5 * coeff, prop_cache)

    psum = PauliSum(PauliString(3, [:X, :Z], [1, 3], 0.8))
    for makesum in (identity, VectorPauliSum, psum -> MultiPauliSum(VectorPauliSum(psum), 2))
        out = propagate([HalvingGate()], makesum(psum))
        @test getcoeff(out, [:X, :Z], [1, 3]) ≈ 0.4
    end

    # a custom gate that only defines `apply` takes the generic path, which a multi sum runs zone by zone
    struct SwappingGate <: StaticGate end
    PauliPropagation.PropagationBase.apply(::SwappingGate, pstr, coeff; kwargs...) =
        ((setpauli(setpauli(pstr, getpauli(pstr, 3), 1), getpauli(pstr, 1), 3), coeff),)

    for makesum in (identity, psum -> MultiPauliSum(VectorPauliSum(psum), 2), psum -> MultiPauliSum(psum, 2))
        out = propagate([SwappingGate()], makesum(psum))
        @test getcoeff(out, [:Z, :X], [1, 3]) ≈ 0.8
    end

    @test isempty(Test.detect_ambiguities(PauliPropagation, PauliPropagation.PropagationBase, PauliPropagation.Performance))
end
