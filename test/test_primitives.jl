function test_reducers(thing)
    @test mapreducecoeffs(identity, *, thing; init=1.0) == -24.0
    @test mapreducecoeffs(identity, min, thing; init=Inf) == -2.0
    @test mapreduce((term, coefficient) -> term + coefficient, max, thing; init=-Inf) == 7.0

    add_reducer = (left, right) -> left + right
    @test_throws ErrorException mapreducecoeffs(abs, add_reducer, thing; init=0.0)
    @test mapreducecoeffs(abs, add_reducer, thing; init=0.0, neutral=0.0) == 9.0
end

const REDUCTION_SUM = PauliSum(2, Dict{UInt8,Float64}(0x01 => -2.0, 0x02 => 3.0, 0x03 => 4.0))

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

    test_reducers(REDUCTION_SUM)

    filtered_terms = filterterms(term -> term != 0x01, dict_sum)
    @test length(filtered_terms) == 1
    @test getcoeff(filtered_terms, 0x03) == 3.0

    vector_sum = VectorPauliSum(2, UInt8[0x03, 0x01, 0x02], [3.0, 1.0, 2.0])
    sortterms!(vector_sum)
    @test collect(PauliPropagation.PropagationBase.terms(vector_sum)) == UInt8[0x01, 0x02, 0x03]
    @test PauliPropagation.PropagationBase.sortedprefix(vector_sum) == 3
    @test getcoeff(vector_sum, 0x02) == 2.0
    test_reducers(VectorPauliSum(REDUCTION_SUM))
    test_reducers(MultiPauliSum(VectorPauliSum(REDUCTION_SUM), 2))

    # More than two chunks with eight threads: every chunk must start from `one`, not `zero(init)`.
    parallel_product = VectorPauliSum(2, fill(UInt8(0x01), 32_769), ones(32_769))
    @test mapreducecoeffs(identity, *, parallel_product; init=2.0, thread=true) == 2.0

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
    test_reducers(PropagationCache(VectorPauliSum(REDUCTION_SUM)))

    multi_cache = PropagationCache(MultiPauliSum(
        VectorPauliSum(2, UInt8[0x01, 0x02, 0x03], [1.0, -2.0, 3.0]), 2))

    mapterms!(term -> term ⊻ 0x03, multi_cache)
    mapcoeffs!(abs, multi_cache)
    filter!((term, coefficient) -> coefficient >= 2, multi_cache)

    @test length(multi_cache) == 2
    @test mapreduce((term, coefficient) -> (term + 1) * coefficient, +, multi_cache) == 7.0
    test_reducers(PropagationCache(MultiPauliSum(VectorPauliSum(REDUCTION_SUM), 2)))
    @test mapreducecoeffs(identity, *, VectorPauliSum(2); init=3.0) == 3.0
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
        if commutes(gate_mask, pstr)
            return PB.Unchanged()
        else
            _, sign = PauliPropagation.paulirotationproduct(gate_mask, pstr)
            return PB.Branch(coeff * cos(theta), coeff * sin(theta) * sign)
        end
    end

    # a rule that only rescales: every Pauli string with a Z on the first qubit
    rescale(pstr, coeff) = getpauli(pstr, 1) == 3 ? PB.Kept(0.5 * coeff) : PB.Unchanged()
    rescaled_reference = mapcoeffs(identity, dict_sum)
    for (pstr, coeff) in dict_sum
        if getpauli(pstr, 1) == 3
            set!(rescaled_reference, pstr, 0.5 * coeff)
        end
    end

    # a coefficient map that reads the pair, and the filter that keeps what it halves
    halve_positive(pstr, coeff) = coeff > 0 ? 0.5 * coeff : coeff
    keep_positive(pstr, coeff) = coeff > 0

    for makesum in (identity, VectorPauliSum, psum -> MultiPauliSum(VectorPauliSum(psum), 4), psum -> MultiPauliSum(psum, 4))
        branched = PauliPropagation.PropagationBase.xorbranch(rotate, makesum(dict_sum), gate_mask)
        @test PauliSum(branched) ≈ reference

        # a rule may only return the three outcomes
        bare_coefficient(pstr, coeff) = 0.5 * coeff
        @test_throws ArgumentError PB.xorbranch(bare_coefficient, makesum(dict_sum), gate_mask; thread=false)

        # truncating in the merge agrees with truncating afterwards
        truncfunc(pstr, coeff) = abs(coeff) < 0.05
        truncated = PauliPropagation.PropagationBase.xorbranch(rotate, makesum(dict_sum), gate_mask; truncfunc)
        @test PauliSum(truncated) ≈ truncate(reference; min_abs_coeff=0.05)

        rescaled = PauliPropagation.PropagationBase.xorbranch(rescale, makesum(dict_sum), gate_mask)
        @test PauliSum(rescaled) ≈ rescaled_reference

        halved = mapcoeffsbypair!(halve_positive, deepcopy(makesum(dict_sum)))
        @test length(halved) == length(dict_sum)
        @test all(getcoeff(halved, pstr) ≈ halve_positive(pstr, coeff) for (pstr, coeff) in dict_sum)

        filtered = filter(keep_positive, makesum(dict_sum))
        @test length(filtered) == count(coeff > 0 for (_, coeff) in dict_sum)
        @test all(getcoeff(filtered, pstr) ≈ coeff for (pstr, coeff) in dict_sum if coeff > 0)

        # This is intentionally cache-only: it fuses a coefficient map with removal, whereas the
        # term-sum `mapcoeffsbypair!` above always preserves its number of terms.
        trunc_negative(pstr, coeff) = coeff <= 0
        map_and_truncated_cache = PropagationCache(deepcopy(makesum(dict_sum)))
        PB.mapandtruncate!(halve_positive, trunc_negative, map_and_truncated_cache; thread=false)
        map_and_truncated = PauliSum(extractsum!(map_and_truncated_cache))
        @test length(map_and_truncated) == count(coeff > 0 for (_, coeff) in dict_sum)
        @test all(getcoeff(map_and_truncated, pstr) ≈ 0.5 * coeff for (pstr, coeff) in dict_sum if coeff > 0)
    end

    # on a sorted array sum, none of the passes disturbs the sorted prefix
    vector_sum = VectorPauliSum(dict_sum)
    sortterms!(vector_sum)
    prop_cache = PropagationCache(vector_sum)
    PauliPropagation.PropagationBase.setsortedprefix!(mainsum(prop_cache), length(prop_cache))

    PauliPropagation.PropagationBase.xorbranch!(rotate, prop_cache, gate_mask; thread=false)
    @test PauliPropagation.PropagationBase.sortedprefix(mainsum(prop_cache)) == length(prop_cache)
    @test issorted(PauliPropagation.PropagationBase.terms(prop_cache))

    mapcoeffsbypair!(halve_positive, prop_cache; thread=false)
    @test PauliPropagation.PropagationBase.sortedprefix(mainsum(prop_cache)) == length(prop_cache)
    @test issorted(PauliPropagation.PropagationBase.terms(prop_cache))

    filter!(keep_positive, prop_cache; thread=false)
    @test PauliPropagation.PropagationBase.sortedprefix(mainsum(prop_cache)) == length(prop_cache)
    @test issorted(PauliPropagation.PropagationBase.terms(prop_cache))

    # the kernels for arrays that are not on the CPU agree with the CPU kernels
    cpu_cache = PropagationCache(VectorPauliSum(dict_sum))
    portable_cache = PropagationCache(VectorPauliSum(dict_sum))
    PauliPropagation.PropagationBase._branchcpu!(rotate, cpu_cache, gate_mask; thread=false)
    PauliPropagation.PropagationBase._branchflagged!(rotate, portable_cache, gate_mask; thread=false)
    @test PauliPropagation.PropagationBase.terms(cpu_cache) == PauliPropagation.PropagationBase.terms(portable_cache)
    @test coefficients(cpu_cache) == coefficients(portable_cache)

    PauliPropagation.PropagationBase._filtercpu!(keep_positive, cpu_cache; thread=false)
    PauliPropagation.PropagationBase.flag!(keep_positive, portable_cache; thread=false)
    PauliPropagation.PropagationBase.filterviaflags!(portable_cache; thread=false)
    @test PauliPropagation.PropagationBase.terms(cpu_cache) == PauliPropagation.PropagationBase.terms(portable_cache)
    @test coefficients(cpu_cache) == coefficients(portable_cache)
end

@testset "Flat map primitive" begin
    nq = 4
    rng = MersenneTwister(11)
    dict_sum = PauliSum(nq)
    for _ in 1:100
        add!(dict_sum, symboltoint(rand(rng, (:I, :X, :Y, :Z), nq)), randn(rng))
    end

    # a Z on the first qubit branches into the term and a partner, an identity there drops the term
    x_mask = symboltoint(paulitype(dict_sum), :X, 1)
    function branch_or_drop(pstr, coeff)
        pauli = getpauli(pstr, 1)
        if pauli == 0
            return ()
        elseif pauli == 3
            return ((pstr, 0.5 * coeff), (pstr ⊻ x_mask, 0.25 * coeff))
        else
            return ((pstr, coeff),)
        end
    end

    reference = PauliSum(nq)
    n_pairs = 0
    for (pstr, coeff) in dict_sum
        for (new_pstr, new_coeff) in branch_or_drop(pstr, coeff)
            add!(reference, new_pstr, new_coeff)
            n_pairs += 1
        end
    end
    @test length(reference) < n_pairs

    for makesum in (identity, VectorPauliSum, psum -> MultiPauliSum(VectorPauliSum(psum), 4), psum -> MultiPauliSum(psum, 4))
        input = makesum(dict_sum)
        mapped = PauliPropagation.PropagationBase.flatmap(branch_or_drop, input)
        @test PauliSum(mapped) ≈ reference
        @test PauliSum(input) == dict_sum
        @test mapped isa typeof(input)

        cache = PropagationCache(deepcopy(makesum(dict_sum)))
        PauliPropagation.PropagationBase.flatmap!(branch_or_drop, cache)
        @test PauliSum(PauliPropagation.PropagationBase.extractsum!(cache)) ≈ reference
    end

    # the pairs of an array sum are not merged, and outgrow the room the cache had
    vector_cache = PropagationCache(VectorPauliSum(dict_sum))
    PauliPropagation.PropagationBase.flatmap!(branch_or_drop, vector_cache; thread=false)
    @test length(vector_cache) == n_pairs
    @test PauliPropagation.PropagationBase.sortedprefix(mainsum(vector_cache)) == 0

    # the kernels for arrays that are not on the CPU agree with the CPU kernels
    portable_cache = PropagationCache(VectorPauliSum(dict_sum))
    PauliPropagation.PropagationBase._flatmapflagged!(branch_or_drop, portable_cache; thread=false)
    PauliPropagation.PropagationBase._commitwrite!(portable_cache, n_pairs, 0)
    @test PauliPropagation.PropagationBase.terms(vector_cache) == PauliPropagation.PropagationBase.terms(portable_cache)
    @test coefficients(vector_cache) == coefficients(portable_cache)

    # several tasks count the pairs before writing them
    many_terms = VectorPauliSum(nq, rand(rng, UInt8, 40_000), randn(rng, 40_000))
    many_reference = PauliPropagation.PropagationBase.flatmap(branch_or_drop, many_terms; thread=false)
    many_mapped = PauliPropagation.PropagationBase.flatmap(branch_or_drop, many_terms; thread=true)
    @test PauliPropagation.PropagationBase.terms(many_mapped) == PauliPropagation.PropagationBase.terms(many_reference)
    @test coefficients(many_mapped) == coefficients(many_reference)
end

@testset "Unknown storage primitive defaults" begin
    # This storage deliberately has no primitive-specific methods.  It only implements the basic
    # term-sum operations that the serial fallback paths require.
    struct FallbackStorage <: PB.StorageType end

    mutable struct FallbackPauliSum <: PP.AbstractPauliSum
        nqubits::Int
        data::Dict{UInt8,Float64}
    end

    PB.StorageType(::FallbackPauliSum) = FallbackStorage()
    PB.storage(psum::FallbackPauliSum) = psum.data
    PB.nsites(psum::FallbackPauliSum) = psum.nqubits
    PP.nqubits(psum::FallbackPauliSum) = psum.nqubits
    PB._terms(::FallbackStorage, psum::FallbackPauliSum) = keys(psum.data)
    PB._coefficients(::FallbackStorage, psum::FallbackPauliSum) = values(psum.data)
    PB._add!(::FallbackStorage, psum::FallbackPauliSum, term, coefficient) =
        (psum.data[term] = get(psum.data, term, 0.0) + coefficient; psum)
    PB._empty!(::FallbackStorage, psum::FallbackPauliSum) = (empty!(psum.data); psum)

    function fallback_sum()
        return FallbackPauliSum(3, Dict{UInt8,Float64}(
            0x01 => 0.8,
            0x03 => -0.3,
            0x0c => 0.4,
        ))
    end

    pauli_sum(psum::FallbackPauliSum) = PauliSum(psum.nqubits, copy(psum.data))
    function matches(reference, result)
        @test length(result) == length(reference)
        @test all(getcoeff(result, term) ≈ coefficient for (term, coefficient) in reference)
    end

    # These gates exercise the broad specializations through their primitive fallbacks: map!,
    # mapcoeffsbypair!, and xorbranch!, respectively.
    for (gate, parameter) in (
        (CliffordGate(:H, 1), nothing),
        (DepolarizingNoise(1), 0.4),
        (PauliRotation(:Z, 1), 0.3),
        (AmplitudeDampingNoise(1), 0.2),
    )
        input = fallback_sum()
        reference_input = pauli_sum(input)
        reference = isnothing(parameter) ?
            propagate(gate, reference_input; min_abs_coeff=0.0) :
            propagate(gate, reference_input, parameter; min_abs_coeff=0.0)
        result = isnothing(parameter) ?
            propagate(gate, input; min_abs_coeff=0.0) :
            propagate(gate, input, parameter; min_abs_coeff=0.0)
        matches(reference, result)
    end

    # The cache-level defaults expose the main sum and combine an explicitly populated auxiliary
    # sum through the public add! contract.
    cache = PropagationCache(fallback_sum())
    add!(auxsum(cache), 0x01, 0.2)
    merge!(cache)
    @test getcoeff(mainsum(cache), 0x01) ≈ 1.0

    mapcoeffs!(coeff -> 2 * coeff, cache)
    @test getcoeff(mainsum(cache), 0x01) ≈ 2.0
end

@testset "Gates written for every cache dispatch without ties" begin
    # a custom gate defined on the abstract Pauli cache runs on every storage, since the primitives
    # it is written with dispatch on the storage of the cache
    struct HalvingGate <: StaticGate end
    PauliPropagation.PropagationBase.applytoall!(::HalvingGate, prop_cache::PauliPropagation.AbstractPauliPropagationCache; kwargs...) =
        mapcoeffs!(coeff -> 0.5 * coeff, prop_cache)

    psum = PauliSum(PauliString(3, [:X, :Z], [1, 3], 0.8))
    for makesum in (identity, VectorPauliSum, psum -> MultiPauliSum(VectorPauliSum(psum), 2))
        out = propagate([HalvingGate()], makesum(psum))
        @test getcoeff(out, [:X, :Z], [1, 3]) ≈ 0.4
    end

    # a custom gate that only defines `apply` takes the generic path, `flatmap!` over every storage
    struct SwappingGate <: StaticGate end
    PauliPropagation.PropagationBase.apply(::SwappingGate, pstr, coeff; kwargs...) =
        ((setpauli(setpauli(pstr, getpauli(pstr, 3), 1), getpauli(pstr, 1), 3), coeff),)

    for makesum in (identity, VectorPauliSum, psum -> MultiPauliSum(VectorPauliSum(psum), 2), psum -> MultiPauliSum(psum, 2))
        out = propagate([SwappingGate()], makesum(psum))
        @test getcoeff(out, [:Z, :X], [1, 3]) ≈ 0.8
    end

    # a frozen gate reaches the `applymergetruncate!` of the gate it froze: the normalization of an
    # imaginary rotation, and the fused damping and truncation of Pauli noise
    imaginary_state = PauliSum(3)
    add!(imaginary_state, :I, 1, 1.0)
    add!(imaginary_state, [:X, :Y], [1, 2], 0.5)
    for makesum in (identity, VectorPauliSum)
        unfrozen = propagate(ImaginaryPauliRotation(:X, 1), makesum(imaginary_state), 0.3; heisenberg=false)
        frozen = propagate(FrozenGate(ImaginaryPauliRotation(:X, 1), 0.3), makesum(imaginary_state); heisenberg=false)
        @test PauliSum(frozen) ≈ PauliSum(unfrozen)
        @test getcoeff(frozen, 0) ≈ 1.0

        unfrozen = propagate(DepolarizingNoise(1), makesum(psum), 0.5; min_abs_coeff=0.3)
        frozen = propagate(DepolarizingNoise(1, 0.5), makesum(psum); min_abs_coeff=0.3)
        @test PauliSum(frozen) ≈ PauliSum(unfrozen)
    end

    @test isempty(Test.detect_ambiguities(PauliPropagation, PauliPropagation.PropagationBase, PauliPropagation.Performance))
end
