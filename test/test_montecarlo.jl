using Test
using PauliPropagation
const PP = PauliPropagation
using Random


@testset "mcsample!/mcpropagate! in-place mechanics" begin
    nq = 2
    gate = PauliRotation(:X, 1)
    theta = 0.3
    pstr = PauliString(nq, :Z, 1)

    # mcsample! only mutates the active view in place: sums are never swapped or resized
    prop_cache = PropagationCache(VectorPauliSum(pstr))
    main_before, aux_before = mainsum(prop_cache), auxsum(prop_cache)
    @test mcsample!(gate, prop_cache, theta) === prop_cache
    @test mainsum(prop_cache) === main_before
    @test auxsum(prop_cache) === aux_before

    # bare gate and bare parameter get promoted to [gate]/[theta], as for propagate!
    psum = VectorPauliSum(pstr)
    @test mcsample!(gate, psum, theta) === psum

    original = VectorPauliSum(pstr)
    result = mcsample(gate, original, theta)
    @test result !== original
    @test length(original) == 1
    @test getcoeff(original, :Z, 1) ≈ 1.0

    # mcpropagate! reduces to a single deterministic propagate! step here since max_size is never hit
    psum2 = VectorPauliSum(pstr)
    @test mcpropagate!(gate, psum2, theta; max_size=100) === psum2

    original2 = VectorPauliSum(pstr)
    result2 = mcpropagate(gate, original2, theta; max_size=100)
    @test result2 !== original2
    @test length(original2) == 1
    @test getcoeff(original2, :Z, 1) ≈ 1.0
end


@testset "mcapplytoall! for CliffordGate and FrozenGate" begin
    nq = 1
    gate = CliffordGate(:H, [1])
    term = symboltoint(nq, :Y, 1)

    # H Y H = -Y, so this gate is a case where conjugation flips the coefficient's sign
    exact_term, exact_coeff = only(apply(gate, term, 1.0, clifford_map[gate.symbol]))
    @test exact_coeff ≈ -1.0

    # squared=false: Clifford application is deterministic and matches apply() exactly
    psum1 = VectorPauliSum(nq, [term], [1.0])
    mcapplytoall!(gate, psum1)
    @test only(paulis(psum1)) == exact_term
    @test only(coefficients(psum1)) ≈ exact_coeff

    # squared=true: the sign flip is undone so 2-norm sampling never picks up a spurious sign
    psum2 = VectorPauliSum(nq, [term], [1.0])
    mcapplytoall!(gate, psum2; squared=true)
    @test only(paulis(psum2)) == exact_term
    @test only(coefficients(psum2)) ≈ 1.0

    # FrozenGate wraps a ParametrizedGate with a fixed parameter and just redirects to it
    theta = 0.42
    rot = PauliRotation(:X, 1)
    frozen = FrozenGate(rot, theta)
    rot_term = symboltoint(nq, :Z, 1)

    Random.seed!(123)
    psum_a = VectorPauliSum(nq, [rot_term], [1.0])
    mcapplytoall!(rot, psum_a, theta)

    Random.seed!(123)
    psum_b = VectorPauliSum(nq, [rot_term], [1.0])
    mcapplytoall!(frozen, psum_b)

    @test psum_a == psum_b
end


@testset "mcapplytoall! for PauliRotation matches the exact branch formula" begin
    nq = 2
    gate = PauliRotation([:X, :Z], [1, 2])
    theta = 0.37
    term = symboltoint(nq, [:Z, :Z], [1, 2])
    gate_mask = symboltoint(nq, gate.symbols, gate.qinds)
    @test !commutes(gate_mask, term)

    # a single call keeps only one of the two exact branches, boosted by 1/probability so that
    # averaging many independent calls reproduces the deterministic split in expectation
    cos_val, sin_val = cos(theta), sin(theta)
    normalization = abs(cos_val) + abs(sin_val)
    new_term, prod_sign = PauliPropagation.paulirotationproduct(gate_mask, term)

    seen_stay, seen_flip = false, false
    for _ in 1:200
        psum = VectorPauliSum(nq, [term], [1.0])
        mcapplytoall!(gate, psum, theta)
        t, c = only(paulis(psum)), only(coefficients(psum))
        if t == term
            @test c ≈ normalization * sign(cos_val)
            seen_stay = true
        else
            @test t == new_term
            @test c ≈ normalization * sign(sin_val) * prod_sign
            seen_flip = true
        end
    end
    @test seen_stay && seen_flip

    # a commuting term is left completely untouched
    commuting_term = symboltoint(nq, :X, 1)
    @test commutes(gate_mask, commuting_term)
    psum = VectorPauliSum(nq, [commuting_term], [1.0])
    mcapplytoall!(gate, psum, theta)
    @test only(paulis(psum)) == commuting_term
    @test only(coefficients(psum)) ≈ 1.0
end


@testset "mcapplytoall! for PauliRotation exactly preserves |coeff| (squared=true)" begin
    nq = 2
    gate = PauliRotation([:X, :Z], [1, 2])
    theta = 0.9
    term = symboltoint(nq, [:Z, :Z], [1, 2])
    gate_mask = symboltoint(nq, gate.symbols, gate.qinds)
    new_term, _ = PauliPropagation.paulirotationproduct(gate_mask, term)

    # for squared=true, normalization = cos^2 + sin^2 = 1 exactly, so every branch leaves the
    # coefficient bit-for-bit unchanged and only randomizes which term it is attached to
    seen_stay, seen_flip = false, false
    for _ in 1:200
        psum = VectorPauliSum(nq, [term], [2.5])
        mcapplytoall!(gate, psum, theta; squared=true)
        t, c = only(paulis(psum)), only(coefficients(psum))
        @test c == 2.5
        seen_stay |= (t == term)
        seen_flip |= (t == new_term)
    end
    @test seen_stay && seen_flip
end


@testset "mcsample! statistically reproduces propagate" begin
    nq = 3
    nl = 1
    # mixes CliffordGate (CNOT) and PauliRotation gates
    circuit = efficientsu2circuit(nq, nl)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 2)

    exact_psum = propagate(circuit, pstr, thetas)

    reps = 100
    init_psum = VectorPauliSum(nq, fill(pstr.term, reps), fill(float(pstr.coeff), reps))
    sampled_psum = mcsample(circuit, init_psum, thetas)
    merge!(sampled_psum)
    mult!(sampled_psum, 1 / reps)

    @test overlapwithzero(sampled_psum) ≈ overlapwithzero(exact_psum) atol = 0.2
end


@testset "mcsample! exactly conserves the squared 2-norm (squared=true)" begin
    nq = 4
    nl = 3
    circuit = efficientsu2circuit(nq, nl)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 2)

    # each walker's |coeff|^2 is individually preserved by every gate (Clifford and rotation
    # alike), so the ensemble's squared 2-norm is an exact invariant of squared=true sampling
    psum = VectorPauliSum(nq, fill(pstr.term, 50), fill(1.0, 50))
    norm_before = sum(abs2, coefficients(psum))
    mcsample!(circuit, psum, thetas; squared=true)
    norm_after = sum(abs2, coefficients(psum))

    @test norm_after ≈ norm_before
end


@testset "mcapplytoall! is written for every Pauli sum" begin
    nq = 3
    gate = PauliRotation([:X, :X], [1, 2])
    gate_mask = symboltoint(nq, [:X, :X], [1, 2])
    theta = 0.7
    terms = [symboltoint(nq, :Z, 1), symboltoint(nq, :Z, 3), symboltoint(nq, [:Y, :Z], [2, 3])]
    branches = Set(vcat(terms, [first(PP.paulirotationproduct(gate_mask, term)) for term in terms]))

    # every term keeps one of its two branches, with unit weight when sampling squared coefficients
    for psum in (PauliSum(nq, Dict(term => 1.0 for term in terms)), VectorPauliSum(nq, copy(terms), ones(3)))
        sampled = mcapplytoall!(gate, psum, theta; squared=true, thread=false)
        @test length(sampled) == 3
        @test all(term in branches for (term, _) in sampled)
        @test all(abs(coeff) ≈ 1 for coeff in coefficients(sampled))
    end

    # path properties count the branch a term keeps, and a Clifford gate counts nothing
    Random.seed!(7)
    plain = mcapplytoall!(gate, VectorPauliSum(nq, copy(terms), ones(3)), theta; thread=false)
    Random.seed!(7)
    tracked = mcapplytoall!(gate, VectorPauliSum(nq, copy(terms), PauliFreqTracker.(ones(3))), theta; thread=false)

    @test paulis(tracked) == paulis(plain)
    @test [coeff.coeff for coeff in coefficients(tracked)] ≈ coefficients(plain)
    for (term, coeff) in zip(terms, coefficients(tracked))
        @test coeff.freq == coeff.nsins + coeff.ncos == !commutes(gate_mask, term)
    end

    counts = [(coeff.nsins, coeff.ncos, coeff.freq) for coeff in coefficients(tracked)]
    mcapplytoall!(CliffordGate(:H, 1), plain; thread=false)
    mcapplytoall!(CliffordGate(:H, 1), tracked; thread=false)
    @test [coeff.coeff for coeff in coefficients(tracked)] ≈ coefficients(plain)
    @test [(coeff.nsins, coeff.ncos, coeff.freq) for coeff in coefficients(tracked)] == counts
end

@testset "mcsample converts PauliString and PauliSum inputs" begin
    nq = 3
    circuit = efficientsu2circuit(nq, 1)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 2)
    psum = PauliSum(pstr)

    # sampling keeps one branch per gate, so a single walker stays a single term
    from_pstr = mcsample(circuit, pstr, thetas)
    @test from_pstr isa PauliString
    @test nqubits(from_pstr) == nq

    from_psum = mcsample(circuit, psum, thetas)
    @test from_psum isa PauliSum
    @test length(from_psum) == 1

    # the input is converted, not consumed
    @test length(psum) == 1
    @test getcoeff(psum, pstr.term) ≈ 1.0

    @test_throws ArgumentError mcsample!(circuit, pstr, thetas)
    @test_throws ArgumentError mcsample!(circuit, psum, thetas)
end


@testset "mcpropagate! matches propagate! exactly below max_size" begin
    nq = 4
    nl = 3
    circuit = efficientsu2circuit(nq, nl)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 2)

    exact_psum = propagate(circuit, pstr, thetas; min_abs_coeff=0)
    # max_size effectively infinite: applymergetruncateresample! never resamples,
    # so this is bit-for-bit the same computation as propagate!
    mc_psum = mcpropagate(circuit, VectorPauliSum(pstr), thetas; max_size=10^9, min_abs_coeff=0)

    @test length(mc_psum) == length(exact_psum)
    for (term, coeff) in zip(paulis(mc_psum), coefficients(mc_psum))
        @test coeff == getcoeff(exact_psum, term)
    end
end


@testset "mcpropagate! bounds the ensemble size" begin
    nq = 5
    nl = 4
    circuit = efficientsu2circuit(nq, nl)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 2)
    max_size = 10

    result = mcpropagate(circuit, VectorPauliSum(pstr), thetas; max_size, min_abs_coeff=1e-8)
    @test !isempty(result)
    @test length(result) <= max_size

    # squared through a circuit containing Clifford gates should also run cleanly
    result2 = mcpropagate(circuit, VectorPauliSum(pstr), thetas; max_size, squared=true)
    @test !isempty(result2)
    @test length(result2) <= max_size
end


@testset "mcpropagate converts PauliString and PauliSum inputs" begin
    nq = 4
    nl = 3
    circuit = efficientsu2circuit(nq, nl)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 2)
    psum = PauliSum(pstr)

    exact_psum = propagate(circuit, pstr, thetas; min_abs_coeff=0)

    # max_size effectively infinite, so both conversions must reproduce propagate exactly
    from_pstr = mcpropagate(circuit, pstr, thetas; max_size=10^9, min_abs_coeff=0)
    @test from_pstr isa VectorPauliSum
    @test length(from_pstr) == length(exact_psum)
    for (term, coeff) in zip(paulis(from_pstr), coefficients(from_pstr))
        @test coeff == getcoeff(exact_psum, term)
    end

    from_psum = mcpropagate(circuit, psum, thetas; max_size=10^9, min_abs_coeff=0)
    @test from_psum isa PauliSum
    @test length(from_psum) == length(exact_psum)
    for (term, coeff) in zip(paulis(from_psum), coefficients(from_psum))
        @test coeff == getcoeff(exact_psum, term)
    end

    # the input is converted, not consumed
    @test length(psum) == 1
    @test getcoeff(psum, pstr.term) ≈ 1.0

    # resampling still bounds the ensemble when it runs through the conversions
    max_size = 10
    @test length(mcpropagate(circuit, pstr, thetas; max_size, min_abs_coeff=1e-8)) <= max_size
    @test length(mcpropagate(circuit, psum, thetas; max_size, min_abs_coeff=1e-8)) <= max_size

    @test_throws ArgumentError mcpropagate!(circuit, pstr, thetas; max_size=10^9)
    @test_throws ArgumentError mcpropagate!(circuit, psum, thetas; max_size=10^9)
end


@testset "resample!/resample preserve total weight and respect target_size" begin
    nq = 4
    pstrs = [PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq), rand() + 0.1) for _ in 1:30]
    psum = merge!(VectorPauliSum(pstrs))
    n = length(psum)
    target_size = max(1, n ÷ 2)

    # systematic_resample!'s comb step is quantized and randomly offset, so both the survivor
    # count and the weight it carries land close to, but not always exactly at, their targets
    term_tol = 3

    prop_cache = PropagationCache(deepcopy(psum))
    total_before = sum(PP.activecoeffs(prop_cache))
    resample!(prop_cache, target_size; resample_func=PP.systematic_resample!)
    @test abs(PP.activesize(prop_cache) - target_size) <= term_tol
    @test isapprox(sum(PP.activecoeffs(prop_cache)), total_before; atol=term_tol * total_before / target_size)

    # target_size equal to the current size is allowed, only exceeding it is an error
    same_size_cache = PropagationCache(deepcopy(psum))
    resample!(same_size_cache, n; resample_func=PP.systematic_resample!)
    @test abs(PP.activesize(same_size_cache) - n) <= term_tol

    over_cache = PropagationCache(deepcopy(psum))
    @test_throws ArgumentError resample!(over_cache, n + 1)

    # out-of-place resample leaves the input psum untouched
    original = deepcopy(psum)
    result = resample(psum, target_size; resample_func=PP.systematic_resample!)
    @test psum == original
    @test abs(length(result) - target_size) <= term_tol
end


@testset "mcpropagate! handles complex coefficients" begin
    nq = 4
    nl = 3
    circuit = efficientsu2circuit(nq, nl)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 2, 1.0 + 0.5im)

    result = mcpropagate(circuit, VectorPauliSum(pstr), thetas; max_size=10, min_abs_coeff=1e-8)
    @test !isempty(result)
    @test length(result) <= 10

    # squared=true routes through multinomial_resample!
    result_sq = mcpropagate(circuit, VectorPauliSum(pstr), thetas; max_size=10, squared=true)
    @test !isempty(result_sq)
    @test length(result_sq) <= 10
end


@testset "resample! variants handle complex coefficients" begin
    nq = 4
    pstrs = [PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq), (rand() + 0.1) * cis(2π * rand())) for _ in 1:30]
    base_psum = merge!(VectorPauliSum(pstrs))
    n = length(base_psum)
    target_size = max(1, n ÷ 2)
    term_tol = 3

    # multinomial_resample! keeps every term drawn, so at most target_size terms survive
    cache = PropagationCache(deepcopy(base_psum))
    resample!(cache, target_size; resample_func=PP.multinomial_resample!)
    @test 1 <= PP.activesize(cache) <= target_size

    for f in (PP.systematic_resample!, PP.semideterministic_systematic_resample!)
        cache = PropagationCache(deepcopy(base_psum))
        resample!(cache, target_size; resample_func=f)
        @test 1 <= PP.activesize(cache) <= target_size + term_tol
    end
end


@testset "resample! variants stay within target_size" begin
    nq = 4
    pstrs = [PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq), rand() + 0.1) for _ in 1:30]
    base_psum = merge!(VectorPauliSum(pstrs))
    n = length(base_psum)
    target_size = max(1, n ÷ 2)
    term_tol = 3

    # multinomial_resample! keeps every term drawn, so at most target_size terms survive
    cache = PropagationCache(deepcopy(base_psum))
    resample!(cache, target_size; resample_func=PP.multinomial_resample!)
    @test 1 <= PP.activesize(cache) <= target_size

    # the deduplicating variants' comb step is quantized, so the survivor count can land a
    # few terms above target_size, and may also land well below it if many terms deduplicate
    for f in (PP.systematic_resample!, PP.semideterministic_systematic_resample!)
        cache = PropagationCache(deepcopy(base_psum))
        resample!(cache, target_size; resample_func=f)
        @test 1 <= PP.activesize(cache) <= target_size + term_tol
    end
end

@testset "resample takes a PauliSum" begin
    nq = 4
    pstrs = [PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq), rand() + 0.1) for _ in 1:30]
    psum = PauliSum(pstrs)
    n = length(psum)
    target_size = max(1, n ÷ 2)

    resampled = resample(psum, target_size)
    @test resampled isa PauliSum
    @test 1 <= length(resampled) <= target_size + 3
    @test sum(abs, coefficients(resampled)) ≈ sum(abs, coefficients(psum)) rtol = 0.3

    # out of place leaves the input untouched, in place resamples it
    @test length(psum) == n
    resample!(psum, target_size)
    @test 1 <= length(psum) <= target_size + 3
end


@testset "resample! forwards squared to the resampler" begin
    nq = 4
    pstrs = [PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq), rand() + 0.1) for _ in 1:30]
    base_psum = merge!(VectorPauliSum(pstrs))
    target_size = 5

    # under squared=true, multinomial_resample! gives every survivor its draws' share of the *squared*
    # 2-norm, |coeff|^2 = n_draws * sum(abs2) / target_size, which conserves that norm. If squared were
    # dropped on the way to the resampler, the 1-norm would be conserved instead.
    cache = PropagationCache(deepcopy(base_psum))
    resample!(cache, target_size; resample_func=PP.multinomial_resample!, squared=true)
    @test sum(abs2, PP.activecoeffs(cache)) ≈ sum(abs2, coefficients(base_psum))
end


@testset "mcpropagate! on a MultiPauliSum runs zone by zone" begin
    nq = 5
    nl = 4
    circuit = efficientsu2circuit(nq, nl)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 2)

    exact_psum = propagate(circuit, pstr, thetas; min_abs_coeff=0)

    # with either kind of zone, and below max_size bit-for-bit the same computation as propagate!
    for seed in (VectorPauliSum(pstr), PauliSum(pstr)), n_zones in (1, 4)
        msum = mcpropagate(circuit, MultiPauliSum(seed, n_zones), thetas; max_size=10^9, min_abs_coeff=0)
        @test msum isa MultiPauliSum
        @test length(msum) == length(exact_psum)
        @test all(coeff == getcoeff(exact_psum, term) for (term, coeff) in msum)

        # resampling bounds the ensemble, through either resampling strategy
        max_size = 10
        for squared in (false, true)
            result = mcpropagate(circuit, MultiPauliSum(seed, n_zones), thetas; max_size, squared, min_abs_coeff=1e-8)
            @test !isempty(result)
            @test length(result) <= max_size
        end
    end
end


@testset "resample! on a MultiPauliSum resamples zone by zone" begin
    nq = 6
    circuit = efficientsu2circuit(nq, 3)
    thetas = randn(countparameters(circuit))
    psum = propagate(circuit, PauliString(nq, :Z, 2), thetas; min_abs_coeff=0)
    n = length(psum)
    target_size = n ÷ 3
    # the calibrated comb aims the expected survivor count at target_size, and the count fluctuates around it
    term_tol = target_size ÷ 20
    total_weight = sum(abs, coefficients(psum))

    for seed in (VectorPauliSum(psum), psum), thread in (true, false)
        msum = MultiPauliSum(seed, 4)

        # the comb of the systematic strategies conserves the total weight, and only keeps terms of the sum
        for f in (PP.semideterministic_systematic_resample!, PP.systematic_resample!, PP.multinomial_resample!)
            cache = PropagationCache(deepcopy(msum))
            resample!(cache, target_size; resample_func=f, thread)
            @test 1 <= length(cache) <= target_size + term_tol
            @test sum(abs, coefficients(cache)) ≈ total_weight rtol = 0.01
            @test all(sign(coeff) == sign(getcoeff(psum, term)) for (term, coeff) in zip(paulis(cache), coefficients(cache)))
        end

        # the resampled coefficients conserve the squared 2-norm when resampling squared
        squared_norm = sum(abs2, coefficients(psum))
        for f in (PP.systematic_resample!, PP.multinomial_resample!)
            cache = PropagationCache(deepcopy(msum))
            resample!(cache, target_size; resample_func=f, squared=true, thread)
            @test sum(abs2, coefficients(cache)) ≈ squared_norm rtol = 0.01
        end
        cache = PropagationCache(deepcopy(msum))
        @test_throws ArgumentError resample!(cache, target_size; squared=true, resample_func=PP.semideterministic_systematic_resample!)

        # out-of-place resampling leaves the input untouched and returns the same type
        result = resample(msum, target_size)
        @test result isa MultiPauliSum
        @test length(msum) == n
        @test 1 <= length(result) <= target_size + term_tol
    end

    # the terms stay where they are, so sorted vector zones stay sorted
    cache = PropagationCache(MultiPauliSum(VectorPauliSum(psum), 4))
    merge!(cache)
    resample!(cache, target_size)
    @test all(PP.sortedprefix(mainsum(zonecache)) == PP.activesize(zonecache) for zonecache in PP.zonecaches(cache))

    # every zone keeps only what it owns
    for (zone_id, zone) in enumerate(zones(PP.activesum(cache)))
        @test all(zoneof(mainsum(cache), term) == zone_id for term in paulis(zone))
    end
end


@testset "mapslotsandtruncate! gives the same slots on any number of tasks" begin
    PB = PP.PropagationBase
    n = 4 * PB._MIN_ELEMS_PER_TASK
    rng = MersenneTwister(11)
    input_terms = UInt64.(1:n)
    # dyadic coefficients sum exactly in any order, so every split of the terms lays the same slots
    input_coeffs = [rand(rng, (-1, 1)) * rand(rng, 1:64) / 8 for _ in 1:n]
    n_sorted = n ÷ 3
    comb_step = sum(abs, input_coeffs) / (n ÷ 2)
    new_coeff_func(coeff, slot_start, slot_end) = PB._compute_new_coeff(PB._count_combteeth(comb_step, comb_step / 3, slot_start, slot_end), comb_step, coeff, false)

    serial = PropagationCache(VectorPauliSum(32, copy(input_terms), copy(input_coeffs), n_sorted))
    PP.mapslotsandtruncate!(abs, new_coeff_func, PB._truncatezero, serial; thread=false)
    @test 0 < length(serial) < n

    task_partitioner = PB.AK.TaskPartitioner(n, 4, 1)
    in_tasks = PropagationCache(VectorPauliSum(32, copy(input_terms), copy(input_coeffs), n_sorted))
    PB._mapslotsintasks!(abs, new_coeff_func, PB._truncatezero, in_tasks, task_partitioner, task_partitioner.num_tasks)
    @test PB.activeterms(in_tasks) == PB.activeterms(serial)
    @test PB.activecoeffs(in_tasks) == PB.activecoeffs(serial)

    # the terms are their own positions, so the kept ones among the first n_sorted are the sorted prefix
    n_sorted_kept = count(<=(n_sorted), PB.activeterms(serial))
    @test PB.sortedprefix(mainsum(serial)) == n_sorted_kept
    @test PB.sortedprefix(mainsum(in_tasks)) == n_sorted_kept

    # a new coefficient can be costly to find, as for multinomial draws, so every term asks for one only once
    n_calls = Threads.Atomic{Int}(0)
    function counted_new_coeff_func(coeff, slot_start, slot_end)
        Threads.atomic_add!(n_calls, 1)
        return new_coeff_func(coeff, slot_start, slot_end)
    end
    counted = PropagationCache(VectorPauliSum(32, copy(input_terms), copy(input_coeffs), n_sorted))
    PB._mapslotsintasks!(abs, counted_new_coeff_func, PB._truncatezero, counted, task_partitioner, task_partitioner.num_tasks)
    @test n_calls[] == n

    # without a truncation, every term keeps its place and gets the coefficient of its slot
    serial = PropagationCache(VectorPauliSum(32, copy(input_terms), copy(input_coeffs), n_sorted))
    PP.mapslots!(abs, new_coeff_func, serial; thread=false)
    in_tasks = PropagationCache(VectorPauliSum(32, copy(input_terms), copy(input_coeffs), n_sorted))
    PB._mapslotsintasks!(abs, new_coeff_func, nothing, in_tasks, task_partitioner, task_partitioner.num_tasks)
    @test PB.activeterms(serial) == PB.activeterms(in_tasks) == input_terms
    @test PB.activecoeffs(in_tasks) == PB.activecoeffs(serial)
    @test count(iszero, PB.activecoeffs(serial)) > 0
    @test PB.sortedprefix(mainsum(serial)) == PB.sortedprefix(mainsum(in_tasks)) == n_sorted
end


@testset "the calibrated comb step keeps the target in expectation" begin
    PB = PP.PropagationBase
    n = 4 * PB._MIN_ELEMS_PER_TASK
    rng = MersenneTwister(5)
    # log-normal magnitudes over several decades, as in propagated sums
    coeffs = randn(rng, n) .* exp.(1.5 .* randn(rng, n))
    vpsum = VectorPauliSum(32, UInt64.(1:n), coeffs)
    total_weight = sum(abs, coeffs)
    expected_n_unique(step) = sum(coeff -> min(1.0, abs(coeff) / step), coeffs)

    # every storage lands within the tolerance below the target, also when nearly every term has to survive
    for makesum in (identity, vps -> MultiPauliSum(vps, 4), PauliSum), target_size in (n ÷ 10, n ÷ 2, n - n ÷ 100)
        cache = PropagationCache(makesum(deepcopy(vpsum)))
        step = PB._calibrate_prob_step(abs, cache, total_weight, target_size; rtol=0.01, atol=0, thread=true)
        @test 0.99 * target_size <= expected_n_unique(step) <= target_size * (1 + 1e-9)
    end
end


@testset "one pass counts and weighs as two reductions do, on every storage" begin
    PB = PP.PropagationBase
    n = 4 * PB._MIN_ELEMS_PER_TASK
    rng = MersenneTwister(6)
    coeffs = randn(rng, n)
    vpsum = VectorPauliSum(32, UInt64.(1:n), coeffs)
    is_positive(coeff) = coeff > 0
    negative_weight(coeff) = is_positive(coeff) ? 0.0 : abs(coeff)

    for makesum in (identity, vps -> MultiPauliSum(vps, 4), PauliSum, vps -> MultiPauliSum(PauliSum(vps), 4)), thread in (true, false)
        n_positive, weight = PB._countandweigh(is_positive, negative_weight, PropagationCache(makesum(deepcopy(vpsum))); thread)
        @test n_positive == count(is_positive, coeffs)
        @test weight ≈ sum(negative_weight, coeffs)
    end
end
