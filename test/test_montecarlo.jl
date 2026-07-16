using Test
using PauliPropagation
using PauliPropagation.PropagationBase
using PauliPropagation: multinomial_resample!, systematic_resample!, systematic_resample_merged!,
    semideterministic_systematic_resample_merged!, detfraction_systematic_resample_merged!, paulirotationproduct
using Random


@testset "mcsample!/mcpropagate! in-place mechanics" begin
    nq = 2
    gate = PauliRotation(:X, 1)
    theta = 0.3
    pstr = PauliString(nq, :Z, 1)

    # mcsample! only mutates the active view in place: sums are never swapped or resized
    prop_cache = VectorPauliPropagationCache(VectorPauliSum(pstr))
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
    new_term, prod_sign = paulirotationproduct(gate_mask, term)

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
    new_term, _ = paulirotationproduct(gate_mask, term)

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
    nl = 2
    # mixes CliffordGate (CNOT) and PauliRotation gates
    circuit = efficientsu2circuit(nq, nl)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 2)

    exact_psum = propagate(circuit, pstr, thetas)

    reps = 20_000
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


@testset "resample!/resample preserve total weight and respect target_size" begin
    nq = 4
    pstrs = [PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq), rand() + 0.1) for _ in 1:30]
    psum = merge!(VectorPauliSum(pstrs))
    n = length(psum)
    target_size = max(1, n ÷ 2)

    # systematic_resample! hands every survivor an equal share of the total weight, so (for
    # all-positive coefficients) the sum is conserved exactly, not just in expectation
    prop_cache = VectorPauliPropagationCache(deepcopy(psum))
    total_before = sum(activecoeffs(prop_cache))
    resample!(prop_cache, target_size; resample_func=systematic_resample!)
    @test activesize(prop_cache) == target_size
    @test sum(activecoeffs(prop_cache)) ≈ total_before

    # target_size equal to the current size is allowed, only exceeding it is an error
    same_size_cache = VectorPauliPropagationCache(deepcopy(psum))
    resample!(same_size_cache, n; resample_func=systematic_resample!)
    @test activesize(same_size_cache) == n

    over_cache = VectorPauliPropagationCache(deepcopy(psum))
    @test_throws ArgumentError resample!(over_cache, n + 1)

    # out-of-place resample leaves the input psum untouched
    original = deepcopy(psum)
    result = resample(psum, target_size; resample_func=systematic_resample!)
    @test psum == original
    @test length(result) == target_size
end


@testset "resample! variants stay within target_size" begin
    nq = 4
    pstrs = [PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq), rand() + 0.1) for _ in 1:30]
    base_psum = merge!(VectorPauliSum(pstrs))
    n = length(base_psum)
    target_size = max(1, n ÷ 2)

    # multinomial_resample! and systematic_resample! always return exactly target_size terms
    for f in (multinomial_resample!, systematic_resample!)
        cache = VectorPauliPropagationCache(deepcopy(base_psum))
        resample!(cache, target_size; resample_func=f)
        @test activesize(cache) == target_size
    end

    # merged variants deduplicate survivors, so they may return fewer than target_size terms
    for f in (systematic_resample_merged!, semideterministic_systematic_resample_merged!)
        cache = VectorPauliPropagationCache(deepcopy(base_psum))
        resample!(cache, target_size; resample_func=f)
        @test 1 <= activesize(cache) <= target_size
    end

    for deterministic_fraction in (0.0, 0.5, 1.0)
        cache = VectorPauliPropagationCache(deepcopy(base_psum))
        resample!(cache, target_size, deterministic_fraction; resample_func=detfraction_systematic_resample_merged!)
        @test 1 <= activesize(cache) <= target_size
    end
end


@testset "resample! squared=true conserves the squared 2-norm" begin
    nq = 4
    pstrs = [PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq), rand() + 0.1) for _ in 1:30]
    base_psum = merge!(VectorPauliSum(pstrs))
    n = length(base_psum)
    target_size = max(1, n ÷ 2)
    norm_before = sum(abs2, coefficients(base_psum))

    # every drawn slot gets the exact same magnitude sqrt(total_weight/target_size),
    # independent of which term it drew, so the squared 2-norm is conserved exactly
    for f in (multinomial_resample!, systematic_resample!)
        cache = VectorPauliPropagationCache(deepcopy(base_psum))
        resample!(cache, target_size; resample_func=f, squared=true)
        @test sum(abs2, activecoeffs(cache)) ≈ norm_before
    end

    # with calibrate=false the comb step is exactly total_weight/target_size, so the comb
    # teeth exactly tile [0, total_weight) and the squared 2-norm is again conserved exactly
    cache = VectorPauliPropagationCache(deepcopy(base_psum))
    resample!(cache, target_size; resample_func=systematic_resample_merged!, squared=true, calibrate=false)
    @test sum(abs2, activecoeffs(cache)) ≈ norm_before
end


@testset "detfraction_systematic_resample_merged! interpolates the deterministic keep-fraction" begin
    nq = 4
    pstrs = [PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq), rand() + 0.1) for _ in 1:30]
    base_psum = merge!(VectorPauliSum(pstrs))
    n = length(base_psum)
    target_size = max(1, n ÷ 2)
    total_before = sum(activecoeffs(VectorPauliPropagationCache(deepcopy(base_psum))))

    # weight is conserved exactly at every deterministic_fraction, since deterministically-kept
    # terms are copied unchanged and the stochastic remainder is a systematic comb resample
    for deterministic_fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
        cache = VectorPauliPropagationCache(deepcopy(base_psum))
        resample!(cache, target_size, deterministic_fraction; resample_func=detfraction_systematic_resample_merged!)
        @test sum(activecoeffs(cache)) ≈ total_before
    end

    # deterministic_fraction=1.0 is exactly what semideterministic_systematic_resample_merged! does
    Random.seed!(42)
    cache_det = VectorPauliPropagationCache(deepcopy(base_psum))
    resample!(cache_det, target_size, 1.0; resample_func=detfraction_systematic_resample_merged!)

    Random.seed!(42)
    cache_semidet = VectorPauliPropagationCache(deepcopy(base_psum))
    resample!(cache_semidet, target_size; resample_func=semideterministic_systematic_resample_merged!)

    @test VectorPauliSum(cache_det) == VectorPauliSum(cache_semidet)

    # squared=true is not supported since a deterministically-kept term does not preserve the
    # squared 2-norm the way a resampled one does
    cache = VectorPauliPropagationCache(deepcopy(base_psum))
    @test_throws ArgumentError resample!(cache, target_size, 0.5; resample_func=detfraction_systematic_resample_merged!, squared=true)
end


@testset "mcsample!/mcpropagate! respect heisenberg=false" begin
    nq = 4
    nl = 3
    circuit = efficientsu2circuit(nq, nl)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 2)

    exact_psum = propagate(circuit, pstr, thetas; heisenberg=false, min_abs_coeff=0)
    mc_psum = mcpropagate(circuit, VectorPauliSum(pstr), thetas; max_size=10^9, min_abs_coeff=0, heisenberg=false)

    @test length(mc_psum) == length(exact_psum)
    for (term, coeff) in zip(paulis(mc_psum), coefficients(mc_psum))
        @test coeff == getcoeff(exact_psum, term)
    end
end


@testset "mcapplytoall! is currently only implemented for VectorPauliSum" begin
    # dict-backed PauliSum has no mcapplytoall! methods yet; this documents the current scope
    # so that support can be added deliberately rather than silently, if/when it is needed
    nq = 2
    gate = PauliRotation(:X, 1)
    theta = 0.3
    psum = PauliSum(nq, PauliString(nq, :Z, 1))

    @test_throws ErrorException mcsample(gate, psum, theta)
end
