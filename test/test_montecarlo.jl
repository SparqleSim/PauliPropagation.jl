using Test
using PauliPropagation
using PauliPropagation.PropagationBase
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

    # systematic_resample!'s comb step is quantized and randomly offset, so both the survivor
    # count and the weight it carries land close to, but not always exactly at, their targets
    term_tol = 3

    prop_cache = VectorPauliPropagationCache(deepcopy(psum))
    total_before = sum(activecoeffs(prop_cache))
    resample!(prop_cache, target_size; resample_func=systematic_resample!)
    @test abs(activesize(prop_cache) - target_size) <= term_tol
    @test isapprox(sum(activecoeffs(prop_cache)), total_before; atol=term_tol * total_before / target_size)

    # target_size equal to the current size is allowed, only exceeding it is an error
    same_size_cache = VectorPauliPropagationCache(deepcopy(psum))
    resample!(same_size_cache, n; resample_func=systematic_resample!)
    @test abs(activesize(same_size_cache) - n) <= term_tol

    over_cache = VectorPauliPropagationCache(deepcopy(psum))
    @test_throws ArgumentError resample!(over_cache, n + 1)

    # out-of-place resample leaves the input psum untouched
    original = deepcopy(psum)
    result = resample(psum, target_size; resample_func=systematic_resample!)
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

    # multinomial_resample! draws exactly target_size samples, so its count is always exact
    cache = VectorPauliPropagationCache(deepcopy(base_psum))
    resample!(cache, target_size; resample_func=multinomial_resample!)
    @test activesize(cache) == target_size

    for f in (systematic_resample!, semideterministic_systematic_resample!)
        cache = VectorPauliPropagationCache(deepcopy(base_psum))
        resample!(cache, target_size; resample_func=f)
        @test 1 <= activesize(cache) <= target_size + term_tol
    end
end


@testset "resample! variants stay within target_size" begin
    nq = 4
    pstrs = [PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq), rand() + 0.1) for _ in 1:30]
    base_psum = merge!(VectorPauliSum(pstrs))
    n = length(base_psum)
    target_size = max(1, n ÷ 2)
    term_tol = 3

    # multinomial_resample! draws exactly target_size samples, so its count is always exact
    cache = VectorPauliPropagationCache(deepcopy(base_psum))
    resample!(cache, target_size; resample_func=multinomial_resample!)
    @test activesize(cache) == target_size

    # the deduplicating variants' comb step is quantized, so the survivor count can land a
    # few terms above target_size, and may also land well below it if many terms deduplicate
    for f in (systematic_resample!, semideterministic_systematic_resample!)
        cache = VectorPauliPropagationCache(deepcopy(base_psum))
        resample!(cache, target_size; resample_func=f)
        @test 1 <= activesize(cache) <= target_size + term_tol
    end
end

@testset "resample! forwards squared to the resampler" begin
    nq = 4
    pstrs = [PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq), rand() + 0.1) for _ in 1:30]
    base_psum = merge!(VectorPauliSum(pstrs))
    target_size = 5

    # under squared=true, multinomial_resample! assigns every survivor the same weight share
    # of the *squared* 2-norm: |coeff| = sqrt(sum(abs2) / target_size). If squared were dropped
    # on the way to the resampler, the magnitude would be the 1-norm share sum(abs) / target_size.
    squared_share = sqrt(sum(abs2, coefficients(base_psum)) / target_size)

    cache = VectorPauliPropagationCache(deepcopy(base_psum))
    resample!(cache, target_size; resample_func=multinomial_resample!, squared=true)
    @test all(isapprox(abs(c), squared_share) for c in activecoeffs(cache))
end
