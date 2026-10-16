# Tests for groups.jl, subgroups.jl and symmetrypropagate.jl
using Test
using Random: MersenneTwister, randperm

# all elements of a group, also for a PermutationSymmetry (which does not list them itself)
_sp_elements(G) = G isa PermutationSymmetry ? PP._closure(nqubits(G), PP.generators(G); maxorder=10^6) : PP.elements(G)

_sp_alltoall(nq) = [(i, j) for i in 1:nq for j in i+1:nq]

# the gate of an entry of a plan, the groups of its merges, and the plan of one layer in the order
# the gates are applied with the stabilizer after each gate (the trivial group where nothing is merged)
_sp_gate(entry) = entry isa SymmetryMerge ? entry.gate : entry
_sp_groups(plan) = [entry.group for entry in plan if entry isa SymmetryMerge]
function _sp_plan(G, gates, thetas=nothing; order=:auto)
    plan = symmetrycircuit(G, gates, thetas; heisenberg=false, order)
    return plan, [entry isa SymmetryMerge ? entry.group : PP.TrivialSymmetry(nqubits(G)) for entry in plan]
end
_sp_firstqubit(plan) = [PP._rotationof(_sp_gate(entry)).qinds[1] for entry in plan]
_sp_signature(plan) = [(repr(_sp_gate(entry)), entry isa SymmetryMerge ? repr(entry.group) : "") for entry in plan]

# brute-force stabilizer order of a set of Pauli rotations in a listable group
function _sp_bruteforcestabilizer(G, gates, thetas)
    nq = nqubits(G)
    TT = PP.getinttype(nq)
    applied = Dict(PP._paulistringof(TT, g) => th for (g, th) in zip(gates, thetas))
    fixes(perm) = all(get(applied, PP._permute(p, perm), nothing) == th for (p, th) in applied)
    return count(fixes, _sp_elements(G))
end

# coefficient-wise comparison over the union of keys, as in test_symmetries.jl
function _sp_maxdiff(a::PauliSum, b::PauliSum)
    allkeys = union(Set(paulis(a)), Set(paulis(b)))
    isempty(allkeys) && return 0.0
    return maximum(abs(getcoeff(a, p) - getcoeff(b, p)) for p in allkeys)
end

function _sp_tosum(cache, nq)
    psum = PauliSum(nq)
    for (t, c) in zip(PP.activeterms(cache), PP.activecoeffs(cache))
        add!(psum, t, c)
    end
    return psum
end

function _sp_observable(nq)
    psum = PauliSum(nq)
    for i in 1:nq
        add!(psum, :X, i)
    end
    add!(psum, [:Z, :Y], [2, min(5, nq)], 0.3)
    add!(psum, :Z, 1, 0.45)
    return psum
end


@testset "Site symmetry groups" begin
    # explicit groups
    shift = [2, 3, 4, 5, 6, 1]
    mirror = [6, 5, 4, 3, 2, 1]
    D6 = SiteSymmetry(6, [shift, mirror])
    @test PP.grouporder(D6) == 12
    @test nqubits(D6) == 6
    @test PP.elements(D6)[1] == 1:6
    @test !PP.istrivial(D6)
    @test PP.istrivial(SiteSymmetry(6, Vector{Int}[]))
    @test_throws ArgumentError SiteSymmetry(6, [[1, 2, 3]])
    @test_throws ArgumentError SiteSymmetry(6, [[2, 1, 3, 4, 5, 6], [2, 3, 4, 5, 6, 1]]; maxorder=100)   # S_6 is too large

    @test ReflectionSymmetry(6) == SiteSymmetry(6, [mirror])
    @test PP.grouporder(ReflectionSymmetry(3, 2)) == 4
    @test PP.grouporder(ReflectionSymmetry(3, 2; axes=:x)) == 2
    @test nqubits(ReflectionSymmetry(3, 2)) == 6

    T = TranslationSymmetry(6)
    @test PP.grouporder(T) == 6
    @test length(PP.elements(T)) == 6
    @test SiteSymmetry(6, PP.generators(T)) == SiteSymmetry(6, [shift])
    @test PP.grouporder(TranslationSymmetry(3, 2)) == 6
    @test length(PP.generators(TranslationSymmetry(3, 2))) == 2
    @test PP.istrivial(TranslationSymmetry(1))

    # the symmetric group and Young subgroups
    S4 = PermutationSymmetry(4)
    @test PP.grouporder(S4) == 24
    @test length(_sp_elements(S4)) == 24
    @test length(PP.generators(S4)) == 3
    Y = PermutationSymmetry(4, [[1, 2], [3, 4]])
    @test PP.grouporder(Y) == 4
    @test Set(_sp_elements(Y)) == Set([[1, 2, 3, 4], [2, 1, 3, 4], [1, 2, 4, 3], [2, 1, 4, 3]])
    @test_throws ArgumentError PP.elements(S4)
    @test PermutationSymmetry(4, [[2, 1], [3], Int[]]) == PermutationSymmetry(4, [[1, 2]])
    @test PP.istrivial(PermutationSymmetry(4, [[1], [2]]))
    @test_throws ArgumentError PermutationSymmetry(4, [[1, 2], [2, 3]])
    @test_throws ArgumentError PermutationSymmetry(4, [[1, 5]])
    @test string(PermutationSymmetry(6, [[2, 3], [4, 5, 6]])) == "PermutationSymmetry(6, [2:3, 4:6])"
    @test PP.grouporder(PP.TrivialSymmetry(5)) == 1

    # merging under a group equals the dedicated merges
    nq = 6
    rng = MersenneTwister(7)
    psum = PauliSum(nq)
    for _ in 1:40
        add!(psum, rand(rng, [:I, :X, :Y, :Z], nq), collect(1:nq), rand(rng))
    end
    @test symmetrymerge(PermutationSymmetry(nq), psum) == permutationmerge(psum)
    @test symmetrymerge(TranslationSymmetry(nq), psum) == translationmerge(psum)
    @test symmetrymerge(TranslationSymmetry(3, 2), psum) == translationmerge(psum, 3, 2)
    @test symmetrymerge(ReflectionSymmetry(nq), psum) == reflectionmerge(psum)
    @test symmetrymerge(ReflectionSymmetry(3, 2), psum) == reflectionmerge(psum, 3, 2)
    @test symmetrymerge(PermutationSymmetry(nq, [[2, 3], [4, 5, 6]]), psum) == permutationmerge(psum, ((1, 1), (2, 3), (4, 6)))
    @test symmetrymerge(PP.TrivialSymmetry(nq), psum) == psum
    @test PauliSum(symmetrymerge!(PermutationSymmetry(nq), VectorPauliSum(psum))) == permutationmerge(psum)
    @test PauliSum(symmetrymerge!(PermutationSymmetry(nq), PropagationCache(VectorPauliSum(psum)))) == permutationmerge(psum)
    @test_throws ArgumentError symmetrymerge(PermutationSymmetry(5), psum)

    # a canonical form is constant on orbits and separates them (non-contiguous classes, explicit groups)
    for G in (PermutationSymmetry(nq, [[1, 3, 5], [2, 6]]), D6, TranslationSymmetry(3, 2))
        TT = PP.getinttype(nq)
        canon = PP.canonicalform(G, TT)
        for _ in 1:30
            pstr = symboltoint(rand(rng, [:I, :X, :Y, :Z], nq))
            rep = canon(pstr)
            @test all(canon(PP._permute(pstr, perm)) == rep for perm in _sp_elements(G))
            @test any(PP._permute(pstr, perm) == rep for perm in _sp_elements(G))
        end
    end

    # groups on more than 32 qubits use wide integers
    big = PauliSum(40)
    add!(big, [:X, :Z], [1, 40])
    add!(big, [:Z, :X], [1, 40])
    @test length(symmetrymerge(PermutationSymmetry(40, [[1, 40]]), big)) == 1
    bigt = PauliSum(40)
    add!(bigt, [:X, :Z], [1, 40])
    add!(bigt, [:X, :Z], [2, 1])
    @test length(symmetrymerge(TranslationSymmetry(40), bigt)) == 1

    # groups of different types compare and hash consistently
    @test PermutationSymmetry(2) == SiteSymmetry(2, [[2, 1]])
    @test SiteSymmetry(2, [[2, 1]]) == PermutationSymmetry(2)
    @test PermutationSymmetry(4, [[1, 2]]) != TranslationSymmetry(4)
    @test PermutationSymmetry(4) != PP.TrivialSymmetry(4)
    @test PermutationSymmetry(4, Vector{Int}[]) == PP.TrivialSymmetry(4)
    @test hash(PermutationSymmetry(2)) == hash(SiteSymmetry(2, [[2, 1]]))
    @test length(Set([PermutationSymmetry(4, [[1, 2], [3, 4]]), PermutationSymmetry(4, [[3, 4], [1, 2]])])) == 1
    @test length(Set([SiteSymmetry(3, [[2, 3, 1]]), SiteSymmetry(3, [[3, 1, 2]])])) == 1
    @test length(Set([TranslationSymmetry(4), TranslationSymmetry(4, 1), PP.TrivialSymmetry(4)])) == 2
    @test occursin("AbstractSiteSymmetry", string(@doc symmetrymerge!))
end


@testset "Stabilizers and invariance" begin
    nq = 8
    T = TranslationSymmetry(nq)
    bonds = [PauliRotation([:Z, :Z], [i, mod1(i + 1, nq)]) for i in 1:nq]
    th = fill(0.1, nq)
    @test PP.stabilizer(T, bonds, th) == T
    @test PP.stabilizer(T, bonds[1:7], th[1:7]) != T
    @test PP.stabilizer(T, bonds, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]) != T
    @test PP.stabilizer(T, bonds[1:1], th[1:1]) == PP.TrivialSymmetry(nq)
    @test PP.stabilizer(T, bonds[[1, 5]], th[[1, 5]]) == SiteSymmetry(nq, [[5, 6, 7, 8, 1, 2, 3, 4]])
    @test PP.stabilizer(T, bonds[[1, 3, 5, 7]], th[[1, 3, 5, 7]]) == SiteSymmetry(nq, [[3, 4, 5, 6, 7, 8, 1, 2]])
    # frozen rotations carry their angles
    @test PP.stabilizer(T, [PauliRotation([:Z, :Z], [1, 2], 0.1), PauliRotation([:Z, :Z], [5, 6], 0.1)]) ==
          SiteSymmetry(nq, [[5, 6, 7, 8, 1, 2, 3, 4]])
    @test_throws ArgumentError PP.stabilizer(T, bonds)                 # angles missing
    @test_throws ArgumentError PP.stabilizer(T, bonds, th[1:3])
    @test_throws ArgumentError PP.stabilizer(T, [CliffordGate(:CNOT, [1, 2])])

    # repeated Pauli strings count with their summed angle
    S2 = PermutationSymmetry(2)
    xs = [PauliRotation(:X, 1), PauliRotation(:X, 1), PauliRotation(:X, 2)]
    @test PP.stabilizer(S2, xs, [0.2, 0.3, 0.5]) == S2
    @test PP.stabilizer(S2, xs, [0.2, 0.3, 0.4]) != S2

    # the permutation stabilizer refines the paper's blocks and never exceeds the true stabilizer
    nq = 5
    pairs = _sp_alltoall(nq)
    gates = [PauliRotation([:Y, :Y], [p...]) for p in pairs]
    th = fill(0.17, length(gates))
    S = PermutationSymmetry(nq)
    for k in 1:length(gates)
        H = PP.stabilizer(S, gates[1:k], th[1:k])
        @test PP.grouporder(H) == _sp_bruteforcestabilizer(S, gates[1:k], th[1:k])      # the staircase order gives the full stabilizer
        i, j = pairs[k]
        blocks = ((1, i - 1), (i, i), (i + 1, j), (j + 1, nq))     # the paper's blocks G_ij
        paper = PermutationSymmetry(nq, [collect(lo:hi) for (lo, hi) in blocks])
        @test all(any(issubset(c, class) for class in (PP.istrivial(H) ? Vector{Int}[] : H.classes)) for c in paper.classes)
    end
    @test PP.stabilizer(S, gates, th) == S
    @test PP.stabilizer(S, gates[1:9], th[1:9]) != S
    @test PP.stabilizer(S, gates[[1, 8]], th[[1, 8]]) == PermutationSymmetry(nq, [[1, 2], [3, 4]])    # (1,2),(3,4): swaps within the pairs

    # a single gate is promoted to a layer
    @test PP.stabilizer(TranslationSymmetry(3), PauliRotation(:X, 1), 0.1) == PP.TrivialSymmetry(3)
    @test PP.stabilizer(TranslationSymmetry(3), PauliRotation(:X, 1, 0.1)) == PP.TrivialSymmetry(3)
end


@testset "Plans of a layer" begin
    nq = 8
    T = TranslationSymmetry(nq)
    bonds = [PauliRotation([:Z, :Z], [i, mod1(i + 1, nq)]) for i in 1:nq]
    th = fill(0.1, nq)
    plan, groups = _sp_plan(T, bonds, th)
    @test _sp_firstqubit(plan) == [1, 5, 3, 7, 2, 6, 4, 8]
    @test PP.grouporder.(groups) == [1, 2, 1, 4, 1, 2, 1, 8]
    @test groups[end] == T
    @test countparameters(plan) == 0 && all(entry -> PP._angleof(_sp_gate(entry)) == 0.1, plan)
    # in the Heisenberg picture the layer is reversed before it is planned, so the walk starts at the last
    # bond; the plan is in written order, ready for `propagate`, and reads backwards
    applied = reverse(symmetrycircuit(T, bonds, th))
    @test _sp_firstqubit(applied)[1] == nq
    @test PP.grouporder.([entry isa SymmetryMerge ? entry.group : PP.TrivialSymmetry(nq) for entry in applied]) == [1, 2, 1, 4, 1, 2, 1, 8]
    # written order: nothing merged before the end
    given, givengroups = _sp_plan(T, bonds, th; order=:given)
    @test _sp_firstqubit(given) == 1:nq
    @test all(PP.istrivial, givengroups[1:end-1]) && givengroups[end] == T
    explicit, explicitgroups = _sp_plan(T, bonds, th; order=[1, 5, 3, 7, 2, 6, 4, 8])
    @test PP.grouporder.(explicitgroups) == PP.grouporder.(groups)
    @test_throws ArgumentError symmetrycircuit(T, bonds, th; order=[1, 2, 3])
    @test_throws ArgumentError symmetrycircuit(T, bonds, th; order=:unknown)

    # the symmetric group: lexicographic order whatever the listing, products of symmetric groups
    nq = 6
    S = PermutationSymmetry(nq)
    pairs = _sp_alltoall(nq)
    gates = [PauliRotation([:Y, :Y], [p...]) for p in reverse(pairs)]
    plan, groups = _sp_plan(S, gates, fill(0.2, length(gates)))
    @test [PP._rotationof(_sp_gate(entry)).qinds for entry in plan] == [[p...] for p in pairs]     # :auto sorts lexicographically
    @test groups[end] == S
    @test groups[1] == PermutationSymmetry(nq, [[1, 2], [3, 4, 5, 6]])
    @test_throws ArgumentError symmetrycircuit(S, gates, fill(0.2, length(gates)); order=:orbits)

    # a plan prints its merges
    zz = [PauliRotation([:Z, :Z], [i, j]) for i in 1:4 for j in i+1:4]
    plan_given = symmetrycircuit(PermutationSymmetry(4), zz, fill(0.1, 6); order=:given)
    @test occursin("SymmetryMerge(PermutationSymmetry(4)", string(plan_given))
    @test count(entry -> entry isa SymmetryMerge, plan_given) == 6     # the given order is the staircase order, so every step merges

    # the subgroup-chain order of a cycle, and a long chain planned quickly
    @test PP._subgroupchainorder(8) == [0, 4, 2, 6, 1, 5, 3, 7]
    @test PP._subgroupchainorder(6) == [0, 2, 4, 1, 3, 5]
    @test PP._subgroupchainorder(1) == [0]
    @test sort(PP._subgroupchainorder(12)) == 0:11
    nq = 64
    bonds = [PauliRotation([:Z, :Z], [i, mod1(i + 1, nq)]) for i in 1:nq]
    plan, groups = _sp_plan(TranslationSymmetry(nq), bonds, fill(0.1, nq))
    @test isperm(_sp_firstqubit(plan))
    @test count(!PP.istrivial, groups[1:end-1]) >= 20
    @test PP.grouporder(groups[32]) == 32
    @test groups[end] == TranslationSymmetry(nq)
    # two orbits of gates: the second starts once the first is complete
    fields = [PauliRotation(:X, i) for i in 1:8]
    bonds8 = [PauliRotation([:Z, :Z], [i, mod1(i + 1, 8)]) for i in 1:8]
    plan, groups = _sp_plan(TranslationSymmetry(8), vcat(fields, bonds8), fill(0.1, 16); order=:orbits)
    @test _sp_firstqubit(plan)[1:8] == [1, 5, 3, 7, 2, 6, 4, 8]
    @test PP.grouporder(groups[8]) == 8
end


@testset "symmetrypropagate is exact" begin
    # all-to-all Heisenberg under the symmetric group
    nq = 6
    pairs = _sp_alltoall(nq)
    circuit = heisenbergtrottercircuit(nq, 2; topology=pairs)
    thetas = Float64[]
    for l in 1:2, k in 1:3, _ in pairs
        push!(thetas, 0.05 + 0.02k + 0.01l)
    end
    obs = _sp_observable(nq)
    S = PermutationSymmetry(nq)
    reference = permutationmerge(propagate(circuit, obs, thetas; min_abs_coeff=0.0))

    merged = symmetrypropagate(S, circuit, obs, thetas; min_abs_coeff=0.0)
    @test _sp_maxdiff(merged, reference) < 1e-12
    @test length(merged) == length(reference)
    plan = symmetrycircuit(S, circuit, thetas)
    @test count(entry -> entry isa SymmetryMerge && entry.group == S, plan) == 6     # one merge under S per layer
    @test _sp_groups(symmetrycircuit(S, circuit, thetas; heisenberg=false))[end] == S
    @test _sp_maxdiff(propagate(plan, obs; min_abs_coeff=0.0), reference) < 1e-12   # the plan is a circuit
    # two sizes per merge, before and after it, walked along the plan in the order applied; the peak
    # is a size before a merge, above every size after one
    counts = @countpaulis symmetrypropagate(S, circuit, obs, thetas; min_abs_coeff=0.0)
    @test length(counts) == length(plan) + count(entry -> entry isa SymmetryMerge, plan)
    k = 0
    merges, poststep = Tuple{Int,Int}[], Int[]
    for entry in reverse(plan)
        if entry isa SymmetryMerge
            push!(merges, (counts[k+1], counts[k+2]))
            push!(poststep, counts[k+2])
            k += 2
        else
            push!(poststep, counts[k+1])
            k += 1
        end
    end
    @test k == length(counts)
    @test all(before >= after for (before, after) in merges) && maximum(first, merges) > maximum(poststep)
    @test (@peakpaulis symmetrypropagate(S, circuit, obs, thetas; min_abs_coeff=0.0)) == maximum(counts) == maximum(first, merges)

    # an arbitrary gate order: the swap-based stabilizer is then only a subgroup of the true one, still exact
    @test _sp_maxdiff(symmetrypropagate(S, circuit, obs, thetas; order=randperm(MersenneTwister(3), 15), min_abs_coeff=0.0), reference) < 1e-12
    # with check=false the gates are applied as given
    @test _sp_signature(symmetrycircuit(S, circuit, thetas; order=:given)) == _sp_signature(symmetrycircuit(S, circuit, thetas; check=false))

    # same through the other entry points and backends
    @test _sp_maxdiff(PauliSum(symmetrypropagate(S, circuit, VectorPauliSum(obs), thetas; min_abs_coeff=0.0)), reference) < 1e-12
    cache = PropagationCache(VectorPauliSum(obs))
    @test _sp_maxdiff(PauliSum(symmetrypropagate(S, circuit, cache, thetas; min_abs_coeff=0.0)), reference) < 1e-12   # out of place
    @test length(cache) == length(obs)
    cache = PropagationCache(VectorPauliSum(obs))
    @test symmetrypropagate!(S, circuit, cache, thetas; min_abs_coeff=0.0) === cache
    @test _sp_maxdiff(_sp_tosum(cache, nq), reference) < 1e-12
    copied = deepcopy(obs)
    @test symmetrypropagate!(S, circuit, copied, thetas; min_abs_coeff=0.0) === copied
    @test _sp_maxdiff(copied, reference) < 1e-12
    @test _sp_maxdiff(symmetrypropagate(S, circuit, obs, thetas; min_abs_coeff=0.0, thread=false), reference) < 1e-12
    @test _sp_maxdiff(symmetrypropagate(S, circuit, obs, thetas; min_abs_coeff=0.0, order=:given), reference) < 1e-12
    # merging under the full group only after each layer
    @test _sp_maxdiff(symmetrypropagate(S, circuit, obs, thetas; min_abs_coeff=0.0, subgroups=false), reference) < 1e-12
    plan_full = symmetrycircuit(S, circuit, thetas; subgroups=false)
    @test all(==(S), _sp_groups(plan_full)) && length(_sp_groups(plan_full)) == 6
    @test (@peakpaulis symmetrypropagate(S, circuit, obs, thetas; min_abs_coeff=0.0, subgroups=false)) >= maximum(counts)
    @test _sp_maxdiff(PauliSum(symmetrypropagate(S, circuit, MultiPauliSum(obs, 2), thetas; min_abs_coeff=0.0)), reference) < 1e-12
    @test obs == _sp_observable(nq)   # the out-of-place version does not mutate

    # Schrödinger picture, explicit layers, frozen gates
    reference_s = permutationmerge(propagate(circuit, obs, thetas; heisenberg=false, min_abs_coeff=0.0))
    @test _sp_maxdiff(symmetrypropagate(S, circuit, obs, thetas; heisenberg=false, min_abs_coeff=0.0), reference_s) < 1e-12
    layered = [PauliRotationLayer(circuit[15k-14:15k]) for k in 1:6]
    layered_thetas = [thetas[15k-14:15k] for k in 1:6]
    @test _sp_maxdiff(symmetrypropagate(S, layered, obs, layered_thetas; min_abs_coeff=0.0), reference) < 1e-12
    @test _sp_maxdiff(symmetrypropagate(S, layered, obs, layered_thetas; heisenberg=false, min_abs_coeff=0.0), reference_s) < 1e-12
    # a layer given in the circuit is kept, so three layers are applied although all gates commute
    zfields = [PauliRotation(:Z, i) for i in 1:4]
    zzlayer = PauliRotationLayer([:Z, :Z], [[i, j] for (i, j) in _sp_alltoall(4)])
    given = Gate[zfields..., zzlayer, zfields...]
    given_thetas = Any[fill(0.2, 4)..., fill(0.3, 6), fill(0.4, 4)...]
    flat = Gate[zfields..., zzlayer.gates..., zfields...]
    flat_thetas = vcat(fill(0.2, 4), fill(0.3, 6), fill(0.4, 4))
    obs4 = _sp_observable(4)
    @test _sp_maxdiff(symmetrypropagate(PermutationSymmetry(4), given, obs4, given_thetas; min_abs_coeff=0.0),
        permutationmerge(propagate(flat, obs4, flat_thetas; min_abs_coeff=0.0))) < 1e-12
    @test count(==(PermutationSymmetry(4)), _sp_groups(symmetrycircuit(PermutationSymmetry(4), given, given_thetas))) == 3
    frozen = freeze(circuit, thetas)
    @test _sp_maxdiff(symmetrypropagate(S, frozen, obs; min_abs_coeff=0.0), reference) < 1e-12

    # translation symmetry with the automatic Z_2 / Z_4 order, noise layers in between
    nq = 8
    T = TranslationSymmetry(nq)
    circ = Gate[]
    for _ in 1:3
        append!(circ, tfitrottercircuit(nq, 1; topology=staircasetopology(nq; periodic=true)))
        append!(circ, [DepolarizingNoise(i, 0.02) for i in 1:nq])
    end
    th = Float64[]
    for _ in 1:3
        append!(th, fill(0.3, nq))
        append!(th, fill(0.2, nq))
    end
    obs = PauliSum(nq)
    add!(obs, :Z, 1)
    add!(obs, [:X, :X], [2, 3], 0.4)
    reference = translationmerge(propagate(circ, obs, th; min_abs_coeff=0.0))
    merged = symmetrypropagate(T, circ, obs, th; min_abs_coeff=0.0)
    @test _sp_maxdiff(merged, reference) < 1e-12
    @test Set(PP.grouporder.(_sp_groups(symmetrycircuit(T, circ, th)))) == Set([2, 4, 8])
    @test _sp_maxdiff(symmetrypropagate(T, circ, obs, th; min_abs_coeff=0.0, order=[1, 5, 3, 7, 2, 6, 4, 8]), reference) < 1e-12

    # a 2d grid under translations, and reflections of an open chain
    circ2d = heisenbergtrottercircuit(6, 1; topology=rectangletopology(3, 2; periodic=true))
    th2d = fill(0.15, countparameters(circ2d))
    obs2d = PauliSum(6)
    add!(obs2d, [:X, :Y], [1, 5])
    @test _sp_maxdiff(symmetrypropagate(TranslationSymmetry(3, 2), circ2d, obs2d, th2d; min_abs_coeff=0.0),
        translationmerge(propagate(circ2d, obs2d, th2d; min_abs_coeff=0.0), 3, 2)) < 1e-12

    R = ReflectionSymmetry(6)
    circr = tfitrottercircuit(6, 2; topology=staircasetopology(6))
    thr = fill(0.25, countparameters(circr))
    obsr = PauliSum(6)
    add!(obsr, :Z, 2)
    add!(obsr, [:X, :Y], [1, 4], 0.4)
    @test _sp_maxdiff(symmetrypropagate(R, circr, obsr, thr; min_abs_coeff=0.0),
        reflectionmerge(propagate(circr, obsr, thr; min_abs_coeff=0.0))) < 1e-12

    # with truncation the merged coefficients are judged, so a tiny threshold changes nothing
    @test _sp_maxdiff(symmetrypropagate(R, circr, obsr, thr; min_abs_coeff=1e-14),
        reflectionmerge(propagate(circr, obsr, thr; min_abs_coeff=0.0))) < 1e-12
end


@testset "Layer utilities" begin
    nq = 6
    circuit = heisenbergtrottercircuit(nq, 2; topology=_sp_alltoall(nq))
    thetas = fill(0.1, countparameters(circuit))
    layered, layered_thetas = tolayers(circuit, thetas; reorder=true)
    @test [length(layer.gates) for layer in layered] == fill(15, 6)
    # the stabilizer takes a layer with its parameters, a frozen layer, a vector of gates or a single gate
    S = PermutationSymmetry(nq)
    @test PP.stabilizer(S, layered[1], layered_thetas[1]) == S
    @test PP.stabilizer(S, freeze(layered, layered_thetas)[1]) == S
    @test PP.stabilizer(S, circuit[1:15], thetas[1:15]) == S
    @test PP.stabilizer(S, circuit[1], 0.1) == PermutationSymmetry(nq, [[1, 2], [3, 4, 5, 6]])
    @test_throws ArgumentError PP.stabilizer(S, [CliffordGate(:H, 1)])
    @test_throws ArgumentError PP.stabilizer(S, circuit[1:15], thetas[1:3])
    @test_throws ArgumentError PP.stabilizer(S, freeze(layered, layered_thetas)[1], layered_thetas[1])   # a frozen layer carries its angles
    # a merge gate takes a frozen gate and acts on every qubit
    merge = SymmetryMerge(S, freeze(circuit[1], 0.1))
    @test PP.qinds(merge) == 1:nq && countparameters([merge]) == 0
    @test_throws ArgumentError SymmetryMerge(S, circuit[1])
end


@testset "symmetrypropagate checks its input" begin
    nq = 6
    circuit = heisenbergtrottercircuit(nq, 1; topology=_sp_alltoall(nq))
    obs = _sp_observable(nq)
    S = PermutationSymmetry(nq)
    # unequal angles break the symmetry of a layer
    @test_throws ArgumentError symmetrypropagate(S, circuit, obs, 0.1 .* (1:countparameters(circuit)))
    # a layer given in the circuit that does not commute
    @test_throws ArgumentError symmetrypropagate(S, [GateLayer(circuit; guaranteed_commutes=true)], obs, [fill(0.1, countparameters(circuit))])
    # an explicit order of the wrong length
    @test_throws ArgumentError symmetrypropagate(S, circuit, obs, fill(0.1, countparameters(circuit)); order=[1, 2, 3])
    # a plan is a circuit of its own, not an input for another plan, also inside a layer
    plan = symmetrycircuit(S, circuit, fill(0.1, countparameters(circuit)))
    @test_throws ArgumentError symmetrypropagate(S, plan, obs)
    @test_throws ArgumentError symmetrycircuit(S, [GateLayer(plan[1:3]; guaranteed_commutes=true)]; check=false)
    # a trivial group plans no merges
    @test !any(entry -> entry isa SymmetryMerge, symmetrycircuit(PermutationSymmetry(nq, Vector{Int}[]), circuit, fill(0.1, countparameters(circuit))))
    # the wrong number of qubits
    @test_throws ArgumentError symmetrypropagate(PermutationSymmetry(5), circuit, obs, fill(0.1, countparameters(circuit)))
    # noise that is not symmetric
    noise = [DepolarizingNoise(i, i == 3 ? 0.05 : 0.02) for i in 1:nq]
    @test_throws ArgumentError symmetrypropagate(S, noise, obs)
    @test symmetrypropagate(S, noise, obs; check=false) isa PauliSum
    # check=false skips the tests and applies the gates as given
    @test symmetrypropagate(S, circuit, obs, 0.1 .* (1:countparameters(circuit)); check=false) isa PauliSum
    # other gates are checked as a set of whole gates that commute
    cliffords = [CliffordGate(:H, i) for i in 1:nq]
    @test _sp_maxdiff(symmetrypropagate(S, cliffords, obs), permutationmerge(propagate(cliffords, obs))) < 1e-12
    @test_throws ArgumentError symmetrypropagate(S, cliffords[1:1], obs)
    @test symmetrypropagate(S, cliffords[1:1], obs; check=false) isa PauliSum
    mixed = [CliffordGate(:H, 1), CliffordGate(:X, 2)]
    @test_throws ArgumentError symmetrypropagate(PermutationSymmetry(2), mixed, PauliSum(2))
    cz = [CliffordGate(:CZ, [1, 2]), CliffordGate(:CZ, [4, 3])]
    @test symmetrypropagate(PermutationSymmetry(4, [[1, 2], [3, 4]]), cz, PauliSum(4)) isa PauliSum
    @test symmetrypropagate(PermutationSymmetry(4, [[1, 2], [3, 4]]), cz, PauliSum(4); heisenberg=false) isa PauliSum   # transposed CZ
    @test symmetrypropagate(ReflectionSymmetry(4), cz, PauliSum(4)) isa PauliSum
    # a ring of CNOTs does not commute, so its gates are checked one by one and are not invariant
    cnots = [CliffordGate(:CNOT, [i, mod1(i + 1, 3)]) for i in 1:3]
    @test_throws ArgumentError symmetrypropagate(TranslationSymmetry(3), cnots, PauliSum(3))
    @test symmetrypropagate(TranslationSymmetry(3), cnots, PauliSum(3); check=false) isa PauliSum
    # Clifford gates that commute on shared qubits form a layer that is checked as a set, in both pictures
    czring = [CliffordGate(:CZ, [i, mod1(i + 1, 3)]) for i in 1:3]
    czobs = PauliSum(3)
    add!(czobs, [:X, :Z], [1, 2])
    add!(czobs, :Y, 3, 0.4)
    for heisenberg in (true, false)
        @test _sp_maxdiff(symmetrypropagate(TranslationSymmetry(3), czring, czobs; heisenberg), translationmerge(propagate(czring, czobs; heisenberg))) < 1e-12
    end
    # with check=false, consecutive other gates are one block merged only afterwards: a SWAP as three CNOTs
    swapgates = [CliffordGate(:CNOT, [1, 2]), CliffordGate(:CNOT, [2, 1]), CliffordGate(:CNOT, [1, 2])]
    S2 = SiteSymmetry(2, [[2, 1]])
    swapobs = PauliSum(2)
    add!(swapobs, :X, 1)
    add!(swapobs, :Z, 2, 0.5)
    add!(swapobs, [:Y, :Z], [1, 2], 0.3)
    @test_throws ArgumentError symmetrypropagate(S2, swapgates, swapobs)
    @test _sp_maxdiff(symmetrypropagate(S2, swapgates, swapobs; check=false), symmetrymerge(S2, propagate(swapgates, swapobs))) < 1e-12
    # a given layer whose gates do not commute, whatever it asserts, is refused
    noncommuting = Gate[CliffordGate(:H, 1), PauliRotation(:X, 1), PauliRotation(:X, 2), CliffordGate(:H, 2), PauliRotation(:X, 3), CliffordGate(:H, 3)]
    @test_throws ArgumentError symmetrypropagate(TranslationSymmetry(3), [GateLayer(noncommuting; guaranteed_commutes=true)], PauliSum(3), [[0.3, 0.3, 0.3]])
    # a given layer mixing rotations with other gates is checked as a set of whole gates; this one is not invariant
    mixedlayer = [PauliRotation(:X, 1), PauliRotation(:Z, 2), CliffordGate(:H, 1), CliffordGate(:H, 2)]
    @test_throws ArgumentError symmetrypropagate(PermutationSymmetry(2), [GateLayer(mixedlayer; guaranteed_commutes=true)], PauliSum(2), [[0.7, 0.7]])
end


@testset "symmetrypropagate: more groups, entry points and bookkeeping" begin
    # the dihedral group of a ring, given by generators
    nq = 6
    shift = [2, 3, 4, 5, 6, 1]
    mirror = [6, 5, 4, 3, 2, 1]
    D6 = SiteSymmetry(nq, [shift, mirror])
    circ = tfitrottercircuit(nq, 2; topology=staircasetopology(nq; periodic=true))
    th = fill(0.3, countparameters(circ))
    obs = PauliSum(nq)
    add!(obs, :Z, 1)
    add!(obs, [:X, :Y], [2, 4], 0.4)
    reference = symmetrymerge(D6, propagate(circ, obs, th; min_abs_coeff=0.0))
    @test _sp_maxdiff(symmetrypropagate(D6, circ, obs, th; min_abs_coeff=0.0), reference) < 1e-12
    @test length(reference) < length(translationmerge(propagate(circ, obs, th; min_abs_coeff=0.0)))

    # the trivial group reproduces propagate exactly
    @test _sp_maxdiff(symmetrypropagate(PP.TrivialSymmetry(nq), circ, obs, th; min_abs_coeff=0.0), propagate(circ, obs, th; min_abs_coeff=0.0)) < 1e-12

    # a Pauli string as the observable
    pstr = PauliString(nq, :Z, 1)
    @test _sp_maxdiff(symmetrypropagate(TranslationSymmetry(nq), circ, pstr, th; min_abs_coeff=0.0),
        translationmerge(propagate(circ, pstr, th; min_abs_coeff=0.0))) < 1e-12

    # T gates count as rotations with their fixed angle
    tobs = PauliSum(2)
    add!(tobs, [:X, :Y], [1, 2])
    @test _sp_maxdiff(symmetrypropagate(PermutationSymmetry(2), [TGate(1), TGate(2)], tobs), permutationmerge(propagate([TGate(1), TGate(2)], tobs))) < 1e-12

    # one count per gate plus one before every merge, the last one after the merge of the layer
    noise = [DepolarizingNoise(i, 0.1) for i in 1:nq]
    counts = @countpaulis merged = symmetrypropagate(TranslationSymmetry(nq), noise, obs)
    @test length(counts) == nq + 1
    @test counts[end] == length(merged)
    allgates = heisenbergtrottercircuit(nq, 1; topology=_sp_alltoall(nq))
    counts = @countpaulis symmetrypropagate(PermutationSymmetry(nq), allgates, obs, fill(0.1, 45); min_abs_coeff=0.0)
    @test length(counts) == 45 + count(entry -> entry isa SymmetryMerge, symmetrycircuit(PermutationSymmetry(nq), allgates, fill(0.1, 45)))
    # the plan goes through the Performance module, which gathers the runs between merges into layers
    plan = symmetrycircuit(TranslationSymmetry(nq), circ, th)
    @test _sp_maxdiff(PauliSum(PP.Performance.propagate(plan, VectorPauliSum(obs); min_abs_coeff=0.0)), translationmerge(propagate(circ, obs, th; min_abs_coeff=0.0))) < 1e-12
    # with subgroups=false the runs between merges are whole layers, which a Dict sum gathers into GateLayers
    gates8 = heisenbergtrottercircuit(8, 1; topology=_sp_alltoall(8))
    obs8 = _sp_observable(8)
    whole = symmetrycircuit(PermutationSymmetry(8), gates8, fill(0.1, 84); subgroups=false)
    gathered = PP.Performance._tolayers(whole, nothing, obs8)[1]
    @test count(entry -> entry isa GateLayer, gathered) == 3 && count(entry -> entry isa SymmetryMerge, gathered) == 3
    @test _sp_maxdiff(PP.Performance.propagate(whole, obs8; min_abs_coeff=0.0), permutationmerge(propagate(gates8, obs8, fill(0.1, 84); min_abs_coeff=0.0))) < 1e-12
    # frequency truncation wraps the coefficients as propagate does and truncates; propagate! requires them wrapped
    @test length(symmetrypropagate(PermutationSymmetry(nq), allgates, obs, fill(0.1, 45); max_freq=10)) < length(symmetrypropagate(PermutationSymmetry(nq), allgates, obs, fill(0.1, 45)))
    @test_throws ArgumentError symmetrypropagate!(PermutationSymmetry(nq), allgates, PauliString(nq, :Z, 1), fill(0.1, 45); max_freq=10)

    # a rotation in a mixed layer is recognised however it lists its qubits
    swapped = [PauliRotation([:X, :Z], [1, 2]), PauliRotation([:Z, :X], [4, 3]), CliffordGate(:H, 5), CliffordGate(:H, 6)]
    @test symmetrypropagate(SiteSymmetry(6, [[3, 4, 1, 2, 6, 5]]), [GateLayer(swapped)], PauliSum(6), [[0.7, 0.7]]) isa PauliSum
    # docstrings are bound to the exported names
    @test !occursin("No documentation", string(@doc symmetrypropagate))
    @test !occursin("No documentation", string(@doc symmetrycircuit))
end
