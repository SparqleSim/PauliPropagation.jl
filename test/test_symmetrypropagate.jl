# Tests for groups.jl and subgroups.jl
using Test
using Random: MersenneTwister

# all elements of a group, also for a PermutationSymmetry (which does not list them itself)
_sp_elements(G) = G isa PermutationSymmetry ? PP._closure(nqubits(G), PP.generators(G); maxorder=10^6) : PP.elements(G)

_sp_alltoall(nq) = [(i, j) for i in 1:nq for j in i+1:nq]

# brute-force stabilizer order of a set of Pauli rotations in a listable group
function _sp_bruteforcestabilizer(G, gates, thetas)
    nq = nqubits(G)
    TT = PP.getinttype(nq)
    applied = Dict(PP._paulistringof(TT, g) => th for (g, th) in zip(gates, thetas))
    fixes(perm) = all(get(applied, PP._permute(p, perm), nothing) == th for (p, th) in applied)
    return count(fixes, _sp_elements(G))
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


@testset "Subgroup schedules" begin
    nq = 8
    T = TranslationSymmetry(nq)
    bonds = [PauliRotation([:Z, :Z], [i, mod1(i + 1, nq)]) for i in 1:nq]
    th = fill(0.1, nq)

    sched = subgroupschedule(T, bonds, th)
    @test sched.order == [1, 5, 3, 7, 2, 6, 4, 8]
    @test PP.grouporder.(sched.groups) == [1, 2, 1, 4, 1, 2, 1, 8]
    @test sched.groups[end] == T
    @test length(sched) == nq
    @test occursin("SiteSymmetry(8, order 4)", string(sched))

    given = subgroupschedule(T, bonds, th; order=:given)
    @test given.order == 1:nq
    @test all(PP.istrivial, given.groups[1:end-1])
    explicit = subgroupschedule(T, bonds, th; order=[1, 5, 3, 7, 2, 6, 4, 8])
    @test PP.grouporder.(explicit.groups) == PP.grouporder.(sched.groups)
    @test_throws ArgumentError subgroupschedule(T, bonds, th; order=[1, 2, 3])
    @test_throws ArgumentError subgroupschedule(T, bonds, th; order=:unknown)

    nq = 6
    S = PermutationSymmetry(nq)
    pairs = _sp_alltoall(nq)
    gates = [PauliRotation([:Y, :Y], [p...]) for p in reverse(pairs)]
    sched = subgroupschedule(S, gates, fill(0.2, length(gates)))
    @test sched.order == reverse(1:length(gates))                        # :auto sorts lexicographically
    @test sched.groups[end] == S
    @test sched.groups[1] == PermutationSymmetry(nq, [[1, 2], [3, 4, 5, 6]])
    @test_throws ArgumentError subgroupschedule(S, gates, fill(0.2, length(gates)); order=:orbits)

    # printing a schedule
    zz = [PauliRotation([:Z, :Z], [i, j]) for i in 1:4 for j in i+1:4]
    sched_given = subgroupschedule(PermutationSymmetry(4), zz, fill(0.1, 6); order=:given)
    @test occursin("TrivialSymmetry(4)", string(sched_given)) || !any(PP.istrivial, sched_given.groups)
    @test string(sched_given) isa String

    # the subgroup-chain order of a cycle, and a long chain scheduled quickly
    @test PP._subgroupchainorder(8) == [0, 4, 2, 6, 1, 5, 3, 7]
    @test PP._subgroupchainorder(6) == [0, 2, 4, 1, 3, 5]
    @test PP._subgroupchainorder(1) == [0]
    @test sort(PP._subgroupchainorder(12)) == 0:11
    nq = 64
    bonds = [PauliRotation([:Z, :Z], [i, mod1(i + 1, nq)]) for i in 1:nq]
    sched = subgroupschedule(TranslationSymmetry(nq), bonds, fill(0.1, nq))
    @test isperm(sched.order)
    @test count(!PP.istrivial, sched.groups[1:end-1]) >= 20
    @test PP.grouporder(sched.groups[32]) == 32
    @test sched.groups[end] == TranslationSymmetry(nq)
    # two orbits of gates: the second starts once the first is complete
    fields = [PauliRotation(:X, i) for i in 1:8]
    bonds8 = [PauliRotation([:Z, :Z], [i, mod1(i + 1, 8)]) for i in 1:8]
    sched = subgroupschedule(TranslationSymmetry(8), vcat(fields, bonds8), fill(0.1, 16); order=:orbits)
    @test sched.order[1:8] == [1, 5, 3, 7, 2, 6, 4, 8]
    @test PP.grouporder(sched.groups[8]) == 8
end


@testset "Layer utilities" begin
    nq = 6
    circuit = heisenbergtrottercircuit(nq, 2; topology=_sp_alltoall(nq))
    thetas = fill(0.1, countparameters(circuit))
    TT = PP.getinttype(nq)
    layer = PP.RotationLayer(TT, circuit[1:15], thetas[1:15])
    @test length(layer) == 15 && PP.iscommuting(layer)
    @test layer == PP.RotationLayer(TT, freeze(circuit[1:15], thetas[1:15]))
    @test !PP.iscommuting(PP.RotationLayer(TT, circuit[1:16], thetas[1:16]))
    @test_throws ArgumentError PP.RotationLayer(TT, [CliffordGate(:H, 1)])
    @test_throws ArgumentError PP.RotationLayer(TT, circuit[1:15], thetas[1:3])
    @test length.(PP._commutinglayers(circuit)) == fill(15, 6)
end
