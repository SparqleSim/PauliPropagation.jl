# Tests for groups.jl
using Test
using Random: MersenneTwister

# all elements of a group, also for a PermutationSymmetry (which does not list them itself)
_sp_elements(G) = G isa PermutationSymmetry ? PP._closure(nqubits(G), PP.generators(G); maxorder=10^6) : PP.elements(G)


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
