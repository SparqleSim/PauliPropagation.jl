using Test
using PauliPropagation
using Random

include("../src/IndexedPropagation.jl")
using .IndexedPropagation
const IP = IndexedPropagation

# Same sum, up to ordering and terms one side dropped as zero.
function samesum(psum1, psum2; atol=1e-12)
    d1 = Dict(pstr => coeff for (pstr, coeff) in merge(psum1))
    d2 = Dict(pstr => coeff for (pstr, coeff) in merge(psum2))
    for pstr in union(keys(d1), keys(d2))
        isapprox(get(d1, pstr, 0.0), get(d2, pstr, 0.0); atol) || return false
    end
    return true
end

@testset "symplectic view" begin
    Random.seed!(42)
    for nq in (3, 5, 40)
        TT = getinttype(nq)
        top = TT(1) << (2nq) - TT(1)
        for _ in 1:500
            pstr, gen = rand(TT) & top, rand(TT) & top

            # the columns of a generator have odd parity over exactly the anticommuting strings
            parity = 0
            for (k, word) in enumerate(IP.keywords(pstr))
                sym = IP.symplecticword(word)
                for c in IP.columnsof(gen)
                    64 * (k - 1) < c <= 64 * k && (parity ⊻= (sym >> (c - 1 - 64 * (k - 1))) & 1)
                end
            end
            @test iszero(parity) == commutes(gen, pstr)

            # a rotation sends a pair of anticommuting strings into each other with opposite signs
            if !commutes(gen, pstr)
                _, sign = PauliPropagation.paulirotationproduct(gen, pstr)
                _, backsign = PauliPropagation.paulirotationproduct(gen, pstr ⊻ gen)
                @test backsign == -sign
            end
        end
    end
end

@testset "transposed index" begin
    Random.seed!(42)
    nq = 12
    TT = getinttype(nq)
    pstrs = [rand(TT) & (TT(1) << (2nq) - TT(1)) for _ in 1:500]

    index = IP.TransposedIndex(nq)
    IP.appendterms!(index, pstrs, 300)
    IP.appendterms!(index, pstrs, 500)   # incremental sync of the rest

    for _ in 1:50
        gen = rand(TT) & (TT(1) << (2nq) - TT(1))
        marked = Int[]
        IP.foreachbranching(i -> push!(marked, i), index, IP.columnsof(gen), 500)
        @test marked == findall(p -> !commutes(gen, p), pstrs)
    end
end

@testset "term table" begin
    Random.seed!(42)
    nq = 12
    TT = getinttype(nq)
    pstrs = unique(rand(TT) for _ in 1:400)

    table = IP.TermTable(length(pstrs))
    IP.refill!(table, pstrs, length(pstrs))
    for (i, pstr) in enumerate(pstrs)
        @test IP.findslot(table, pstrs, pstr, IP.termhash(pstr))[2] == i
    end
    @test IP.findslot(table, pstrs, ~pstrs[1], IP.termhash(~pstrs[1]))[2] == 0
end

@testset "propagation matches the library" begin
    Random.seed!(42)

    circuits = Dict(
        "tfi bricklayer" => (6, tfitrottercircuit(6, 3)),
        "heisenberg" => (5, heisenbergtrottercircuit(5, 2)),
        "tilted tfi" => (7, tiltedtfitrottercircuit(7, 2)),
    )

    for (name, (nq, circuit)) in circuits, min_abs_coeff in (0.0, 1e-3), max_weight in (Inf, 3)
        thetas = randn(countparameters(circuit))
        obs = PauliSum(nq)
        add!(obs, symboltoint(nq, [:Z], [1]), 1.0)
        add!(obs, symboltoint(nq, [:X, :Y], [2, 3]), 0.5)

        want = propagate(circuit, obs, thetas; min_abs_coeff, max_weight)
        got = IP.propagate(circuit, obs, thetas; min_abs_coeff, max_weight)

        @test samesum(want, got)
        @test length(got) == length(want)
    end
end

@testset "noise and truncated terms" begin
    Random.seed!(42)
    nq = 6
    circuit = Gate[]
    for gate in tfitrottercircuit(nq, 3)
        push!(circuit, gate)
    end
    for q in 1:nq
        push!(circuit, DepolarizingNoise(q, 0.05))
    end
    thetas = randn(countparameters(circuit))
    obs = PauliString(nq, :Z, 3)

    for min_abs_coeff in (0.0, 1e-3)
        want = propagate(circuit, obs, thetas; min_abs_coeff)
        got = IP.propagate(circuit, obs, thetas; min_abs_coeff)
        @test samesum(want, got)
    end
end

@testset "imaginary rotations" begin
    Random.seed!(42)
    nq = 5
    circuit = Gate[]
    for l in 1:3
        for q in 1:nq-1
            push!(circuit, ImaginaryPauliRotation([:Z, :Z], [q, q + 1]))
        end
        for q in 1:nq
            push!(circuit, ImaginaryPauliRotation(:X, q))
        end
    end
    taus = fill(0.05, countparameters(circuit))
    obs = PauliSum(nq)
    add!(obs, symboltoint(nq, [:I], [1]), 1.0)
    add!(obs, symboltoint(nq, [:Z], [2]), 0.3)

    for min_abs_coeff in (0.0, 1e-4)
        want = propagate(circuit, obs, taus; heisenberg=false, min_abs_coeff)
        got = IP.propagate(circuit, obs, taus; heisenberg=false, min_abs_coeff)
        @test samesum(want, got; atol=1e-10)
    end
end

@testset "zoned propagation" begin
    Random.seed!(42)
    nq = 7
    topology = bricklayertopology(nq)
    circuit = tfitrottercircuit(nq, 3; topology)

    # a Jordan-Wigner string makes a generator that spans more qubits than a column tuple unrolls
    push!(circuit, PauliRotation([:X, :Z, :Z, :Z, :Y], 1:5))
    push!(circuit, PauliRotation([:Y, :Z, :Z, :Z, :Z, :X], 2:7))
    for q in 1:nq
        push!(circuit, DepolarizingNoise(q, 0.03))
    end
    thetas = randn(countparameters(circuit))
    obs = PauliString(nq, :Z, 4)

    for min_abs_coeff in (0.0, 1e-3), max_weight in (Inf, 4)
        want = propagate(circuit, obs, thetas; min_abs_coeff, max_weight)
        for n_zones in (1, 2, 4, 8), thread in (false, true)
            cache = MultiIndexedPauliPropagationCache(PauliSum(obs), n_zones)
            propagate!(circuit, cache, thetas; min_abs_coeff, max_weight, thread)
            @test nliveterms(cache) == length(want)
            @test samesum(want, VectorPauliSum(cache))
            for (pstr, coeff) in want
                @test getcoeff(cache, pstr) ≈ coeff
            end
        end
    end
end

@testset "zoned imaginary rotations" begin
    nq = 5
    circuit = Gate[ImaginaryPauliRotation(:X, q) for q in 1:nq]
    append!(circuit, [ImaginaryPauliRotation([:Z, :Z], [q, q + 1]) for q in 1:nq-1])
    taus = fill(0.1, countparameters(circuit))
    identity_pstr = PauliString(nq, :I, 1)

    want = propagate(circuit, PauliSum(identity_pstr), taus; heisenberg=false, min_abs_coeff=1e-10)
    for n_zones in (1, 4)
        got = IP.propagate(circuit, PauliSum(identity_pstr), taus; n_zones, heisenberg=false, min_abs_coeff=1e-10)
        @test length(got) == length(want)
        @test samesum(want, got; atol=1e-10)
    end
end

@testset "cache round trip" begin
    nq = 5
    obs = PauliString(nq, :Z, 2)
    circuit = tfitrottercircuit(nq, 2)
    thetas = fill(0.3, countparameters(circuit))

    cache = IndexedPauliPropagationCache(obs)
    propagate!(circuit, cache, thetas; min_abs_coeff=1e-4)

    want = propagate(circuit, obs, thetas; min_abs_coeff=1e-4)
    for (pstr, coeff) in want
        @test getcoeff(cache, pstr) ≈ coeff
    end
    @test nliveterms(cache) == length(want)
    @test samesum(want, PauliSum(cache))
end
