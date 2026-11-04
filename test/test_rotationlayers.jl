using Random
using Test

# the rotations of the layers as written, those of every layer in the order of its gates
function expandlayers(layers, thetas)
    rotations = Gate[]
    angles = Float64[]
    for (layer, theta) in zip(layers, thetas)
        for (index, rotation) in enumerate(layer.gates)
            push!(rotations, rotation)
            angle = if theta isa Number
                theta
            else
                theta[index]
            end
            push!(angles, angle)
        end
    end
    return rotations, angles
end

function randompaulisum(rng, nq, nterms, max_weight)
    psum = PauliSum(nq)
    for _ in 1:nterms
        weight = rand(rng, 1:max_weight)
        add!(psum, rand(rng, [:X, :Y, :Z], weight), randperm(rng, nq)[1:weight], randn(rng))
    end
    return psum
end

# one angle per rotation
randomangles(rng, layers) = [randn(rng, countparameters(layer)) for layer in layers]

multivectorsum(psum) = MultiPauliSum(VectorPauliSum(psum), 4)
multidictsum(psum) = MultiPauliSum(psum, 4)

# rotations on one to four qubits that commute with each other, drawn at random and kept where they commute with those
# kept before
function randomcommutingrotations(rng, nq, n_rotations)
    TT = getinttype(nq)
    rotations = PauliRotation[]
    generators = TT[]
    for _ in 1:10_000
        if length(rotations) == n_rotations
            break
        end
        weight = rand(rng, 1:4)
        qinds = randperm(rng, nq)[1:weight]
        symbols = rand(rng, [:X, :Y, :Z], weight)
        generator = symboltoint(TT, symbols, qinds)
        if all(other -> commutes(other, generator), generators)
            push!(rotations, PauliRotation(symbols, qinds))
            push!(generators, generator)
        end
    end
    return rotations
end

# every kind of sum in the Heisenberg picture, and the two that the others are made of in the Schrödinger picture as well
const LAYER_TEST_SUMS = ((PauliSum, true), (VectorPauliSum, true), (multivectorsum, true), (multidictsum, true),
    (PauliSum, false), (VectorPauliSum, false))

# A vector sum in either picture, which share their compiled code. A layer is compiled afresh for every kind of sum, term
# type and largest number of rotations that flip one bit, so most layers are checked on this sum alone, and only some on
# eight qubits with at most two rotations flipping a bit on every kind of sum.
const VECTOR_TEST_SUMS = ((VectorPauliSum, true), (VectorPauliSum, false))

function layersmatchrotations(layers, thetas, psum; sums=LAYER_TEST_SUMS, kwargs...)
    rotations, angles = expandlayers(layers, thetas)
    matches = true
    for (T, heisenberg) in sums
        layered = propagate(layers, T(psum), thetas; heisenberg, kwargs...)
        reference = propagate(rotations, T(psum), angles; heisenberg, kwargs...)
        # the same terms, as two sums also compare equal if one of them holds more terms with a coefficient of zero
        matches &= !isempty(layered) && length(layered) == length(reference) && PauliSum(layered) == PauliSum(reference)
    end
    return matches
end

# a few light Pauli strings and a heavy one across the qubits, which anticommutes with many rotations
function lightandheavypaulisum(rng, nq)
    psum = randompaulisum(rng, nq, 12, 3)
    add!(psum, repeat([:Y, :Z], 4), round.(Int, range(1, nq; length=8)), 1.0)
    return psum
end

function testlayersonallsums()
    nq = 8
    rng = MersenneTwister(nq)
    psum = lightandheavypaulisum(rng, nq)
    circuit = repeat([PauliRotationLayer(:X, 1:nq), PauliRotationLayer([:Z, :Z], staircasetopology(nq))], 2)

    # without any truncation, which 4^nq bounds, and with either truncation
    matches = true
    for (max_weight, min_abs_coeff) in ((Inf, 0.0), (4.0, 0.0), (Inf, 1e-3))
        matches &= layersmatchrotations(circuit, randomangles(rng, circuit), psum; max_weight, min_abs_coeff)
    end
    @test matches
end

function testlayershapes()
    nq = 8
    rng = MersenneTwister(nq + 1)
    psum = lightandheavypaulisum(rng, nq)
    chain = staircasetopology(nq)
    disjoint_bonds = [(i, i + 1) for i in 1:2:nq-1]
    # bonds between any two qubits, which share qubits and come more than once
    any_bonds = [(1, 5), (5, 1), (2, 7), (7, 3), (3, 2), (8, 4), (6, 8), (1, 5)]
    # the squares of a ladder of two rows, with every qubit acted on with X and with Y
    squares = [(i, i + 1, i + nq ÷ 2, i + nq ÷ 2 + 1) for i in 1:nq÷2-1]

    # rotations that share qubits and rotations that do not, in the order of the qubits and in any other, on some of the
    # qubits, with bonds that close a cycle or meet at one qubit, and on three or four qubits
    layer_sets = (
        [PauliRotationLayer(:Y, nq:-1:1), PauliRotationLayer([:X, :X], staircasetopology(nq; periodic=true))],
        [PauliRotationLayer(:Z, randperm(rng, nq)[1:nq÷2+1]), PauliRotationLayer([:Y, :Y], shuffle(rng, chain))],
        [PauliRotationLayer([:X, :Y], disjoint_bonds), PauliRotationLayer([:Z, :Z], [(i + 3, i) for i in 1:nq-3])],
        # every bond in both directions with different Paulis, so that every qubit is acted on with X and with Z, and XX,
        # YY and ZZ on the same bonds
        [PauliRotationLayer([:X, :Z], vcat(disjoint_bonds, reverse.(disjoint_bonds))),
            GateLayer([PauliRotation([s, s], [i, i + 1]) for s in (:X, :Y, :Z) for i in 1:2:nq-1]; guaranteed_commutes=true)],
        [PauliRotationLayer([:Z, :Z], [(1, i) for i in 2:4]), PauliRotationLayer([:Z, :Z], rectangletopology(2, nq ÷ 2)),
            PauliRotationLayer([:X, :X], any_bonds)],
        [PauliRotationLayer([:Z, :Z, :Z], [(i, i + 1, i + 2) for i in 1:nq-2]),
            GateLayer(vcat(PauliRotationLayer([:X, :X, :X, :X], squares).gates, PauliRotationLayer([:Y, :Y, :Y, :Y], squares).gates); guaranteed_commutes=true)],
        [PauliRotationLayer(randomcommutingrotations(rng, nq, nq ÷ 2 + 2)), PauliRotationLayer(randomcommutingrotations(rng, nq, nq ÷ 2 + 2))],
    )

    matches = true
    for layers in layer_sets, (max_weight, min_abs_coeff) in ((Inf, 0.0), (4.0, 0.0), (Inf, 1e-3))
        circuit = repeat(layers, 2)
        matches &= layersmatchrotations(circuit, randomangles(rng, circuit), psum; sums=VECTOR_TEST_SUMS, max_weight, min_abs_coeff)
    end
    @test matches
end

function testwidelayers()
    # on two limbs and on four, with more than 64 rotations in a layer on four, and two rotations flipping a bit in either
    # layer, so that each width compiles one pass
    matches = true
    for nq in (40, 100)
        rng = MersenneTwister(nq)
        psum = lightandheavypaulisum(rng, nq)
        layers = [PauliRotationLayer([:Z, :Z], staircasetopology(nq)), PauliRotationLayer([:X, :X], reverse(staircasetopology(nq)))]
        for (max_weight, min_abs_coeff) in ((Inf, 2e-2), (4.0, 0.0), (3.0, 1e-4))
            matches &= layersmatchrotations(layers, randomangles(rng, layers), psum; sums=VECTOR_TEST_SUMS, max_weight, min_abs_coeff)
        end
    end
    @test matches
end

function testlargeclassoftwostrings()
    # two Pauli strings of one class of eight rotations, which must be collected in one place to be mixed
    nq = 8
    psum = PauliSum(nq)
    yz = repeat([:Y, :Z], 4)
    add!(psum, yz, 1:8, 1.0)
    add!(psum, [yz[1:2]; :Z; yz[4:8]], 1:8, 0.5)
    layers = [PauliRotationLayer(:X, 1:nq)]
    @test layersmatchrotations(layers, [fill(0.3, nq)], psum; min_abs_coeff=1e-3)
end

function testlayers()
    testlayersonallsums()
    testlayershapes()
    testwidelayers()
    testlargeclassoftwostrings()
end

@testset "PauliRotationLayer propagates like its rotations" begin
    testlayers()
end

@testset "PauliRotationLayer acting on a qubit with two Paulis" begin
    # XI commutes with XX but not with ZZ, which act on its first qubit with X and with Z
    nq = 8
    xx_zz = GateLayer([PauliRotation([:X, :X], [1, 2]), PauliRotation([:Z, :Z], [1, 2])]; guaranteed_commutes=true)
    @test layersmatchrotations([xx_zz], [[0.3, 0.7]], PauliSum(PauliString(nq, :X, 1)); min_abs_coeff=0.0)

    # the identity, which commutes with XZ on (1, 2) and on (2, 1), and XX, which anticommutes with both, in either order
    # of the two in the sum
    xz_zx = PauliRotationLayer([:X, :Z], [(1, 2), (2, 1)])
    identity, xx = PauliString(nq, :I, 1), PauliString(nq, [:X, :X], [1, 2])
    matches = true
    for psum in (PauliSum([identity, xx]), PauliSum([xx, identity]))
        matches &= layersmatchrotations([xz_zx], [[0.3, 0.7]], psum; min_abs_coeff=0.0)
    end
    @test matches
end

# The echelon basis of a plan spans the generators of its rotations, and its reduction adds up the other bits of the
# basis vectors of any pivot bits. The partners of a Pauli string under the rotations it anticommutes with anticommute
# with the same rotations and have its class representative, which is a representative of its own.
function testplans()
    nq = 8
    rng = MersenneTwister(5)
    cache = PropagationCache(PauliSum(nq))
    TT = paulitype(cache)
    any_bonds = [Tuple(randperm(rng, nq)[1:2]) for _ in 1:nq]
    heisenberg = GateLayer([PauliRotation([s, s], [i, i + 1]) for s in (:X, :Y, :Z) for i in 1:2:nq-1]; guaranteed_commutes=true)
    layers = [PauliRotationLayer(:Z, 1:nq), PauliRotationLayer([:Z, :Z], staircasetopology(nq; periodic=true)),
        PauliRotationLayer([:Z, :Z], rectangletopology(2, 4)), PauliRotationLayer([:X, :Z], [(1, 2), (2, 1)]),
        PauliRotationLayer([:X, :X], any_bonds), PauliRotationLayer([:Z, :Z], any_bonds), heisenberg,
        PauliRotationLayer(randomcommutingrotations(rng, nq, 8))]
    matches = true
    for layer in layers
        plan = PP._preparelayer(layer.gates, randn(rng, length(layer.gates)), cache)
        pivots, reductions = PB._echelonbasis(plan.gate_masks)
        for position in PB._bitpositions(pivots)
            matches &= PB._reduce(plan.reduction, one(TT) << position) == reductions[position+1]
        end
        for gate_mask in plan.gate_masks
            matches &= iszero(PB._classrepresentative(plan, gate_mask))
        end

        for _ in 1:50
            pstr = PauliString(nq, rand(rng, [:I, :X, :Y, :Z], nq), 1:nq).term
            positions = PB._branchinggates(plan.lookup, pstr)
            representative = PB._classrepresentative(plan, pstr)
            matches &= PB._classrepresentative(plan, representative) == representative
            rotations = [rotation for rotation in eachindex(plan.gate_masks) if !commutes(plan.gate_masks[rotation], pstr)]
            matches &= [rotation for rotation in eachindex(plan.gate_masks) if !iszero(positions[(rotation-1)>>6+1] & (one(UInt64) << ((rotation - 1) & 63)))] == rotations
            for rotation in rotations
                partner = pstr ⊻ plan.gate_masks[rotation]
                matches &= PB._branchinggates(plan.lookup, partner) == positions
                matches &= PB._classrepresentative(plan, partner) == representative
            end
        end
    end
    return matches
end

@testset "PauliRotationLayer plans find the classes from the echelon basis" begin
    @test testplans()
end

# Every class shares one hash, so that the classes of a batch are told apart by their representatives alone. The method
# replaced here is deleted first, which redefining it would report as overwritten.
Base.delete_method(which(PB._classhash, Tuple{Any}))
@eval PauliPropagation.PropagationBase _classhash(representative) = 0

@testset "PauliRotationLayer rules on the Pauli strings a rotation acts on agree with the general rule" begin
    rng = MersenneTwister(9)
    for nq in (5, 40), gate in (PauliRotation(:Y, 2), PauliRotation([:X, :Z], [1, 3]), PauliRotation([:Z, :Z, :X], [2, 3, 4]))
        cache = PropagationCache(PauliSum(nq))
        theta = 0.4
        general = PP._branchrule(gate, cache, theta)
        acting = PB.onlimbs(PP.ActingRotationRule, PP._branchmask(gate, cache), cos(theta), sin(theta))
        floor = PB._creationfloor(acting, 1e-3)
        @test floor ≈ 1e-3 / abs(sin(theta)) && floor < 1e-3 / abs(sin(theta))
        n_acting = 0
        matches = true
        for _ in 1:200
            pstr = PauliString(nq, rand(rng, [:I, :X, :Y, :Z], nq), 1:nq).term
            outcome = general(pstr, 0.7)
            if outcome isa PB.Branch
                n_acting += 1
                partner = pstr ⊻ PP._branchmask(gate, cache)
                matches &= acting(pstr, 0.7) == outcome
                matches &= PB.branchpair(acting, pstr, 0.7, partner, -0.3) == PB.branchpair(general, pstr, 0.7, partner, -0.3)
                matches &= PB.branchpair(acting, pstr, 0.7, partner, 0.0) == PB.branchpair(general, pstr, 0.7, partner, 0.0)
            end
        end
        @test matches && n_acting > 20
    end
end

@testset "PauliRotationLayer with classes that share their hash" begin
    testlayers()
end

Base.delete_method(which(PB._classhash, Tuple{Any}))
@eval PauliPropagation.PropagationBase @inline _classhash(representative) = _hashbits(representative) % Int

# A class is a Pauli string with its products with the generators of the rotations it anticommutes with. The class kernel
# takes a class as two vectors, applies the rotations to it in the cache of a scratch and writes what it keeps to a
# Pauli sum, which has to hold what the rotations of the layer make of the class alone; a cache over a dictionary sum
# does the same, through the dictionary path of the gates.
function testclasskernelagainstrotations()
    nq = 8
    rng = MersenneTwister(11)
    cache = PropagationCache(PauliSum(nq))
    TT = paulitype(cache)
    classes = (
        (PauliRotationLayer([:Z, :Z], staircasetopology(nq; periodic=true)), PauliString(nq, [:X, :Y, :X], [1, 3, 6])),
        (PauliRotationLayer(:Y, 1:nq), PauliString(nq, [:X, :Z, :X, :Z], [1, 2, 5, 8])),
        (PauliRotationLayer([:X, :Y], [(i, i + 1) for i in 1:2:nq-1]), PauliString(nq, [:Z, :Z, :X], [1, 4, 6])),
    )
    matches = true
    for (layer, pstr) in classes, (max_weight, min_abs_coeff) in ((Inf, 0.0), (4.0, 1e-2))
        thetas = randn(rng, length(layer.gates))
        plan = PP._preparelayer(layer.gates, thetas, cache)
        truncfunc = buildtruncfunc(cache; min_abs_coeff, max_weight)
        rotations = [rotation for rotation in eachindex(plan.gate_masks) if !commutes(plan.gate_masks[rotation], pstr.term)]

        # members made by random sets of the anticommuting generators, some of them more than once
        class_terms = [reduce(xor, (plan.gate_masks[rotation] for rotation in rotations if rand(rng, Bool)); init=pstr.term) for _ in 1:8]
        class_coeffs = [rand(rng, (-1, 1)) * (0.2 + rand(rng)) for _ in 1:8]

        reference = PauliSum(nq)
        for (term, coeff) in zip(class_terms, class_coeffs)
            add!(reference, term, coeff)
        end
        for (rotation, angle) in zip(expandlayers([layer], [thetas])...)
            reference = propagate(rotation, reference, angle; min_abs_coeff, max_weight)
        end
        matches &= !isempty(reference) && length(rotations) > 1

        for class_cache in (PB.ClassScratch{TT,Float64}().cache, PropagationCache(PauliSum(nq)))
            by_class = PauliSum(nq)
            PB._applytoclass!(by_class, class_cache, plan.rules, plan, truncfunc, copy(class_terms), copy(class_coeffs))
            matches &= length(by_class) == length(reference) && by_class == reference
        end
    end
    return matches
end

@testset "PauliRotationLayer applies the rotations to one class like its rotations" begin
    @test testclasskernelagainstrotations()
end

@testset "PauliRotationLayer applied by several tasks" begin
    # the tasks are handed over directly, so that they are tested with any number of threads and terms
    for nq in (8, 100)
        rng = MersenneTwister(nq)
        psum = randompaulisum(rng, nq, 40, 4)
        add!(psum, repeat([:Y, :Z], 4), 1:8, 1.0)
        layers = [PauliRotationLayer([:X, :X], staircasetopology(nq)), PauliRotationLayer([:Z, :Z], staircasetopology(nq))]
        thetas = randomangles(rng, layers)
        vpsum = propagate(layers, VectorPauliSum(psum), thetas; max_weight=5.0, min_abs_coeff=1e-4)
        plans = [PauliPropagation._preparelayer(layer.gates, theta, PropagationCache(vpsum)) for (layer, theta) in zip(layers, thetas)]

        matches = true
        for plan in plans, capacity in (length(vpsum), 4 * length(vpsum))
            by_one_task = PropagationCache(deepcopy(vpsum))
            by_four_tasks = PropagationCache(deepcopy(vpsum))

            # with room for all that the tasks write, and with so little that they keep most of it in their buffers
            resize!(by_four_tasks, capacity)

            truncfunc = buildtruncfunc(by_one_task; min_abs_coeff=1e-4, max_weight=5.0)
            TT, CT = paulitype(vpsum), coefftype(vpsum)
            workspace = PB.LayerWorkspace(TT, CT, PB.ClassScratch{TT,CT})
            one_task = PauliPropagation.AK.TaskPartitioner(length(vpsum), 1, 1)
            four_tasks = PauliPropagation.AK.TaskPartitioner(length(vpsum), 4, 1)

            PB._applypass!(by_one_task, plan, truncfunc, workspace,
                PB._arraysources(by_one_task, workspace, one_task, 1), true)
            PB._applypass!(by_four_tasks, plan, truncfunc, workspace,
                PB._arraysources(by_four_tasks, workspace, four_tasks, 4), true)
            matches &= length(by_four_tasks) == length(by_one_task)
            matches &= PauliSum(extractsum!(by_four_tasks)) == PauliSum(extractsum!(by_one_task))
            matches &= PauliSum(extractsum!(by_one_task)) != PauliSum(vpsum)
        end
        @test matches
    end
end

@testset "PauliRotationLayer on a sum that several threads share out" begin
    # every Pauli string on eight qubits, more than the number of terms from which a sum is split into tasks
    nq = 8
    rng = MersenneTwister(nq)
    psum = PauliSum(nq, Dict{UInt16,Float64}(UInt16(i) => randn(rng) for i in 0:4^nq-1))
    layers = [PauliRotationLayer(:X, 1:nq), PauliRotationLayer([:Z, :Z], staircasetopology(nq))]
    thetas = randomangles(rng, layers)
    @test length(psum) > 2 * PB._MIN_ELEMS_PER_TASK

    reference = propagate(layers, psum, thetas; max_weight=6.0, min_abs_coeff=1e-8)

    for T in (VectorPauliSum, multivectorsum, multidictsum)
        layered = propagate(layers, T(psum), thetas; max_weight=6.0, min_abs_coeff=1e-8)
        @test length(layered) == length(reference)
        @test PauliSum(layered) == reference
    end
end

@testset "PauliRotationLayer branches large classes in passes like its rotations" begin
    # every Pauli string on six qubits, with coefficients from the cutoff to a hundred times it, so that the classes hold
    # 64 Pauli strings and many of their coefficients lie below what creates anything that the truncation keeps
    nq = 6
    rng = MersenneTwister(13)
    min_abs_coeff = 1e-6
    psum = PauliSum(nq, Dict{UInt16,Float64}(UInt16(i) => rand(rng, (-1, 1)) * min_abs_coeff * 100^rand(rng) for i in 0:4^nq-1))
    layers = [PauliRotationLayer(:X, 1:nq), PauliRotationLayer([:Z, :Z], staircasetopology(nq))]
    thetas = [0.1 * randn(rng, countparameters(layer)) for layer in layers]
    plan = PP._preparelayer(layers[1].gates, thetas[1], PropagationCache(psum))
    floor = PB._creationfloor(plan.rules[1], min_abs_coeff)
    @test count(coeff -> abs(coeff) < floor, coefficients(psum)) > length(psum) / 4
    @test layersmatchrotations(layers, thetas, psum; sums=VECTOR_TEST_SUMS, min_abs_coeff)
end

# A basis other than Pauli strings, built on PropagationBase alone: a term is a set of Majorana modes, the bits of an
# integer, and a rotation by two modes acts on the terms that hold one of them, flips both, and creates its sine with the
# sign of the number of modes between them. Its sums are held in dictionaries, and its cache holds sums of its own type.
struct MajoranaSum{TT,CT} <: PB.AbstractTermSum
    nsites::Int
    terms::Dict{TT,CT}
end

PB.storage(msum::MajoranaSum) = msum.terms
PB.nsites(msum::MajoranaSum) = msum.nsites

mutable struct MajoranaCache{MS<:MajoranaSum} <: PB.AbstractPropagationCache
    main_sum::MS
    aux_sum::MS
end

PB.PropagationCache(msum::MajoranaSum) = MajoranaCache(msum, similar(msum))
PB.mainsum(cache::MajoranaCache) = cache.main_sum
PB.auxsum(cache::MajoranaCache) = cache.aux_sum

function PB.setmainsum!(cache::MajoranaCache, msum)
    cache.main_sum = msum
    return cache
end

function PB.setauxsum!(cache::MajoranaCache, msum)
    cache.aux_sum = msum
    return cache
end

struct MajoranaRotation <: PB.ParametrizedGate
    modes::Tuple{Int,Int}
end

PB.qinds(gate::MajoranaRotation) = collect(gate.modes)

majoranamask(gate::MajoranaRotation, ::Type{TT}) where {TT} = (one(TT) << (gate.modes[1] - 1)) | (one(TT) << (gate.modes[2] - 1))

function majoranarule(mask::TT, cos_val, sin_val) where {TT}
    between = (one(TT) << (8 * sizeof(TT) - leading_zeros(mask) - 1)) - (one(TT) << (trailing_zeros(mask) + 1))
    function rotate(term, coeff)
        if iseven(count_ones(term & mask))
            return PB.Unchanged()
        end
        sign = 1 - 2 * (count_ones(term & between) & 1)
        return PB.Branch(coeff * cos_val, coeff * sin_val * sign)
    end
    return rotate
end

function majoranatruncation(min_abs_coeff)
    truncates(term, coeff) = abs(coeff) < min_abs_coeff
    return truncates
end

function PB.applymergetruncate!(gate::MajoranaRotation, cache::MajoranaCache, theta; min_abs_coeff=0.0)
    mask = majoranamask(gate, PB.termtype(cache))
    PB.xorbranchmergeandtruncate!(majoranarule(mask, cos(theta), sin(theta)), majoranatruncation(min_abs_coeff), cache, mask)
    return
end

# the layer through the pass of PropagationBase: the plan is all that the basis provides
function PB.applymergetruncate!(layer::GateLayer{MajoranaRotation}, cache::MajoranaCache, thetas; min_abs_coeff=0.0)
    masks = [majoranamask(gate, PB.termtype(cache)) for gate in layer.gates]
    rules = [majoranarule(mask, cos(theta), sin(theta)) for (mask, theta) in zip(masks, thetas)]
    plan = PB.LayerPlan(masks, rules, PB.PrecomputedLookup(masks))
    PB._applypass!(cache, plan, majoranatruncation(min_abs_coeff))
    return
end

@testset "A basis other than Pauli strings applies its layers through the same pass" begin
    n_modes = 12
    rng = MersenneTwister(17)
    terms = Dict{UInt16,Float64}(rand(rng, UInt16(0):UInt16(2^n_modes - 1)) => rand(rng, (-1, 1)) * (0.1 + rand(rng)) for _ in 1:300)
    layers = [GateLayer([MajoranaRotation((m, m + 1)) for m in 1:2:n_modes-1]), GateLayer([MajoranaRotation((m, m + 1)) for m in 2:2:n_modes-2])]
    thetas = [0.4 * randn(rng, length(layer.gates)) for layer in layers]
    for min_abs_coeff in (0.0, 1e-2)
        layered = PropagationCache(MajoranaSum(n_modes, copy(terms)))
        one_by_one = PropagationCache(MajoranaSum(n_modes, copy(terms)))
        for (layer, theta) in zip(layers, thetas)
            applymergetruncate!(layer, layered, theta; min_abs_coeff)
            for (gate, angle) in zip(layer.gates, theta)
                applymergetruncate!(gate, one_by_one, angle; min_abs_coeff)
            end
        end
        @test length(mainsum(layered)) == length(mainsum(one_by_one)) > length(terms)
        @test PB.storage(mainsum(layered)) == PB.storage(mainsum(one_by_one))
    end
end

@testset "PauliRotationLayer constructors and the order of its rotations" begin
    nq = 8
    x_layer = PauliRotationLayer(:X, 1:nq)
    zz_layer = PauliRotationLayer([:Z, :Z], staircasetopology(nq))

    @test x_layer isa GateLayer
    @test countparameters([x_layer, zz_layer]) == 2
    @test countparameters(x_layer) == nq
    zz_rotations = [PauliRotation([:Z, :Z], bond) for bond in staircasetopology(nq)]
    @test PauliRotationLayer(zz_rotations).gates == zz_rotations

    # the rotations are applied in the order of `qinds`, and in reverse in the Heisenberg picture
    layer = PauliRotationLayer([:Z, :Z], staircasetopology(4; periodic=true))
    @test [rotation.qinds for rotation in layer.gates] == [[1, 2], [2, 3], [3, 4], [4, 1]]
    layer = PauliRotationLayer(:X, [3, 1, 1, 2])
    @test [rotation.qinds for rotation in layer.gates] == [[3], [1], [1], [2]]
    heisenberg_layer, heisenberg_angles = PauliPropagation._toheisenberg(PauliRotationLayer(:X, [3, 1, 2]), [1.0, 2.0, 3.0])
    @test [rotation.qinds for rotation in heisenberg_layer.gates] == [[2], [1], [3]]
    @test heisenberg_angles == [3.0, 2.0, 1.0]

    # a layer equals its rotations in the order of `qinds`, whatever that order, in either picture; on this input, applying
    # the bonds in another order truncates differently
    rng = MersenneTwister(3)
    psum = randompaulisum(rng, nq, 8, 3)
    bonds = shuffle(rng, staircasetopology(nq; periodic=true))
    thetas = randn(rng, nq)
    bond_rotations = [PauliRotation([:Z, :Z], bond) for bond in bonds]
    for T in (PauliSum, VectorPauliSum), heisenberg in (true, false)
        layered = propagate(PauliRotationLayer([:Z, :Z], bonds), T(psum), thetas; heisenberg, max_weight=3.0, min_abs_coeff=1e-3)
        reference = propagate(bond_rotations, T(psum), thetas; heisenberg, max_weight=3.0, min_abs_coeff=1e-3)
        @test length(layered) == length(reference) && PauliSum(layered) == PauliSum(reference)
    end

    # generators that hold the identity on some of their qubits
    identity_layers = [PauliRotationLayer([:I, :Y], bonds), PauliRotationLayer([:X, :I], staircasetopology(nq)), PauliRotationLayer(:I, 1:nq)]
    @test layersmatchrotations(identity_layers, randomangles(rng, identity_layers), psum; sums=VECTOR_TEST_SUMS, max_weight=3.0, min_abs_coeff=1e-3)

    @test_throws ArgumentError PauliRotationLayer([:X, :Z], staircasetopology(nq))
    @test_throws ArgumentError PauliRotationLayer([:Z, :Z], [1, 2])
    @test_throws ArgumentError PauliRotationLayer([PauliRotation([:X, :Z], [1, 2]), PauliRotation([:X, :Z], [2, 3])])
    @test_throws ArgumentError PauliRotationLayer(:X, [0, 1])
    @test_throws ArgumentError propagate(zz_layer, PauliString(nq, :X, 1), [0.1, 0.2])
    @test_throws ArgumentError propagate(zz_layer, PauliString(nq, :X, 1), 0.1)
    @test_throws ArgumentError propagate(zz_layer, PauliString(nq - 1, :X, 1), fill(0.3, nq - 1))
end

# the circuit and its layers propagate alike, in either picture
function layeredmatchescircuit(circuit, thetas, psum; kwargs...)
    layered_circuit, layered_thetas = tolayers(circuit, thetas)
    matches = true
    for (T, heisenberg) in VECTOR_TEST_SUMS
        layered = propagate(layered_circuit, T(psum), layered_thetas; heisenberg, kwargs...)
        reference = propagate(circuit, T(psum), thetas; heisenberg, kwargs...)
        matches &= !isempty(layered) && length(layered) == length(reference) && PauliSum(layered) == PauliSum(reference)
    end
    return matches
end

@testset "tolayers gathers the commuting rotations of a circuit into layers" begin
    nq = 6
    circuit = Gate[
        PauliRotation([:Z, :Z], [1, 2]), PauliRotation([:Z, :Z], [2, 3]), PauliRotation(:Z, 4, 0.3), PauliRotation([:Z, :Z, :Z], [4, 5, 6]),
        # X on the first qubit anticommutes with ZZ on the first two, and a rotation may come twice
        PauliRotation(:X, 1), PauliRotation(:X, 2), PauliRotation(:X, 2),
        CliffordGate(:CNOT, [3, 4]),
        # a layer of frozen rotations takes no parameter
        PauliRotation(:Y, 5, 0.2), PauliRotation(:Y, 6, -0.4),
        CliffordGate(:H, [1]),
        # rotations on the same qubits that commute, and ZZ on the last two qubits, which anticommutes with X on the fifth on
        # its second qubit
        PauliRotation([:X, :X], [3, 4]), PauliRotation([:Y, :Y], [3, 4]), PauliRotation([:Z, :Z], [3, 4]), PauliRotation(:X, 5),
        PauliRotation([:Z, :Z], [6, 5]),
        PauliRotationLayer(:Z, 1:nq),
        PauliRotation(:Y, 1),
    ]
    rng = MersenneTwister(5)
    thetas = Any[randn(rng, 11)...]
    push!(thetas, randn(rng, nq))
    push!(thetas, randn(rng))

    layered_circuit, layered_thetas = tolayers(circuit, thetas)
    @test repr.(layered_circuit) == ["GateLayer(3 PauliRotation, 1 FrozenGate)", "GateLayer(3 PauliRotation)", "CliffordGate(:CNOT, [3, 4])",
        "GateLayer(2 FrozenGate)", "CliffordGate(:H, [1])", "GateLayer(4 PauliRotation)",
        "PauliRotation([:Z, :Z], [6, 5])", "GateLayer(6 PauliRotation)", "PauliRotation([:Y], [1])"]
    @test layered_thetas == [thetas[1:3], thetas[4:6], thetas[7:10], thetas[11], thetas[12], thetas[13]]
    @test repr.(tolayers(circuit)) == repr.(layered_circuit)
    @test isempty(tolayers(Gate[]))

    psum = randompaulisum(rng, nq, 8, 3)
    @test layeredmatchescircuit(circuit, thetas, psum; min_abs_coeff=0.0)
    @test layeredmatchescircuit(circuit, thetas, psum; max_weight=3.0, min_abs_coeff=1e-3)

    # the gradient by the parameters of the layers is that by the parameters of the circuit
    value, gradient = rewindgradient(circuit, psum, thetas, overlapwithzero; min_abs_coeff=0.0)
    layered_value, layered_gradient = rewindgradient(layered_circuit, psum, layered_thetas, overlapwithzero; min_abs_coeff=0.0)
    @test layered_value ≈ value
    @test reduce(vcat, layered_gradient) ≈ reduce(vcat, gradient)

    # a Trotter circuit becomes a layer of ZZ rotations and a layer of X rotations per step
    @test repr.(tolayers(tfitrottercircuit(nq, 2))) == ["GateLayer(5 PauliRotation)", "GateLayer(6 PauliRotation)", "GateLayer(5 PauliRotation)", "GateLayer(6 PauliRotation)"]
end

@testset "PauliRotationLayer truncates and merges what it is given" begin
    nq = 8
    layer = PauliRotationLayer(:X, 1:nq)
    thetas = fill(0.4, nq)
    max_weight = 2.0

    # a Pauli string above the weight limit, which the layer touches, and one it leaves alone
    heavy = PauliSum(nq)
    add!(heavy, [:Z, :Z, :Z], [1, 2, 3], 1.0)
    add!(heavy, [:X, :X, :X], [4, 5, 6], 1.0)
    add!(heavy, [:Y, :Z, :Y], [6, 7, 8], 1.0)
    add!(heavy, [:Z], [5], 1.0)
    for T in (PauliSum, VectorPauliSum)
        layered = propagate(layer, T(heavy), thetas; max_weight, min_abs_coeff=0.0)
        @test !isempty(layered)
        @test all(pstr -> countweight(pstr) <= max_weight, paulis(layered))
    end

    # Pauli strings that come twice are merged where a rotation touches them
    twice = VectorPauliSum(nq, [paulis(VectorPauliSum(PauliString(nq, :Z, 2))); paulis(VectorPauliSum(PauliString(nq, :Z, 2)))], [0.5, 0.25])
    layered = propagate(layer, twice, thetas; min_abs_coeff=0.0)
    reference = propagate(layer, PauliString(nq, :Z, 2, 0.75), thetas; min_abs_coeff=0.0)
    @test PauliSum(layered) == reference
    @test length(layered) == 2

    # and so are Pauli strings that no rotation touches
    untouched_twice = VectorPauliSum(nq, [paulis(VectorPauliSum(PauliString(nq, :X, 3))); paulis(VectorPauliSum(PauliString(nq, :X, 3)))], [0.5, 0.25])
    layered = propagate(layer, untouched_twice, thetas; min_abs_coeff=0.0)
    @test PauliSum(layered) == propagate(layer, PauliString(nq, :X, 3, 0.75), thetas; min_abs_coeff=0.0)
    @test length(layered) == 1
end

@testset "PauliRotationLayer takes its rotations one by one where it has to" begin
    nq = 8
    rng = MersenneTwister(3)
    psum = randompaulisum(rng, nq, 6, 3)
    layers = [PauliRotationLayer(:X, 1:nq), PauliRotationLayer([:Z, :Z], staircasetopology(nq))]
    thetas = randomangles(rng, layers)
    rotations, angles = expandlayers(layers, thetas)

    # coefficients that track their frequencies, and a truncation relative to the largest coefficient
    @test propagate(layers, psum, thetas; max_freq=3.0, min_abs_coeff=1e-6) == propagate(rotations, psum, angles; max_freq=3.0, min_abs_coeff=1e-6)
    for T in (PauliSum, VectorPauliSum)
        @test propagate(layers, T(psum), thetas; min_rel_coeff=1e-3) == propagate(rotations, T(psum), angles; min_rel_coeff=1e-3)
    end
end
