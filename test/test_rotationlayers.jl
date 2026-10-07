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

# every kind of sum in the Heisenberg picture, and the two that the others are made of in the Schrödinger picture as well
const LAYER_TEST_SUMS = ((PauliSum, true), (VectorPauliSum, true), (multivectorsum, true), (multidictsum, true),
    (PauliSum, false), (VectorPauliSum, false))

function layersmatchrotations(layers, thetas, psum; kwargs...)
    rotations, angles = expandlayers(layers, thetas)
    matches = true
    for (T, heisenberg) in LAYER_TEST_SUMS
        layered = propagate(layers, T(psum), thetas; heisenberg, kwargs...)
        reference = propagate(rotations, T(psum), angles; heisenberg, kwargs...)
        # the same terms, as two sums also compare equal if one of them holds more terms with a coefficient of zero
        matches &= !isempty(layered) && length(layered) == length(reference) && PauliSum(layered) == PauliSum(reference)
    end
    return matches
end

function testlayersagainstrotations()
    # on one limb, on two and on four
    for nq in (8, 40, 100)
        rng = MersenneTwister(nq)
        psum = randompaulisum(rng, nq, 12, 3)
        chain = staircasetopology(nq)
        ring = staircasetopology(nq; periodic=true)
        disjoint_bonds = [(i, i + 1) for i in 1:2:nq-1]
        distant_bonds = [(i + 3, i) for i in 1:nq-3]

        # rotations that share qubits and rotations that do not, in the order of the qubits and in any other
        layer_sets = (
            [PauliRotationLayer(:X, 1:nq), PauliRotationLayer([:Z, :Z], chain)],
            [PauliRotationLayer(:Y, nq:-1:1), PauliRotationLayer([:X, :X], ring)],
            [PauliRotationLayer(:Z, randperm(rng, nq)), PauliRotationLayer([:Y, :Y], shuffle(rng, chain))],
            [PauliRotationLayer([:X, :Y], disjoint_bonds), PauliRotationLayer([:Z, :Z], distant_bonds)],
            # every bond in both directions with different Paulis, so that every qubit is acted on with X and with Z
            [PauliRotationLayer(:Y, 1:nq), PauliRotationLayer([:X, :Z], vcat(disjoint_bonds, reverse.(disjoint_bonds)))],
        )

        # without any truncation only where 4^nq bounds the sum
        truncations = nq == 8 ? ((Inf, 0.0), (4.0, 0.0), (Inf, 1e-3)) : ((4.0, 0.0), (3.0, 1e-4))

        matches = true
        for layers in layer_sets, (max_weight, min_abs_coeff) in truncations
            circuit = repeat(layers, 2)
            thetas = randomangles(rng, circuit)
            matches &= layersmatchrotations(circuit, thetas, psum; max_weight, min_abs_coeff)
        end
        @test matches
    end
end

function testanyqubitsandpairs()
    # some of the qubits in any order, and bonds between any two qubits, which share qubits and come more than once
    for nq in (8, 100)
        rng = MersenneTwister(nq)
        psum = randompaulisum(rng, nq, 12, 3)
        some_qubits = randperm(rng, nq)[1:nq÷2]
        any_bonds = [Tuple(randperm(rng, nq)[1:2]) for _ in 1:nq]

        layers = [PauliRotationLayer(:Y, some_qubits), PauliRotationLayer([:Z, :Z], any_bonds),
            PauliRotationLayer(:X, some_qubits), PauliRotationLayer([:X, :X], any_bonds)]
        thetas = randomangles(rng, layers)

        @test layersmatchrotations(layers, thetas, psum; max_weight=4.0, min_abs_coeff=1e-4)
        @test layersmatchrotations(layers, thetas, psum; max_weight=3.0, min_abs_coeff=0.0)
    end
end

function testclassesofmanyrotations()
    # heavy Pauli strings anticommute with more rotations than a dense class takes
    nq = 100
    rng = MersenneTwister(nq)
    psum = randompaulisum(rng, nq, 6, 12)
    add!(psum, repeat([:Y, :Z], 5), 1:10, 1.0)
    layers = [PauliRotationLayer(:X, 1:nq), PauliRotationLayer([:Z, :Z], staircasetopology(nq)), PauliRotationLayer(:Y, nq:-1:1)]
    thetas = randomangles(rng, layers)

    @test any(pstr -> countyz(pstr) > 8, paulis(psum))
    @test layersmatchrotations(layers, thetas, psum; max_weight=Inf, min_abs_coeff=2e-2)
    @test layersmatchrotations(layers, thetas, psum; max_weight=9.0, min_abs_coeff=5e-3)
end

function testlargeclassoftwostrings()
    # two Pauli strings of one class of ten rotations, which must be collected in one place to be mixed
    nq = 12
    psum = PauliSum(nq)
    yz = repeat([:Y, :Z], 5)
    add!(psum, yz, 1:10, 1.0)
    add!(psum, [yz[1:2]; :Z; yz[4:10]], 1:10, 0.5)
    layers = [PauliRotationLayer(:X, 1:nq)]
    @test layersmatchrotations(layers, [fill(0.3, nq)], psum; min_abs_coeff=1e-3)
end

function testclassesagainstrotations()
    for nq in (8, 40)
        rng = MersenneTwister(nq)
        psum = randompaulisum(rng, nq, 12, 3)

        # bonds that share qubits, close cycles, meet at one qubit or come more than once
        layer_sets = (
            [PauliRotationLayer(:X, 1:nq), PauliRotationLayer([:Z, :Z], staircasetopology(nq))],
            [PauliRotationLayer(:Y, nq:-1:1), PauliRotationLayer([:X, :X], staircasetopology(nq; periodic=true))],
            [PauliRotationLayer(:X, 1:nq), PauliRotationLayer([:Z, :Z], rectangletopology(2, nq ÷ 2))],
            [PauliRotationLayer(:Y, 1:nq), PauliRotationLayer([:Z, :Z], [(1, i) for i in 2:nq])],
            [PauliRotationLayer([:X, :Y], [(i, i + 1) for i in 1:2:nq-1]), PauliRotationLayer([:Z, :Z], [Tuple(randperm(rng, nq)[1:2]) for _ in 1:nq])],
            [PauliRotationLayer(:Y, 1:nq), PauliRotationLayer([:X, :Z], vcat([(i, i + 1) for i in 1:2:nq-1], [(i + 1, i) for i in 1:2:nq-1]))],
        )
        truncations = nq == 8 ? ((Inf, 0.0), (4.0, 0.0), (Inf, 1e-3)) : ((3.0, 1e-4), (Inf, 2e-2))
        matches = true
        for layers in layer_sets, (max_weight, min_abs_coeff) in truncations
            circuit = repeat(layers, 2)
            thetas = randomangles(rng, circuit)
            matches &= layersmatchrotations(circuit, thetas, psum; max_weight, min_abs_coeff)
        end
        @test matches
    end
end

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

function testwiderotations()
    for nq in (8, 40)
        rng = MersenneTwister(nq + 1)
        psum = randompaulisum(rng, nq, 12, 3)
        triples = [(i, i + 1, i + 2) for i in 1:nq-2]
        # the squares of a ladder of two rows, with every qubit acted on with X and with Z
        squares = [(i, i + 1, i + nq ÷ 2, i + nq ÷ 2 + 1) for i in 1:nq÷2-1]

        layer_sets = (
            [PauliRotationLayer(:X, 1:nq), PauliRotationLayer([:Z, :Z, :Z], triples)],
            [GateLayer(vcat(PauliRotationLayer([:X, :X, :X, :X], squares).gates, PauliRotationLayer([:Z, :Z, :Z, :Z], squares).gates); guaranteed_commutes=true)],
            [PauliRotationLayer(randomcommutingrotations(rng, nq, nq ÷ 2 + 2)), PauliRotationLayer(randomcommutingrotations(rng, nq, nq ÷ 2 + 2))],
        )
        truncations = nq == 8 ? ((Inf, 0.0), (4.0, 0.0), (Inf, 1e-3)) : ((4.0, 0.0), (3.0, 1e-4))
        matches = true
        for layers in layer_sets, (max_weight, min_abs_coeff) in truncations
            circuit = repeat(layers, 2)
            thetas = randomangles(rng, circuit)
            matches &= layersmatchrotations(circuit, thetas, psum; max_weight, min_abs_coeff)
        end
        @test matches
    end
end

# A class is rotated densely if it has few distinguishing bits, and sparsely otherwise. Both ways are tested on the same
# layers.
function testlayers()
    testlayersagainstrotations()
    testanyqubitsandpairs()
    testclassesofmanyrotations()
    testlargeclassoftwostrings()
    testclassesagainstrotations()
    testwiderotations()
end

@testset "PauliRotationLayer propagates like its rotations" begin
    testlayers()
end

@testset "PauliRotationLayer acting on a qubit with two Paulis" begin
    # XI commutes with XX but not with ZZ, which act on its first qubit with X and with Z
    xx_zz = GateLayer([PauliRotation([:X, :X], [1, 2]), PauliRotation([:Z, :Z], [1, 2])]; guaranteed_commutes=true)
    @test layersmatchrotations([xx_zz], [[0.3, 0.7]], PauliSum(PauliString(2, :X, 1)); min_abs_coeff=0.0)

    # the identity, which commutes with XZ on (1, 2) and on (2, 1), and XX, which anticommutes with both, in either order
    # of the two in the sum
    xz_zx = PauliRotationLayer([:X, :Z], [(1, 2), (2, 1)])
    identity, xx = PauliString(2, :I, 1), PauliString(2, [:X, :X], [1, 2])
    matches = true
    for psum in (PauliSum([identity, xx]), PauliSum([xx, identity]))
        matches &= layersmatchrotations([xz_zx], [[0.3, 0.7]], psum; min_abs_coeff=0.0)
    end
    @test matches
end

# The echelon basis of a plan spans the generators of its rotations, and its reduction adds up the other bits of the
# basis vectors of any pivot bits. The partners of a Pauli string under the rotations it anticommutes with anticommute
# with the same rotations and have its representative, and where the rotations act on disjoint pairs of qubits, a class
# has as many distinguishing bits as the rank of its rotations.
function testplans()
    nq = 12
    rng = MersenneTwister(5)
    TT = getinttype(nq)
    any_bonds = [Tuple(randperm(rng, nq)[1:2]) for _ in 1:nq]
    heisenberg = GateLayer([PauliRotation([s, s], [i, i + 1]) for s in (:X, :Y, :Z) for i in 1:2:nq-1]; guaranteed_commutes=true)
    layers = [PauliRotationLayer(:Z, 1:nq), PauliRotationLayer([:Z, :Z], staircasetopology(nq; periodic=true)),
        PauliRotationLayer([:Z, :Z], rectangletopology(3, 4)), PauliRotationLayer([:X, :Z], [(1, 2), (2, 1)]),
        PauliRotationLayer([:X, :X], any_bonds), PauliRotationLayer([:Z, :Z], any_bonds), heisenberg,
        PauliRotationLayer(randomcommutingrotations(rng, nq, 8))]
    matches = true
    for layer in layers
        plan = PP._prepareclasses(layer.gates, randn(rng, length(layer.gates)), TT, nq)
        pivots, reductions = PB._echelonbasis(plan.gate_masks)
        for position in PB._bitpositions(pivots)
            matches &= PB._reduce(plan.reduction, one(TT) << position) == reductions[position+1]
        end
        for gate_mask in plan.gate_masks
            matches &= iszero(PB._classrepresentative(plan, gate_mask, gate_mask))
        end

        for _ in 1:50
            pstr = PauliString(nq, rand(rng, [:I, :X, :Y, :Z], nq), 1:nq).term
            positions, flipped = PB._branchinggates(plan.lookup, pstr)
            representative = PB._classrepresentative(plan, pstr, flipped)
            rotations = [rotation for rotation in eachindex(plan.gate_masks) if !commutes(plan.gate_masks[rotation], pstr)]
            matches &= flipped == reduce(|, plan.gate_masks[rotations]; init=zero(TT))
            for rotation in rotations
                partner = pstr ⊻ plan.gate_masks[rotation]
                matches &= PB._branchinggates(plan.lookup, partner) == (positions, flipped)
                matches &= PB._classrepresentative(plan, partner, flipped) == representative
            end
            if layer === heisenberg
                rank = count_ones(first(PB._echelonbasis(plan.gate_masks[rotations])))
                matches &= count_ones(PB._distinguishingbits(plan, flipped)) == rank
            end
        end
    end
    return matches
end

@testset "PauliRotationLayer plans find the classes from the echelon basis" begin
    @test testplans()
end

@eval PauliPropagation.PropagationBase _rotatesdense(n_distinguishing_bits::Int) = false

@testset "PauliRotationLayer with every class sparse" begin
    testlayers()
end

@eval PauliPropagation.PropagationBase _rotatesdense(n_distinguishing_bits::Int) = n_distinguishing_bits <= _MAX_DENSE_DISTINGUISHING_BITS

# every class shares one hash, so that the classes of a batch are told apart by their representatives alone
@eval PauliPropagation.PropagationBase _classlabel(plan, term) = 0

@testset "PauliRotationLayer with classes that share their hash" begin
    testlayers()
end

@eval PauliPropagation.PropagationBase @inline _classlabel(plan, term) = _hashbits(_classrepresentative(plan, term, _flippedbits(plan.lookup, term))) % Int

# A class is a Pauli string with its products with the generators of the rotations it anticommutes with. Each kernel takes
# a class as two vectors and writes what it keeps to a Pauli sum, which has to hold what the rotations of the layer make
# of the class alone.
function testkernelsagainstrotations()
    nq = 8
    rng = MersenneTwister(11)
    TT = getinttype(nq)
    classes = (
        (PauliRotationLayer([:Z, :Z], staircasetopology(nq; periodic=true)), PauliString(nq, [:X, :Y, :X], [1, 3, 6])),
        (PauliRotationLayer(:Y, 1:nq), PauliString(nq, [:X, :Z, :X, :Z], [1, 2, 5, 8])),
        (PauliRotationLayer([:X, :Y], [(i, i + 1) for i in 1:2:nq-1]), PauliString(nq, [:Z, :Z, :X], [1, 4, 6])),
    )
    matches = true
    n_dense = 0
    for (layer, pstr) in classes, (max_weight, min_abs_coeff) in ((Inf, 0.0), (4.0, 1e-2))
        thetas = randn(rng, length(layer.gates))
        plan = PP._prepareclasses(layer.gates, thetas, TT, nq; min_abs_coeff)
        truncfunc = buildtruncfunc(PropagationCache(PauliSum(nq)); min_abs_coeff, max_weight)
        positions, flipped = PB._branchinggates(plan.lookup, pstr.term)
        rotation_buffer = zeros(Int32, length(layer.gates))
        rotations = view(rotation_buffer, 1:PB._rotationsinorder!(rotation_buffer, plan, positions))
        distinguishing_bits = PB._distinguishingbits(plan, flipped)
        n_distinguishing_bits = sum(count_ones, PB._limbs(distinguishing_bits))

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

        sparse = PB.SparseScratch{TT,Float64}()
        by_sparse = PauliSum(nq)
        PB._rotatesparse!(by_sparse, sparse, plan, truncfunc, class_terms, class_coeffs, rotations, sparse.entry_coordinates, distinguishing_bits, n_distinguishing_bits)

        matches &= !isempty(reference) && length(rotations) > 1
        matches &= length(by_sparse) == length(reference) && by_sparse == reference

        # a class of few distinguishing bits is rotated densely too
        if n_distinguishing_bits <= PB._MAX_DENSE_DISTINGUISHING_BITS
            by_dense = PauliSum(nq)
            PB._rotatedense!(by_dense, PB.DenseScratch{TT,Float64}(), plan, truncfunc, class_terms, class_coeffs, rotations, distinguishing_bits, n_distinguishing_bits)
            matches &= length(by_dense) == length(reference) && by_dense == reference
            n_dense += 1
        end
    end
    return matches && n_dense >= 2
end

@testset "PauliRotationLayer kernels rotate one class like its rotations" begin
    @test testkernelsagainstrotations()
end

@testset "PauliRotationLayer applied by several tasks" begin
    # the tasks are handed over directly, so that they are tested with any number of threads and terms
    for nq in (8, 100)
        rng = MersenneTwister(nq)
        psum = randompaulisum(rng, nq, 40, 4)
        add!(psum, repeat([:Y, :Z], 4), 1:8, 1.0)
        layers = [PauliRotationLayer(:X, 1:nq), PauliRotationLayer([:Z, :Z], staircasetopology(nq))]
        thetas = randomangles(rng, layers)
        vpsum = propagate(layers, VectorPauliSum(psum), thetas; max_weight=5.0, min_abs_coeff=1e-4)
        plans = [PauliPropagation._prepareclasses(layer.gates, theta, paulitype(vpsum), nq) for (layer, theta) in zip(layers, thetas)]

        matches = true
        for plan in plans, capacity in (length(vpsum), 4 * length(vpsum))
            by_one_task = PropagationCache(deepcopy(vpsum))
            by_four_tasks = PropagationCache(deepcopy(vpsum))

            # with room for all that the tasks write, and with so little that they keep most of it in their buffers
            resize!(by_four_tasks, capacity)

            truncfunc = buildtruncfunc(by_one_task; min_abs_coeff=1e-4, max_weight=5)
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
    # the magnetisation on a chain, which grows past the number of terms from which a sum is split into tasks
    nq = 100
    psum = PauliSum(nq)
    for qind in 1:nq
        add!(psum, :Z, qind, 1 / nq)
    end
    layers = repeat([PauliRotationLayer(:X, 1:nq), PauliRotationLayer([:Z, :Z], staircasetopology(nq))], 4)
    thetas = [fill(pi / 4, countparameters(layer)) for layer in layers]

    reference = propagate(layers, psum, thetas; max_weight=6.0, min_abs_coeff=1e-8)
    @test length(reference) > 2 * PB._MIN_ELEMS_PER_TASK

    for T in (VectorPauliSum, multivectorsum, multidictsum)
        layered = propagate(layers, T(psum), thetas; max_weight=6.0, min_abs_coeff=1e-8)
        @test length(layered) == length(reference)
        @test PauliSum(layered) == reference
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
    @test layersmatchrotations(identity_layers, randomangles(rng, identity_layers), psum; max_weight=3.0, min_abs_coeff=1e-3)

    @test_throws ArgumentError PauliRotationLayer([:X, :Z], staircasetopology(nq))
    @test_throws ArgumentError PauliRotationLayer([:Z, :Z], [1, 2])
    @test_throws ArgumentError PauliRotationLayer([PauliRotation([:X, :Z], [1, 2]), PauliRotation([:X, :Z], [2, 3])])
    @test_throws ArgumentError PauliRotationLayer(:X, [0, 1])
    @test_throws ArgumentError propagate(zz_layer, PauliString(nq, :X, 1), [0.1, 0.2])
    @test_throws ArgumentError propagate(zz_layer, PauliString(nq, :X, 1), 0.1)
    @test_throws ArgumentError propagate(zz_layer, PauliString(nq - 1, :X, 1), fill(0.3, nq - 1))
end

# the circuit and its layers propagate alike, on every kind of sum and in either picture
function layeredmatchescircuit(circuit, thetas, psum; kwargs...)
    layered_circuit, layered_thetas = tolayers(circuit, thetas)
    matches = true
    for (T, heisenberg) in LAYER_TEST_SUMS
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
        "FrozenGate(GateLayer(2 PauliRotation), parameter = [0.2, -0.4])", "CliffordGate(:H, [1])", "GateLayer(4 PauliRotation)",
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
