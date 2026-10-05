using Random
using Test

# the rotations of the layers as written, those of every layer in the order of its `qinds`, built without `torotations`
function expandlayers(layers, thetas)
    rotations = Gate[]
    angles = Float64[]
    for (layer, theta) in zip(layers, thetas)
        for (index, rotation_qinds) in enumerate(layer.qinds)
            push!(rotations, PauliRotation(layer.symbols, rotation_qinds))
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

# one angle for every other layer, and one angle per rotation for the rest
randomangles(rng, layers) = Any[isodd(i) ? randn(rng) : randn(rng, length(layer.qinds)) for (i, layer) in enumerate(layers)]

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
            [RotationLayer(:X, 1:nq), RotationLayer([:Z, :Z], chain)],
            [RotationLayer(:Y, nq:-1:1), RotationLayer([:X, :X], ring)],
            [RotationLayer(:Z, randperm(rng, nq)), RotationLayer([:Y, :Y], shuffle(rng, chain))],
            [RotationLayer([:X, :Y], disjoint_bonds), RotationLayer([:Z, :Z], distant_bonds)],
            # every bond in both directions with different Paulis, so that every qubit is acted on with X and with Z
            [RotationLayer(:Y, 1:nq), RotationLayer([:X, :Z], vcat(disjoint_bonds, reverse.(disjoint_bonds)))],
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

        layers = [RotationLayer(:Y, some_qubits), RotationLayer([:Z, :Z], any_bonds),
            RotationLayer(:X, some_qubits), RotationLayer([:X, :X], any_bonds)]
        thetas = randomangles(rng, layers)

        # bonds that share their lowest qubit cannot be read in groups
        bond_layer = layers[2]
        plan = PauliPropagation._prepareclasses(bond_layer, thetas[2], getinttype(nq), Float64, nq)
        @test !plan.reader.reads_in_groups

        @test layersmatchrotations(layers, thetas, psum; max_weight=4.0, min_abs_coeff=1e-4)
        @test layersmatchrotations(layers, thetas, psum; max_weight=3.0, min_abs_coeff=0.0)
    end
end

function testclassesofmanyrotations()
    # heavy Pauli strings anticommute with more rotations than a dense block takes
    nq = 100
    rng = MersenneTwister(nq)
    psum = randompaulisum(rng, nq, 6, 12)
    add!(psum, repeat([:Y, :Z], 5), 1:10, 1.0)
    layers = [RotationLayer(:X, 1:nq), RotationLayer([:Z, :Z], staircasetopology(nq)), RotationLayer(:Y, nq:-1:1)]
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
    layers = [RotationLayer(:X, 1:nq)]
    @test layersmatchrotations(layers, [0.3], psum; min_abs_coeff=1e-3)
end

function testclassesagainstrotations()
    for nq in (8, 40)
        rng = MersenneTwister(nq)
        psum = randompaulisum(rng, nq, 12, 3)

        # bonds that share qubits, close cycles, meet at one qubit or come more than once
        layer_sets = (
            [RotationLayer(:X, 1:nq), RotationLayer([:Z, :Z], staircasetopology(nq))],
            [RotationLayer(:Y, nq:-1:1), RotationLayer([:X, :X], staircasetopology(nq; periodic=true))],
            [RotationLayer(:X, 1:nq), RotationLayer([:Z, :Z], rectangletopology(2, nq ÷ 2))],
            [RotationLayer(:Y, 1:nq), RotationLayer([:Z, :Z], [(1, i) for i in 2:nq])],
            [RotationLayer([:X, :Y], [(i, i + 1) for i in 1:2:nq-1]), RotationLayer([:Z, :Z], [Tuple(randperm(rng, nq)[1:2]) for _ in 1:nq])],
            [RotationLayer(:Y, 1:nq), RotationLayer([:X, :Z], vcat([(i, i + 1) for i in 1:2:nq-1], [(i + 1, i) for i in 1:2:nq-1]))],
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

# The rotations of a layer are read from the whole Pauli string at once if their qubits are few distances apart, and
# through byte tables otherwise, and a class is rotated as a dense block if it has few key bits, and in a table otherwise.
# Every way is tested on the same layers.
function testlayers()
    testlayersagainstrotations()
    testanyqubitsandpairs()
    testclassesofmanyrotations()
    testlargeclassoftwostrings()
    testclassesagainstrotations()
end

@testset "RotationLayer propagates like its rotations" begin
    testlayers()
end

@eval PauliPropagation _readsingroups(groups) = false

@testset "RotationLayer with its rotations read through tables" begin
    nq = 8
    layer = RotationLayer([:Z, :Z], staircasetopology(nq))
    @test !PauliPropagation._prepareclasses(layer, 0.3, getinttype(nq), Float64, nq).reader.reads_in_groups
    testlayers()
end

@eval PauliPropagation _readsingroups(groups) = length(groups) <= _MAX_ROTATION_GROUPS

@eval PauliPropagation _rotatesasblock(n_key_bits::Int) = false

@testset "RotationLayer with every class in a table" begin
    testlayers()
end

@eval PauliPropagation _rotatesasblock(n_key_bits::Int) = n_key_bits <= 12

@testset "RotationLayer with the classes of up to 12 key bits as blocks" begin
    testlayers()
end

@eval PauliPropagation _rotatesasblock(n_key_bits::Int) = n_key_bits <= _MAX_BLOCK_KEY_BITS

# every class shares one hash, so that the classes of a partition are told apart by their keys alone
@eval PauliPropagation _classlabel(plan, pstr) = 0

@testset "RotationLayer with classes that share their hash" begin
    testlayers()
end

@eval PauliPropagation @inline _classlabel(plan, pstr) = _hashbits(_classkey(pstr, _touchedqubits(plan.reader, pstr))) % Int

# A class is a Pauli string with its products with the generators of the rotations it anticommutes with. Each kernel takes
# a class as two vectors and writes what it keeps to a Pauli sum, which has to hold what the rotations of the layer make
# of the class alone.
function testkernelsagainstrotations()
    nq = 8
    rng = MersenneTwister(11)
    TT = getinttype(nq)
    classes = (
        (RotationLayer([:Z, :Z], staircasetopology(nq; periodic=true)), PauliString(nq, [:X, :Y, :X], [1, 3, 6])),
        (RotationLayer(:Y, 1:nq), PauliString(nq, [:X, :Z, :X, :Z], [1, 2, 5, 8])),
        (RotationLayer([:X, :Y], [(i, i + 1) for i in 1:2:nq-1]), PauliString(nq, [:Z, :Z, :X], [1, 4, 6])),
    )
    matches = true
    for (layer, pstr) in classes, (max_weight, min_abs_coeff) in ((Inf, 0.0), (4.0, 1e-2))
        thetas = randn(rng, length(layer.qinds))
        plan = PP._prepareclasses(layer, thetas, TT, Float64, nq; min_abs_coeff)
        truncfunc = buildtruncfunc(PropagationCache(PauliSum(nq)); min_abs_coeff, max_weight)
        positions, touched = PP._anticommuting(plan.reader, pstr.term)
        rotation_buffer = zeros(Int32, length(layer.qinds))
        rotations = view(rotation_buffer, 1:PP._rotationsinorder!(rotation_buffer, plan, positions))
        key_bits = PP._keybits(touched)
        n_key_bits = sum(count_ones, PP._limbs(key_bits))

        # members made by random sets of the anticommuting generators, some of them more than once
        class_terms = [reduce(xor, (plan.masks[rotation] for rotation in rotations if rand(rng, Bool)); init=pstr.term) for _ in 1:8]
        class_coeffs = [rand(rng, (-1, 1)) * (0.2 + rand(rng)) for _ in 1:8]

        reference = PauliSum(nq)
        for (term, coeff) in zip(class_terms, class_coeffs)
            add!(reference, term, coeff)
        end
        for (rotation, angle) in zip(expandlayers([layer], [thetas])...)
            reference = propagate(rotation, reference, angle; min_abs_coeff, max_weight)
        end

        by_block = PauliSum(nq)
        PP._rotateblock!(by_block, PP.BlockScratch{TT,Float64}(), plan, truncfunc, class_terms, class_coeffs, rotations, key_bits, n_key_bits)
        table = PP.TableScratch{TT,Float64}()
        by_table = PauliSum(nq)
        PP._rotatetable!(by_table, table, plan, truncfunc, class_terms, class_coeffs, rotations, table.entry_keys, key_bits, n_key_bits)

        matches &= !isempty(reference) && length(rotations) > 1
        matches &= length(by_block) == length(reference) && by_block == reference
        matches &= length(by_table) == length(reference) && by_table == reference
    end
    return matches
end

@testset "RotationLayer kernels rotate one class like its rotations" begin
    @test testkernelsagainstrotations()
end

@testset "RotationLayer applied by several tasks" begin
    # the tasks are handed over directly, so that they are tested with any number of threads and terms
    for nq in (8, 100)
        rng = MersenneTwister(nq)
        psum = randompaulisum(rng, nq, 40, 4)
        add!(psum, repeat([:Y, :Z], 4), 1:8, 1.0)
        layers = [RotationLayer(:X, 1:nq), RotationLayer([:Z, :Z], staircasetopology(nq))]
        thetas = randomangles(rng, layers)
        vpsum = propagate(layers, VectorPauliSum(psum), thetas; max_weight=5.0, min_abs_coeff=1e-4)
        plans = [PauliPropagation._prepareclasses(layer, theta, paulitype(vpsum), coefftype(vpsum), nq) for (layer, theta) in zip(layers, thetas)]
        @test all(plan -> plan.reader.reads_in_groups, plans)

        matches = true
        for plan in plans, capacity in (length(vpsum), 4 * length(vpsum))
            by_one_task = PropagationCache(deepcopy(vpsum))
            by_four_tasks = PropagationCache(deepcopy(vpsum))

            # with room for all that the tasks write, and with so little that they keep most of it in their buffers
            resize!(by_four_tasks, capacity)

            truncfunc = buildtruncfunc(by_one_task; min_abs_coeff=1e-4, max_weight=5)
            workspace = PauliPropagation.LayerWorkspace(paulitype(vpsum), coefftype(vpsum))
            one_task = PauliPropagation.AK.TaskPartitioner(length(vpsum), 1, 1)
            four_tasks = PauliPropagation.AK.TaskPartitioner(length(vpsum), 4, 1)

            PauliPropagation._applypass!(by_one_task, plan, truncfunc, workspace,
                PauliPropagation._arraysources(by_one_task, workspace, one_task, 1), true)
            PauliPropagation._applypass!(by_four_tasks, plan, truncfunc, workspace,
                PauliPropagation._arraysources(by_four_tasks, workspace, four_tasks, 4), true)
            matches &= length(by_four_tasks) == length(by_one_task)
            matches &= PauliSum(extractsum!(by_four_tasks)) == PauliSum(extractsum!(by_one_task))
            matches &= PauliSum(extractsum!(by_one_task)) != PauliSum(vpsum)
        end
        @test matches
    end
end

@testset "RotationLayer on a sum that several threads share out" begin
    # the magnetisation on a chain, which grows past the number of terms from which a sum is split into tasks
    nq = 100
    psum = PauliSum(nq)
    for qind in 1:nq
        add!(psum, :Z, qind, 1 / nq)
    end
    layers = repeat([RotationLayer(:X, 1:nq), RotationLayer([:Z, :Z], staircasetopology(nq))], 4)
    thetas = fill(pi / 4, length(layers))

    reference = propagate(layers, psum, thetas; max_weight=6.0, min_abs_coeff=1e-8)
    @test length(reference) > 2 * PB._MIN_ELEMS_PER_TASK

    for T in (VectorPauliSum, multivectorsum, multidictsum)
        layered = propagate(layers, T(psum), thetas; max_weight=6.0, min_abs_coeff=1e-8)
        @test length(layered) == length(reference)
        @test PauliSum(layered) == reference
    end
end

@testset "RotationLayer constructors and the order of its rotations" begin
    nq = 8
    x_layer = RotationLayer(:X, 1:nq)
    zz_layer = RotationLayer(PauliRotation, [:Z, :Z], staircasetopology(nq))

    @test x_layer isa ParametrizedGate
    @test countparameters([x_layer, zz_layer]) == 2

    # the rotations are applied in the order of `qinds`, and in reverse in the Heisenberg picture
    rotations, angles = PauliPropagation.torotations(RotationLayer([:Z, :Z], staircasetopology(4; periodic=true)), collect(1.0:4.0))
    @test [rotation.qinds for rotation in rotations] == [[1, 2], [2, 3], [3, 4], [4, 1]]
    @test angles == collect(1.0:4.0)
    rotations, angles = PauliPropagation.torotations(RotationLayer(:X, [3, 1, 1, 2]), [1.0, 2.0, 3.0, 4.0])
    @test [rotation.qinds for rotation in rotations] == [[3], [1], [1], [2]]
    @test angles == [1.0, 2.0, 3.0, 4.0]
    heisenberg_layer, heisenberg_angles = PauliPropagation._toheisenberg(RotationLayer(:X, [3, 1, 2]), [1.0, 2.0, 3.0])
    @test heisenberg_layer.qinds == [[2], [1], [3]]
    @test heisenberg_angles == [3.0, 2.0, 1.0]

    # a layer equals its rotations in the order of `qinds`, whatever that order, in either picture; on this input, applying
    # the bonds in another order truncates differently
    rng = MersenneTwister(3)
    psum = randompaulisum(rng, nq, 8, 3)
    bonds = shuffle(rng, staircasetopology(nq; periodic=true))
    thetas = randn(rng, nq)
    bond_rotations = [PauliRotation([:Z, :Z], bond) for bond in bonds]
    for T in (PauliSum, VectorPauliSum), heisenberg in (true, false)
        layered = propagate(RotationLayer([:Z, :Z], bonds), T(psum), thetas; heisenberg, max_weight=3.0, min_abs_coeff=1e-3)
        reference = propagate(bond_rotations, T(psum), thetas; heisenberg, max_weight=3.0, min_abs_coeff=1e-3)
        @test length(layered) == length(reference) && PauliSum(layered) == PauliSum(reference)
    end


    @test_throws ArgumentError RotationLayer([:X, :Z], staircasetopology(nq))
    @test_throws ArgumentError RotationLayer([:Z, :Z], [1, 2])
    @test_throws ArgumentError RotationLayer([:Z, :Z, :Z], [(1, 2, 3)])
    @test_throws ArgumentError RotationLayer(:I, 1:nq)
    @test_throws ArgumentError RotationLayer(:X, [0, 1])
    @test_throws ArgumentError PauliPropagation.torotations(zz_layer, [0.1, 0.2])
    @test_throws ArgumentError propagate(zz_layer, PauliString(nq - 1, :X, 1), 0.3)
end

@testset "RotationLayer truncates and merges what it is given" begin
    nq = 8
    layer = RotationLayer(:X, 1:nq)
    theta = 0.4
    max_weight = 2.0

    # a Pauli string above the weight limit, which the layer touches, and one it leaves alone
    heavy = PauliSum(nq)
    add!(heavy, [:Z, :Z, :Z], [1, 2, 3], 1.0)
    add!(heavy, [:X, :X, :X], [4, 5, 6], 1.0)
    add!(heavy, [:Y, :Z, :Y], [6, 7, 8], 1.0)
    add!(heavy, [:Z], [5], 1.0)
    for T in (PauliSum, VectorPauliSum)
        layered = propagate(layer, T(heavy), theta; max_weight, min_abs_coeff=0.0)
        @test !isempty(layered)
        @test all(pstr -> countweight(pstr) <= max_weight, paulis(layered))
    end

    # Pauli strings that come twice are merged where a rotation touches them
    twice = VectorPauliSum(nq, [paulis(VectorPauliSum(PauliString(nq, :Z, 2))); paulis(VectorPauliSum(PauliString(nq, :Z, 2)))], [0.5, 0.25])
    layered = propagate(layer, twice, theta; min_abs_coeff=0.0)
    reference = propagate(layer, PauliString(nq, :Z, 2, 0.75), theta; min_abs_coeff=0.0)
    @test PauliSum(layered) == reference
    @test length(layered) == 2

    # and so are Pauli strings that no rotation touches
    untouched_twice = VectorPauliSum(nq, [paulis(VectorPauliSum(PauliString(nq, :X, 3))); paulis(VectorPauliSum(PauliString(nq, :X, 3)))], [0.5, 0.25])
    layered = propagate(layer, untouched_twice, theta; min_abs_coeff=0.0)
    @test PauliSum(layered) == propagate(layer, PauliString(nq, :X, 3, 0.75), theta; min_abs_coeff=0.0)
    @test length(layered) == 1
end

@testset "RotationLayer applied without truncation" begin
    # `applytoall!` leaves the sum merged, as the rotations do once each of them is merged
    nq = 8
    rng = MersenneTwister(5)
    psum = randompaulisum(rng, nq, 6, 3)
    layer = RotationLayer([:Z, :Z], staircasetopology(nq; periodic=true))
    thetas = randn(rng, nq)
    rotations, angles = expandlayers([layer], [thetas])

    for T in (PauliSum, VectorPauliSum, psum -> MultiPauliSum(psum, 4))
        layered = PropagationCache(T(psum))
        reference = PropagationCache(T(psum))
        applytoall!(layer, layered, thetas)
        for (rotation, angle) in zip(rotations, angles)
            applytoall!(rotation, reference, angle)
            merge!(reference)
        end

        @test !PB.requiresmerging(layer, layered)
        @test length(layered) == length(reference) > length(psum)
        @test PauliSum(extractsum!(layered)) == PauliSum(extractsum!(reference))
    end
end

@testset "RotationLayer takes its rotations one by one where it has to" begin
    nq = 8
    rng = MersenneTwister(3)
    psum = randompaulisum(rng, nq, 6, 3)
    layers = [RotationLayer(:X, 1:nq), RotationLayer([:Z, :Z], staircasetopology(nq))]
    thetas = randomangles(rng, layers)
    rotations, angles = expandlayers(layers, thetas)

    # coefficients that track their frequencies, and a truncation relative to the largest coefficient
    @test propagate(layers, psum, thetas; max_freq=3.0, min_abs_coeff=1e-6) == propagate(rotations, psum, angles; max_freq=3.0, min_abs_coeff=1e-6)
    for T in (PauliSum, VectorPauliSum)
        @test propagate(layers, T(psum), thetas; min_rel_coeff=1e-3) == propagate(rotations, T(psum), angles; min_rel_coeff=1e-3)
    end
end
