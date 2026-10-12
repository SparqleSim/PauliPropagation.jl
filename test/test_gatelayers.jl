using Random
using Test

# a gate without the fields that `qinds` reads
struct NoQindsGate <: StaticGate end

# A gate defined by its action alone, which conjugates with Z on its qubit like `CliffordGate(:Z, qind)`, and two gates that
# a layer cannot apply in one visit: one branches every Pauli string into itself and X on its qubit, the other negates the
# identity along with every other Pauli string. All three say that they never create duplicate Pauli strings.
struct ZConjugation <: StaticGate
    qind::Int
end
PB.apply(gate::ZConjugation, pstr, coeff; kwargs...) = ((pstr, getpauli(pstr, gate.qind) in (1, 2) ? -coeff : coeff),)
struct BranchingGate <: StaticGate
    qind::Int
end
PB.apply(gate::BranchingGate, pstr, coeff; kwargs...) = ((pstr, coeff / 2), (setpauli(pstr, :X, gate.qind), coeff / 2))
struct NegatingGate <: StaticGate
    qind::Int
end
PB.apply(::NegatingGate, pstr, coeff; kwargs...) = ((pstr, -coeff),)
PB.requiresmerging(::Union{ZConjugation,BranchingGate,NegatingGate}, ::AbstractPauliPropagationCache) = false

# A gate defined by its action alone that branches every Pauli other than the identity on its qubit into X, Y and Z, by
# the columns of `rotation`, as a rotation of the Bloch sphere would.
struct AxisRotation <: StaticGate
    qind::Int
    rotation::NTuple{9,Float64}
end
function PB.apply(gate::AxisRotation, pstr, coeff; kwargs...)
    pauli = getpauli(pstr, gate.qind)
    if pauli == 0
        return ((pstr, coeff),)
    end
    column = 3 * (pauli - 1)
    return ((setpauli(pstr, :X, gate.qind), coeff * gate.rotation[column+1]),
        (setpauli(pstr, :Y, gate.qind), coeff * gate.rotation[column+2]),
        (setpauli(pstr, :Z, gate.qind), coeff * gate.rotation[column+3]))
end

# the gates of the layers one after the other, as a circuit, with the parameters of the parametrized ones
function flattenlayers(layers, params)
    gates = Gate[]
    flat_params = Float64[]
    for (layer, layer_params) in zip(layers, params)
        param_index = 0
        for gate in layer.gates
            push!(gates, gate)
            if gate isa ParametrizedGate
                param_index += 1
                push!(flat_params, layer_params[param_index])
            end
        end
    end
    return gates, flat_params
end

# every kind of sum in the Heisenberg picture, and the two that the others are made of in the Schrödinger picture as well
const GATELAYER_TEST_SUMS = ((PauliSum, true), (VectorPauliSum, true), (psum -> MultiPauliSum(VectorPauliSum(psum), 4), true),
    (psum -> MultiPauliSum(psum, 4), true), (PauliSum, false), (VectorPauliSum, false))

function matchesgatebygate(layers, params, psum; kwargs...)
    gates, flat_params = flattenlayers(layers, params)
    matches = true
    for (T, heisenberg) in GATELAYER_TEST_SUMS
        layered = propagate(layers, T(psum), params; heisenberg, kwargs...)
        reference = propagate(gates, T(psum), flat_params; heisenberg, kwargs...)
        matches &= !isempty(layered) && length(layered) == length(reference) && PauliSum(layered) == PauliSum(reference)
    end
    return matches
end

# Every Pauli on the qubits 1 to 4, next to a few fixed strings on the others. Pauli strings of different classes then
# agree outside the qubits that the rotations act on, which is where a class could take in strings that are not its own.
# No coefficient is below the truncations of the tests, which rotations one by one leave on strings they do not touch.
function densepaulisum(rng, nq)
    psum = PauliSum(nq)
    rests = ((Symbol[], Int[]), ([:Z], [5]), ([:X, :Y], [6, nq]))
    for code in 0:255, (rest_symbols, rest_qinds) in rests
        local_symbols = [(:I, :X, :Y, :Z)[((code>>(2*(qind-1)))&3)+1] for qind in 1:4]
        add!(psum, vcat(local_symbols, rest_symbols), vcat(1:4, rest_qinds), rand(rng, (-1, 1)) * (0.1 + rand(rng)))
    end
    return psum
end

@testset "GateLayer propagates like its gates one by one" begin
    nq = 8
    rng = MersenneTwister(nq)
    psum = densepaulisum(rng, nq)

    # a generator of its own for every rotation, a rotation with the identity in its generator, and a frozen rotation
    disjoint = GateLayer([PauliRotation(:X, 1), PauliRotation([:Z, :Z], [2, 3]), PauliRotation([:Y, :X], [5, 4]),
        PauliRotation([:I, :Y], [6, 7]), PauliRotation(:Y, nq, 0.3)])
    # rotations that share qubits and commute, some of them acting on a qubit with two different Paulis
    shared = GateLayer([PauliRotation([:X, :X], [1, 2]), PauliRotation([:Z, :Z], [1, 2]), PauliRotation([:X, :Z], [3, 4]),
            PauliRotation(:Y, 5), PauliRotation([:Y, :Y], [5, 6]), PauliRotation([:Z, :X], [3, 4]), PauliRotation(:X, 7, -0.4)];
        guaranteed_commutes=true)
    # a Clifford gate, for which the gates are applied one by one, and a rotation on three qubits
    with_clifford = GateLayer([PauliRotation(:X, 1), CliffordGate(:H, [2]), PauliRotation([:Z, :Z], [3, 4]), CliffordGate(:CNOT, [5, 6])])
    wide = GateLayer([PauliRotation([:Z, :Z, :Z], [1, 2, 3]), PauliRotation(:X, 4)])

    cache = PropagationCache(VectorPauliSum(psum))
    @test PP._haslayerfastpath(disjoint, cache) && PP._haslayerfastpath(shared, cache) && PP._haslayerfastpath(wide, cache)
    @test !PP._haslayerfastpath(with_clifford, cache)

    # every layer without truncation, and circuits of them truncated with the inputs within the weight limit, which
    # rotations one by one leave on strings they do not touch
    matches = true
    for layer in (disjoint, shared, with_clifford, wide)
        matches &= matchesgatebygate([layer], [randn(rng, countparameters(layer))], psum; min_abs_coeff=0.0)
    end
    for layers in ([disjoint, shared, disjoint], [shared, with_clifford, wide])
        params = [randn(rng, countparameters(layer)) for layer in layers]
        matches &= matchesgatebygate(layers, params, psum; max_weight=6.0, min_abs_coeff=1e-2)
    end
    @test matches
end

@testset "GateLayer construction and parameters" begin
    @test PB.qinds(PauliRotation([:X, :Z], [2, 5])) == [2, 5]
    @test PB.qinds(DepolarizingNoise(3)) == 3
    @test PB.qinds(PauliRotation(:Y, 4, 0.1)) == [4]
    @test_throws ArgumentError PB.qinds(NoQindsGate())

    layer = GateLayer([PauliRotation(:X, 1), CliffordGate(:H, [2]), PauliRotation(:Z, 3, 0.2), PauliRotation([:Z, :Z], [4, 5])])
    @test layer isa ParametrizedGate
    @test countparameters(layer) == 2
    @test countparameters([layer, PauliRotation(:X, 1)]) == 2
    @test PB.qinds(layer) == [1, 2, 3, 4, 5]
    @test PP.maxqind([DepolarizingNoise(7), layer, PauliRotation(:X, 1)]) == 7
    @test PP.maxqind(Gate[]) == 0

    # gates on a shared qubit only where they are guaranteed to commute
    @test_throws ArgumentError GateLayer([PauliRotation(:X, 1), PauliRotation([:X, :X], [1, 2])])
    @test GateLayer([PauliRotation(:X, 1), PauliRotation([:X, :X], [1, 2])]; guaranteed_commutes=true) isa GateLayer
    @test_throws ArgumentError PauliRotationLayer([:X, :Z], [(1, 2), (2, 3)])

    # one parameter per parametrized gate, or a number for one parametrized gate
    psum = PauliSum(PauliString(5, :Z, 1))
    @test_throws ArgumentError propagate(layer, psum, 0.1)
    @test_throws ArgumentError propagate(layer, psum, [0.1, 0.2, 0.3])
    single = GateLayer([PauliRotation(:Y, 1), CliffordGate(:H, [2])])
    @test propagate(single, psum, 0.3) == propagate(single, psum, [0.3])

    # each gate in the other picture, and in reverse order in the Heisenberg picture
    schrodinger_layer, schrodinger_params = toschrodinger(layer, [0.1, 0.2])
    @test schrodinger_params == [-0.1, -0.2]
    @test schrodinger_layer.gates[2].symbol == :H_transpose
    @test schrodinger_layer.gates[3].parameter == -0.2
    heisenberg_layer, heisenberg_params = toheisenberg(layer, [0.1, 0.2])
    @test [PB.qinds(gate) for gate in heisenberg_layer.gates] == [[4, 5], [3], [2], [1]]
    @test heisenberg_params == [0.2, 0.1]
    @test toschrodinger(single, 0.3)[2] == -0.3
    @test toheisenberg(single, 0.3)[2] == 0.3
end

@testset "GateLayer prints the types of its gates" begin
    mixed = GateLayer([PauliRotation(:X, 1), PauliRotation([:Z, :Y], [2, 3]), PauliRotation(:Y, 4, 0.2), CliffordGate(:CNOT, [5, 6]), PauliRotation(:X, 7)])
    @test repr(mixed) == "GateLayer(3 PauliRotation, 1 FrozenGate, 1 CliffordGate)"
    @test repr(GateLayer([PauliXNoise(qind) for qind in 1:3])) == "GateLayer(3 PauliXNoise)"

    # the types after the first four are only counted
    many = GateLayer([PauliRotation(:X, 1), PauliRotation(:Y, 2, 0.1), CliffordGate(:H, [3]), PauliXNoise(4), PauliZNoise(5), DepolarizingNoise(6)])
    @test repr(many) == "GateLayer(1 PauliRotation, 1 FrozenGate, 1 CliffordGate, 1 PauliXNoise, 2 more)"

    # a frozen layer rounds every angle
    @test repr(freeze(PauliRotationLayer(:Z, 1:2), [0.1234, 0.5])) == "FrozenGate(GateLayer(2 PauliRotation), parameter = [0.123, 0.5])"
    @test repr(PauliRotation(:X, 1, 0.1234)) == "FrozenGate(PauliRotation([:X], [1]), parameter = 0.123)"
end

@testset "commutes decides whether two gates may share a layer" begin
    # gates on different qubits commute, and so do Pauli rotations with commuting generators
    @test commutes(PauliRotation(:X, 1), CliffordGate(:H, 2))
    @test commutes(DepolarizingNoise(1), PauliRotation(:Z, 2))
    @test commutes(PauliRotation([:X, :X], [1, 2]), PauliRotation([:Z, :Z], [1, 2]))
    @test commutes(PauliRotation([:I, :Y], [1, 2]), PauliRotation(:Z, 1))
    @test !commutes(PauliRotation(:X, 1), PauliRotation(:Z, 1))
    @test !commutes(PauliRotation([:X, :X], [1, 2]), PauliRotation([:Z, :Z], [2, 3]))
    # Clifford gates sharing a qubit are compared on every Pauli string of their qubits
    @test commutes(CliffordGate(:CNOT, [1, 2]), CliffordGate(:CNOT, [1, 3]))
    @test commutes(CliffordGate(:ZZpihalf, [1, 2]), CliffordGate(:ZZpihalf, [2, 3]))
    @test !commutes(CliffordGate(:CNOT, [1, 2]), CliffordGate(:CNOT, [2, 3]))
    @test !commutes(CliffordGate(:H, 1), CliffordGate(:S, 1))
    # other gates sharing a qubit are not known to commute, and a frozen gate commutes like its gate
    @test !commutes(CliffordGate(:H, 1), PauliRotation(:X, 1))
    @test !commutes(DepolarizingNoise(1), CliffordGate(:X, 1))
    @test commutes(PauliRotation(:X, 1, 0.3), PauliRotation(:X, 1))
    @test !commutes(PauliRotation(:Z, 1, 0.3), PauliRotation(:X, 1, 0.2))
end

@testset "GateLayer of Clifford gates and noise propagates like its gates one by one" begin
    nq = 8
    rng = MersenneTwister(nq + 1)
    psum = densepaulisum(rng, nq)

    # Clifford gates on different qubits, ZZ entanglers that share qubits and commute, Pauli noise frozen or not, and a mix
    cliffords = GateLayer([CliffordGate(:H, 1), CliffordGate(:CNOT, [2, 3]), CliffordGate(:S, 4), CliffordGate(:SWAP, [5, 6]), CliffordGate(:Y, nq)])
    entanglers = GateLayer([CliffordGate(:ZZpihalf, [1, 2]), CliffordGate(:ZZpihalf, [2, 3]), CliffordGate(:ZZpihalf, [3, 4])]; guaranteed_commutes=true)
    noise = GateLayer([DepolarizingNoise(1), PauliXNoise(2, 0.1), PauliZNoise(3), DepolarizingNoise(4, 0.2)])
    mixed = GateLayer([CliffordGate(:H, 1), DepolarizingNoise(2, 0.1), CliffordGate(:CNOT, [3, 4]), PauliYNoise(5)])
    cache = PropagationCache(psum)
    @test all(layer -> PP._hasmappath(layer, cache), (cliffords, entanglers, noise, mixed))
    @test !PP._hasmappath(GateLayer([CliffordGate(:H, 1), PauliRotation(:X, 2)]), cache)
    @test PB.isparametrized(noise) && PB.isparametrized(mixed) && !PB.isparametrized(cliffords) && !PB.isparametrized(entanglers)

    # the layers and their gates one after the other, with the parameters of the parametrized gates
    layers = [cliffords, noise, entanglers, mixed]
    params = [[0.05, 0.3], [0.2]]
    gates = reduce(vcat, [layer.gates for layer in layers])
    flat_params = [0.05, 0.3, 0.2]
    @test countparameters(layers) == 2 && countparameters(gates) == 3

    function layersmatchgates(layers, params, gates, flat_params, sums; kwargs...)
        matches = true
        for (T, heisenberg) in sums
            layered = propagate(layers, T(psum), params; heisenberg, kwargs...)
            reference = propagate(gates, T(psum), flat_params; heisenberg, kwargs...)
            matches &= !isempty(layered) && length(layered) == length(reference) && PauliSum(layered) == PauliSum(reference)
        end
        return matches
    end
    @test layersmatchgates(layers, params, gates, flat_params, GATELAYER_TEST_SUMS; min_abs_coeff=0.0)
    # with the inputs within the truncations, which the gates one after the other would otherwise truncate at the first gate
    @test layersmatchgates(layers, params, gates, flat_params, VECTOR_TEST_SUMS; max_weight=6.0, min_abs_coeff=0.08)
    # layers of Clifford gates without a weight limit truncate nothing
    @test layersmatchgates([cliffords, entanglers], [], vcat(cliffords.gates, entanglers.gates), Float64[], VECTOR_TEST_SUMS; min_abs_coeff=1e-2)
end

@testset "the plan of a layer applied in one visit is checked before its entries are read" begin
    cache = PropagationCache(VectorPauliSum(PauliString(4, :Z, 1)))
    layer = GateLayer([CliffordGate(:H, 1), CliffordGate(:CNOT, [2, 3])])
    plan = PP._preparemap(layer, Float64[], cache)
    @test PP._applyactinggates(plan, nothing) isa Function

    # the last gate misses an entry
    pop!(plan.factors)
    @test_throws ArgumentError PP._applyactinggates(plan, nothing)

    # a third gate acts on the fourth qubit, but the plan holds two
    plan = PP._preparemap(layer, Float64[], cache)
    plan.qubit_gates[4] = (UInt64(1) << 2,)
    @test_throws ArgumentError PP._applyactinggates(plan, nothing)
end

@testset "a gate defined by its action alone joins a layer applied in one visit" begin
    nq = 8
    rng = MersenneTwister(nq + 2)
    psum = densepaulisum(rng, nq)

    # the transfer map of the gate is built from `apply`, and equals that of the Clifford gate it imitates
    @test PP._localtransfermap(ZConjugation(2)) == PP._localtransfermap(CliffordGate(:Z, 2))
    layer = GateLayer([ZConjugation(1), CliffordGate(:H, 2), ZConjugation(3), DepolarizingNoise(4, 0.1)])
    reference = [CliffordGate(:Z, 1), CliffordGate(:H, 2), CliffordGate(:Z, 3), DepolarizingNoise(4, 0.1)]
    cache = PropagationCache(psum)
    @test PP._hasmappath(layer, cache) && !PP._hasmappath(GateLayer([ZConjugation(1), PauliRotation(:X, 2)]), cache)
    for T in (PauliSum, VectorPauliSum)
        @test PauliSum(propagate([layer], T(psum))) == PauliSum(propagate(reference, T(psum)))
    end

    # a gate that branches or changes the identity on its qubits has no lookup
    @test_throws ArgumentError PP._locallookup(BranchingGate(1), UInt16)
    @test_throws ArgumentError PP._locallookup(NegatingGate(1), UInt16)
end

@testset "a layer of branching gates on different qubits propagates like its gates one by one" begin
    nq = 8
    rng = MersenneTwister(nq + 3)
    psum = densepaulisum(rng, nq)

    # a rotation of the Bloch sphere by 0.3 about the axis (1, 2, 2) / 3, column by column
    rotation = (0.9601, 0.2036, -0.1912, -0.1688, 0.9677, 0.1874, 0.2232, -0.1487, 0.9632)
    # transfer maps of rotations on one and two qubits, and a map that turns the identity partly into Z
    one_qubit_map = totransfermap(1, [PauliRotation(:X, 1), PauliRotation(:Z, 1)], [0.3, 0.4])
    two_qubit_map = totransfermap(2, [PauliRotation([:X, :Y], [1, 2]), PauliRotation(:Z, 1)], [0.2, 0.5])
    moving_identity = [[(0x00, 1.0), (0x03, 0.2)], [(0x01, 0.9)], [(0x02, 0.9)], [(0x03, 0.8)]]

    custom = GateLayer([AxisRotation(qind, rotation) for qind in 1:4])
    transfer_maps = GateLayer([TransferMapGate(one_qubit_map, [1]), TransferMapGate(one_qubit_map, [3]), TransferMapGate(two_qubit_map, [5, 6])])
    # the gate on qubit 7 acts on the Pauli strings that are the identity there, nearly all of them
    mixed = GateLayer([AxisRotation(1, rotation), CliffordGate(:H, 2), AmplitudeDampingNoise(3), PauliRotation(:Y, 4),
        DepolarizingNoise(5), TransferMapGate(moving_identity, [7])])
    cache = PropagationCache(psum)
    @test all(layer -> PP._hasclasspath(layer, cache), (custom, transfer_maps, mixed))

    # gates sharing a qubit, and gates that normalize the whole sum, apply one after the other
    @test !PP._hasclasspath(GateLayer([AxisRotation(1, rotation), AxisRotation(1, rotation)]; guaranteed_commutes=true), cache)
    @test !PP._hasclasspath(GateLayer([ImaginaryPauliRotation(:Z, 1), ImaginaryPauliRotation(:X, 2)]), cache)

    # The layers and their gates one after the other agree up to the order in which coefficients are added. Transfer map
    # gates have no Schrödinger picture, so the sums are those of the Heisenberg picture.
    function agreeswithgates(layers, params, gates, flat_params; kwargs...)
        agrees = true
        for (T, heisenberg) in GATELAYER_TEST_SUMS
            if !heisenberg
                continue
            end
            layered = PauliSum(propagate(layers, T(psum), params; heisenberg, kwargs...))
            reference = PauliSum(propagate(gates, T(psum), flat_params; heisenberg, kwargs...))
            agrees &= !isempty(layered) && length(layered) == length(reference) &&
                      all(isapprox(getcoeff(layered, pstr), coeff; atol=1e-14) for (pstr, coeff) in reference)
        end
        return agrees
    end
    layers = [custom, transfer_maps, mixed]
    params = [[0.1, 0.3, 0.05]]
    gates = reduce(vcat, [layer.gates for layer in layers])
    flat_params = [0.1, 0.3, 0.05]
    @test agreeswithgates(layers, params, gates, flat_params; min_abs_coeff=0.0)
    # with the inputs within the truncations, which the gates one after the other would otherwise truncate at the first gate
    @test agreeswithgates(layers, params, gates, flat_params; max_weight=6.0, min_abs_coeff=0.08)
end

@testset "tolayers gathers commuting gates of the same kind, and a layer without parametrized gates takes none" begin
    nq = 6
    circuit = Gate[
        # gates on different qubits, of which the Clifford gate and the noise share a layer and the rotations stay alone
        PauliRotation(:X, 1), CliffordGate(:H, 2), DepolarizingNoise(3, 0.1), PauliRotation(:Z, 4),
        # ZZ entanglers that share qubits and commute
        CliffordGate(:ZZpihalf, [1, 2]), CliffordGate(:ZZpihalf, [2, 3]), CliffordGate(:ZZpihalf, [3, 4]),
        # noise on three qubits, then a rotation on one of them
        DepolarizingNoise(1), DepolarizingNoise(2), DepolarizingNoise(3), PauliRotation(:X, 3),
        # a layer stays as it is, and CNOTs sharing their control commute
        PauliRotationLayer(:Y, 1:2), CliffordGate(:CNOT, [5, 6]), CliffordGate(:CNOT, [5, 4]),
    ]
    thetas = Any[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, [0.7, 0.8]]
    layered_circuit, layered_thetas = tolayers(circuit, thetas)
    @test repr.(layered_circuit) == ["PauliRotation([:X], [1])", "GateLayer(1 CliffordGate, 1 FrozenGate)", "PauliRotation([:Z], [4])",
        "GateLayer(3 CliffordGate)", "GateLayer(3 DepolarizingNoise)", "PauliRotation([:X], [3])", "GateLayer(2 PauliRotation)", "GateLayer(2 CliffordGate)"]
    @test layered_thetas == [0.1, 0.2, [0.3, 0.4, 0.5], 0.6, [0.7, 0.8]]
    @test PB.isparametrized.(layered_circuit) == [true, false, true, false, true, true, true, false]
    @test countparameters(layered_circuit) == 5 && countparameters(circuit) == 7

    rng = MersenneTwister(nq)
    psum = randompaulisum(rng, nq, 8, 3)
    @test layeredmatchescircuit(circuit, thetas, psum; min_abs_coeff=0.0)
    @test layeredmatchescircuit(circuit, thetas, psum; max_weight=3.0, min_abs_coeff=1e-3)

    # a layer without parametrized gates takes no parameter wherever a circuit is walked
    entanglers = layered_circuit[4]
    @test propagate([entanglers], psum) == propagate(entanglers.gates, psum)
    @test [gate.qinds for gate in only(toheisenberg([entanglers])).gates] == [[3, 4], [2, 3], [1, 2]]
    @test freeze([entanglers, PauliRotation(:X, 1)], [0.3])[1] === entanglers
    value, gradient = rewindgradient([PauliRotation(:X, 1), entanglers, PauliRotation(:Z, 2)], psum, [0.3, 0.4], overlapwithzero; min_abs_coeff=0.0)
    reference_value, reference_gradient = rewindgradient([PauliRotation(:X, 1), entanglers.gates..., PauliRotation(:Z, 2)], psum, [0.3, 0.4], overlapwithzero; min_abs_coeff=0.0)
    @test value ≈ reference_value && gradient ≈ reference_gradient
end
