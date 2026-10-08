using Random
using Test

# a gate without the fields that `qinds` reads
struct NoQindsGate <: StaticGate end

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
