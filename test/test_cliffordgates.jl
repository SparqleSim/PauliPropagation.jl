using Test

@testset "Test create Clifford gates" begin
    """Test gate map function from a user-defined gate."""
    # CNOT
    CNOT_relations = Dict(
        (:I, :I) => (:I, :I, 1),
        (:I, :X) => (:I, :X, 1),
        (:I, :Y) => (:Z, :Y, 1),
        (:I, :Z) => (:Z, :Z, 1),
        (:X, :I) => (:X, :X, 1),
        (:X, :X) => (:X, :I, 1),
        (:X, :Y) => (:Y, :Z, 1),
        (:X, :Z) => (:Y, :Y, -1),
        (:Y, :I) => (:Y, :X, 1),
        (:Y, :X) => (:Y, :I, 1),
        (:Y, :Y) => (:X, :Z, -1),
        (:Y, :Z) => (:X, :Y, 1),
        (:Z, :I) => (:Z, :I, 1),
        (:Z, :X) => (:Z, :X, 1),
        (:Z, :Y) => (:I, :Y, 1),
        (:Z, :Z) => (:I, :Z, 1),
    )
    mapped_CNOT = createcliffordmap(CNOT_relations)
    @test mapped_CNOT == clifford_map[:CNOT]

    clifford_map[:CNOT2] = mapped_CNOT
    reset_clifford_map!()
    @test clifford_map == PauliPropagation._default_clifford_map

    # H
    H_relations = Dict(
        (:I,) => (:I, 1),
        (:X,) => (:Z, 1),
        (:Y,) => (:Y, -1),
        (:Z,) => (:X, 1),
    )
    mapped_H = createcliffordmap(H_relations)
    @test mapped_H == clifford_map[:H]

end


@testset "Test transposing Clifford maps" begin
    for (gate, map) in clifford_map
        transposed_map = transposecliffordmap(map)
        @test length(transposed_map) == length(map)
        @test transposecliffordmap(transposed_map) == map
    end
end


@testset "Test Clifford gate construction errors" begin
    # unknown symbol
    @test_throws ArgumentError CliffordGate(:NotAGate, [1])

    # qinds must be positive and unique (checked before the clifford_map lookup)
    @test_throws ArgumentError CliffordGate(:H, [0])
    @test_throws ArgumentError CliffordGate(:H, [-1])
    @test_throws ArgumentError CliffordGate(:CNOT, [1, 1])

    # dimension mismatch between qinds and the clifford_map entry
    @test_throws ArgumentError CliffordGate(:H, [1, 2])   # :H acts on 1 qubit, 2 qinds given
    @test_throws ArgumentError CliffordGate(:CNOT, [1])   # :CNOT acts on 2 qubits, 1 qind given
end


@testset "Test composing Clifford maps errors" begin
    # circuit must contain only CliffordGates
    circuit = [CliffordGate(:H, [1]), PauliRotation(:X, 1)]
    @test_throws ArgumentError composecliffordmaps(circuit)

    # more than 4 qubits is unsupported due to the UInt8 restriction
    circuit = [CliffordGate(:H, [5])]
    @test_throws ArgumentError composecliffordmaps(circuit)
end


@testset "Test composing Clifford maps" begin
    circuit = [CliffordGate(:H, [2]), CliffordGate(:CNOT, [1, 2]), CliffordGate(:H, [2])]
    @test composecliffordmaps(circuit) == clifford_map[:CZ]

    circuit = [CliffordGate(:H, [2]), CliffordGate(:CZ, [1, 2]), CliffordGate(:H, [2])]
    @test composecliffordmaps(circuit) == clifford_map[:CNOT]

    circuit = [CliffordGate(:H, [1]), CliffordGate(:Z, [1]), CliffordGate(:H, [1])]
    @test composecliffordmaps(circuit) == clifford_map[:X]

    circuit = [CliffordGate(:SX, [1]), CliffordGate(:SX, [1])]
    @test composecliffordmaps(circuit) == clifford_map[:X]

    circuit = [CliffordGate(:S, [1]), CliffordGate(:S, [1])]
    @test composecliffordmaps(circuit) == clifford_map[:Z]

end


@testset "Test Clifford gates on every term type" begin
    # every map on qubits in both orders and on both sides of a 64-bit word, against the lookup of each term
    clifford_map[:CNOTtransposed] = transposecliffordmap(clifford_map[:CNOT])

    # the image of a term under a lookup map, read and written one qubit at a time
    function lookupimage(lookup_map, qinds, term, coeff)
        new_paulis, sign = lookup_map[getpauli(term, qinds)+1]
        return setpauli(term, new_paulis, qinds), coeff * sign
    end

    rng = MersenneTwister(42)
    for (TT, nq) in ((UInt16, 8), (UInt64, 32), (UInt128, 64), (NTupleInteger{4}, 100))
        terms = unique([symboltoint(TT, rand(rng, (:I, :X, :Y, :Z), nq), 1:nq) for _ in 1:200])
        coeffs = 0.5 .+ rand(rng, length(terms))
        onequbit = [qinds for qinds in [(1,), (nq,), (33,)] if maximum(qinds) <= nq]
        twoqubits = [qinds for qinds in [(1, 2), (2, 1), (nq, 1), (32, 33), (33, 32), (64, 65)] if maximum(qinds) <= nq]

        vector_matches = true
        dict_matches = true
        for (symbol, lookup_map) in clifford_map
            for qinds in (length(lookup_map) == 4 ? onequbit : twoqubits)
                gate = CliffordGate(symbol, collect(qinds))
                expected = [lookupimage(lookup_map, qinds, term, coeff) for (term, coeff) in zip(terms, coeffs)]

                vpsum = propagate(gate, VectorPauliSum(nq, copy(terms), copy(coeffs)))
                vector_matches &= collect(zip(paulis(vpsum), coefficients(vpsum))) == expected

                psum = propagate(gate, PauliSum(nq, Dict(zip(terms, coeffs))))
                dict_matches &= length(psum) == length(expected) && all(getcoeff(psum, term) == coeff for (term, coeff) in expected)
            end
        end
        @test vector_matches
        @test dict_matches
    end

    # a gate on more qubits acts as the circuit its map is composed of
    circuit = [CliffordGate(:CNOT, [1, 2]), CliffordGate(:H, [3]), CliffordGate(:CZ, [2, 3])]
    clifford_map[:composed] = composecliffordmaps(circuit)
    qinds = [5, 2, 9]
    terms = unique([symboltoint(UInt64, rand(rng, (:I, :X, :Y, :Z), 10), 1:10) for _ in 1:200])
    psum = PauliSum(10, Dict(zip(terms, 0.5 .+ rand(rng, length(terms)))))
    @test propagate(CliffordGate(:composed, qinds), psum) == propagate([CliffordGate(gate.symbol, qinds[gate.qinds]) for gate in circuit], psum)

    # the truncation after a gate reads the terms the gate made, so a weight cap counts their new weight
    images = [lookupimage(clifford_map[:CNOT], (1, 2), term, coeff) for (term, coeff) in psum]
    @test propagate(CliffordGate(:CNOT, [1, 2]), psum; max_weight=7.0) == PauliSum(10, Dict(image for image in images if countweight(first(image)) <= 7))

    # the gate only flips signs, so without a weight cap or a custom truncation it truncates nothing
    small_image = lookupimage(clifford_map[:CNOT], (1, 2), first(terms), 1e-12)
    @test propagate(CliffordGate(:CNOT, [1, 2]), PauliSum(10, Dict(first(terms) => 1e-12))) == PauliSum(10, Dict([small_image]))

    reset_clifford_map!()
end