using Test
using Random

# Compares rewindgradient's expectation value and analytic gradient against a
# central-difference gradient computed via plain `propagate` on the same circuit.
# Uses min_abs_coeff=0.0 (exact, no truncation) since these circuits are small enough
# that the untruncated Pauli sum stays tiny, so any mismatch is due to the gradient
# and not truncation noise.
function _checkgradient(circuit, psum, params, overlapfunc; eps=1e-6, atol=1e-7)
    expec, grad = rewindgradient(circuit, deepcopy(psum), params, overlapfunc; min_abs_coeff=0.0)

    @test expec ≈ overlapfunc(propagate(circuit, deepcopy(psum), params; min_abs_coeff=0.0))
    @test length(grad) == length(params)
    # an all-zero gradient would match finite differences no matter what the code did
    @test any(!iszero, grad)

    fd_grad = zeros(length(params))
    for k in eachindex(params)
        plus_params = copy(params)
        plus_params[k] += eps
        minus_params = copy(params)
        minus_params[k] -= eps

        fplus = overlapfunc(propagate(circuit, deepcopy(psum), plus_params; min_abs_coeff=0.0))
        fminus = overlapfunc(propagate(circuit, deepcopy(psum), minus_params; min_abs_coeff=0.0))

        fd_grad[k] = (fplus - fminus) / (2eps)
    end

    @test grad ≈ fd_grad atol = atol
end


@testset "Test rewindgradient against finite differences" begin

    @testset "Pure PauliRotation circuit" begin
        nq = 4
        circuit = [
            PauliRotation(:X, 1),
            PauliRotation(:Y, 2),
            PauliRotation([:X, :Z], [1, 3]),
            PauliRotation(:Z, 4),
            PauliRotation([:Y, :Y], [2, 4]),
        ]
        params = [0.3, -0.7, 1.1, 0.42, -0.15]
        psum = VectorPauliSum(PauliString(nq, [:Z, :Z], [1, 3]))

        _checkgradient(circuit, psum, params, overlapwithzero)
    end

    @testset "Circuit with interspersed Clifford gates" begin
        nq = 4
        circuit = [
            PauliRotation(:X, 1),
            CliffordGate(:H, [2]),
            PauliRotation([:Y, :Z], [1, 3]),
            CliffordGate(:CNOT, [2, 3]),
            PauliRotation(:Z, 4),
            CliffordGate(:S, [4]),
            PauliRotation([:X, :X], [1, 4]),
        ]
        params = [0.2, -0.5, 0.83, -1.0]
        psum = VectorPauliSum(PauliString(nq, [:Z, :Z], [1, 4]))

        _checkgradient(circuit, psum, params, overlapwithzero)
    end

    @testset "Circuit with frozen parametrized gates" begin
        nq = 4
        circuit = [
            PauliRotation(:X, 1),
            PauliRotation(:Y, 2, 0.66),  # frozen, excluded from params/grad
            PauliRotation([:X, :Z], [1, 3]),
            PauliRotation(:Z, 4, -0.31),  # frozen
            PauliRotation([:Y, :Y], [2, 4]),
        ]
        params = [0.44, -0.9, 0.12]
        psum = VectorPauliSum(PauliString(nq, [:Z, :Z], [1, 3]))

        _checkgradient(circuit, psum, params, overlapwithzero)
    end

    @testset "Circuit with Clifford and frozen gates mixed together" begin
        nq = 5
        circuit = [
            PauliRotation(:X, 1),
            CliffordGate(:H, [2]),
            PauliRotation(:Y, 2, 0.5),  # frozen
            CliffordGate(:CZ, [2, 3]),
            PauliRotation([:X, :Z], [1, 3]),
            CliffordGate(:SX, [4]),
            PauliRotation(:Z, 4, -0.2),  # frozen
            PauliRotation([:Y, :X], [3, 5]),
            CliffordGate(:CNOT, [4, 5]),
            PauliRotation(:X, 5),
        ]
        params = [0.31, -0.62, 0.9, -0.44]
        psum = VectorPauliSum(PauliString(nq, [:Z], [3]))

        _checkgradient(circuit, psum, params, overlapwithzero)
    end

    @testset "Different overlap functions" begin
        nq = 4
        circuit = [
            PauliRotation(:X, 1),
            CliffordGate(:H, [2]),
            PauliRotation([:Y, :Z], [1, 3]),
            PauliRotation(:Z, 4, 0.27),  # frozen
            CliffordGate(:CNOT, [3, 4]),
            PauliRotation([:X, :X], [2, 4]),
        ]
        params = [0.18, -0.71, 0.53]
        psum = VectorPauliSum(PauliString(nq, [:Z, :Z], [1, 3]))

        for overlapfunc in (overlapwithzero, overlapwithplus)
            _checkgradient(circuit, psum, params, overlapfunc)
        end

        # unitary propagation conserves the trace, so the maximally mixed overlap is constant
        _, maxmixed_grad = rewindgradient(circuit, deepcopy(psum), params, overlapwithmaxmixed; min_abs_coeff=0.0)
        @test all(iszero, maxmixed_grad)

        # overlapwithcomputational needs the extra `onebitinds` argument, so wrap it
        # into the single-argument form that rewindgradient expects.
        onebitinds = [1, 3]
        computational_overlap(pobj) = overlapwithcomputational(pobj, onebitinds)
        _checkgradient(circuit, psum, params, computational_overlap)
    end

    @testset "Initial observable with multiple Pauli terms" begin
        nq = 4
        circuit = [
            PauliRotation(:X, 1),
            CliffordGate(:H, [2]),
            PauliRotation([:Y, :Z], [1, 3]),
            PauliRotation(:Z, 4, -0.4),  # frozen
            CliffordGate(:CNOT, [3, 4]),
            PauliRotation([:X, :X], [2, 4]),
        ]
        params = [0.37, -0.28, 0.61]

        obs = PauliSum(nq)
        add!(obs, [:Z, :Z], [1, 4], 0.6)
        add!(obs, [:X, :Y], [2, 3], -0.3)
        psum = VectorPauliSum(obs)

        _checkgradient(circuit, psum, params, overlapwithzero)
    end

    @testset "Equivalent results for PauliSum and PropagationCache inputs" begin
        nq = 4
        circuit = [
            PauliRotation(:X, 1),
            CliffordGate(:H, [2]),
            PauliRotation([:Y, :Z], [1, 3]),
            PauliRotation(:Z, 4, -0.31),  # frozen
            PauliRotation([:X, :X], [1, 4]),
        ]
        params = [0.44, -0.9, 0.12]

        dict_psum = PauliSum(nq)
        add!(dict_psum, [:Z, :X], [2, 3], 0.6)
        add!(dict_psum, [:X, :Y], [1, 4], -0.3)
        vec_psum = VectorPauliSum(dict_psum)

        expec_vec, grad_vec = rewindgradient(circuit, vec_psum, params, overlapwithzero; min_abs_coeff=0.0)
        expec_dict, grad_dict = rewindgradient(circuit, dict_psum, params, overlapwithzero; min_abs_coeff=0.0)
        cache = PropagationCache(deepcopy(vec_psum))
        expec_cache, grad_cache = rewindgradient!(circuit, cache, params, overlapwithzero; min_abs_coeff=0.0)

        @test expec_dict ≈ expec_vec
        @test grad_dict ≈ grad_vec
        @test expec_cache ≈ expec_vec
        @test grad_cache ≈ grad_vec

        # the non-mutating entry point must leave the caller's Pauli sums untouched
        @test vec_psum == VectorPauliSum(dict_psum)
    end

    @testset "Noise channels are rejected" begin
        nq = 3
        params = [0.3, 0.4]
        psum = VectorPauliSum(PauliString(nq, [:Z, :Z], [1, 2]))

        for noise in (DepolarizingNoise(2), freeze(DepolarizingNoise(2), 0.1))
            circuit = Gate[PauliRotation(:X, 1), noise, PauliRotation(:Y, 2)]
            @test_throws AssertionError rewindgradient(circuit, psum, params, overlapwithzero)
        end
    end

    @testset "Random small circuits" begin
        Random.seed!(33)

        nq = 5
        one_qubit_symbols = (:X, :Y, :Z)
        clifford_symbols_1q = (:H, :S, :SX)

        for trial in 1:6
            circuit = Gate[]
            for _ in 1:8
                if rand() < 0.3
                    # single-qubit Clifford gate
                    push!(circuit, CliffordGate(rand(clifford_symbols_1q), [rand(1:nq)]))
                elseif rand() < 0.25
                    # two-qubit Clifford gate
                    q1, q2 = randperm(nq)[1:2]
                    push!(circuit, CliffordGate(rand((:CNOT, :CZ)), [q1, q2]))
                elseif rand() < 0.3
                    # frozen (fixed-parameter) Pauli rotation, excluded from params/grad
                    if rand() < 0.5
                        push!(circuit, PauliRotation(rand(one_qubit_symbols), rand(1:nq), randn()))
                    else
                        q1, q2 = randperm(nq)[1:2]
                        push!(circuit, PauliRotation(rand(one_qubit_symbols, 2), [q1, q2], randn()))
                    end
                else
                    # ordinary (differentiated) Pauli rotation
                    if rand() < 0.5
                        push!(circuit, PauliRotation(rand(one_qubit_symbols), rand(1:nq)))
                    else
                        q1, q2 = randperm(nq)[1:2]
                        push!(circuit, PauliRotation(rand(one_qubit_symbols, 2), [q1, q2]))
                    end
                end
            end

            n_params = countparameters(circuit)
            params = randn(n_params)
            pstr = PauliString(nq, [:Z, :Z], randperm(nq)[1:2])
            psum = VectorPauliSum(pstr)

            _checkgradient(circuit, psum, params, overlapwithzero)
        end
    end

end
