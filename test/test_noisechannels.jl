using Test


function depolarizing(channel_circ, damping_circ, thetas1, thetas2, where_ind, q_ind, noise_p)
    # add to the circuit
    insert!(channel_circ, where_ind, DepolarizingNoise(q_ind))

    insert!(damping_circ, where_ind, PauliPropagation.PauliYDamping(q_ind))
    insert!(damping_circ, where_ind, PauliPropagation.PauliXDamping(q_ind))
    insert!(damping_circ, where_ind, PauliPropagation.PauliZDamping(q_ind))

    # add to the parameters
    insert!(thetas1, where_ind, noise_p)

    insert!(thetas2, where_ind, noise_p)
    insert!(thetas2, where_ind, noise_p)
    insert!(thetas2, where_ind, noise_p)

    return channel_circ, damping_circ, thetas1, thetas2
end

# this is also an alias for PauliZNoise
function dephasing(channel_circ, damping_circ, thetas1, thetas2, where_ind, q_ind, noise_p)
    # add to the circuit
    insert!(channel_circ, where_ind, DephasingNoise(q_ind))

    insert!(damping_circ, where_ind, PauliPropagation.PauliYDamping(q_ind))
    insert!(damping_circ, where_ind, PauliPropagation.PauliXDamping(q_ind))

    # add to the parameters
    insert!(thetas1, where_ind, noise_p)

    insert!(thetas2, where_ind, noise_p)
    insert!(thetas2, where_ind, noise_p)

    return channel_circ, damping_circ, thetas1, thetas2
end

function paulixnoise(channel_circ, damping_circ, thetas1, thetas2, where_ind, q_ind, noise_p)
    # add to the circuit
    insert!(channel_circ, where_ind, PauliXNoise(q_ind))

    insert!(damping_circ, where_ind, PauliPropagation.PauliYDamping(q_ind))
    insert!(damping_circ, where_ind, PauliPropagation.PauliZDamping(q_ind))

    # add to the parameters
    insert!(thetas1, where_ind, noise_p)

    insert!(thetas2, where_ind, noise_p)
    insert!(thetas2, where_ind, noise_p)

    return channel_circ, damping_circ, thetas1, thetas2
end

function pauliynoise(channel_circ, damping_circ, thetas1, thetas2, where_ind, q_ind, noise_p)
    # add to the circuit
    insert!(channel_circ, where_ind, PauliYNoise(q_ind))

    insert!(damping_circ, where_ind, PauliPropagation.PauliXDamping(q_ind))
    insert!(damping_circ, where_ind, PauliPropagation.PauliZDamping(q_ind))

    # add to the parameters
    insert!(thetas1, where_ind, noise_p)

    insert!(thetas2, where_ind, noise_p)
    insert!(thetas2, where_ind, noise_p)

    return channel_circ, damping_circ, thetas1, thetas2
end

@testset "Test Pauli Noises" begin

    builder_functions = [
        depolarizing,
        dephasing,
        paulixnoise,
        pauliynoise
    ]


    for builderfunc in builder_functions
        nq = rand(2:5)
        nl = 2
        W = Inf
        min_abs_coeff = 0.0

        pstr = PauliString(nq, rand([:X, :Y, :Z], nq), 1:nq)

        topo = bricklayertopology(nq; periodic=true)
        circ = hardwareefficientcircuit(nq, nl; topology=topo)

        m = countparameters(circ)

        channel_circ = deepcopy(circ)
        damping_circ = deepcopy(circ)

        thetas1 = randn(m)
        thetas2 = deepcopy(thetas1)

        where_ind = rand(1:m)
        q_ind = rand(1:nq)
        noise_p = rand() * 0.2

        channel_circ, damping_circ, thetas1, thetas2 = builderfunc(channel_circ, damping_circ, thetas1, thetas2, where_ind, q_ind, noise_p)



        dnum1 = propagate(channel_circ, pstr, thetas1; max_weight=W, min_abs_coeff=min_abs_coeff)

        dnum2 = propagate(damping_circ, pstr, thetas2; max_weight=W, min_abs_coeff=min_abs_coeff)

        @test overlapwithzero(dnum1) ≈ overlapwithzero(dnum2)
    end

end


@testset "Test AmplitudeDampingNoise values" begin
    # I -> I, X -> sqrt(1-gamma)*X, Y -> sqrt(1-gamma)*Y, Z -> (1-gamma)*Z + gamma*I
    nq = 1
    gate = AmplitudeDampingNoise(1)
    c = 0.37

    for gamma in (0.0, 0.3, 0.7, 1.0)
        out = propagate(gate, PauliString(nq, :I, 1, c), gamma; min_abs_coeff=0.0)
        @test length(out) == 1
        @test getcoeff(out, :I, 1) ≈ c

        for pauli in (:X, :Y)
            out = propagate(gate, PauliString(nq, pauli, 1, c), gamma; min_abs_coeff=0.0)
            @test length(out) == 1
            @test getcoeff(out, pauli, 1) ≈ c * sqrt(1 - gamma)
        end

        out = propagate(gate, PauliString(nq, :Z, 1, c), gamma; min_abs_coeff=0.0)
        @test length(out) == 2
        @test getcoeff(out, :Z, 1) ≈ c * (1 - gamma)
        @test getcoeff(out, :I, 1) ≈ c * gamma
    end

    # the noise channel must act locally, and leave other qubits untouched
    nq = 3
    gamma = 0.4
    gate = AmplitudeDampingNoise(2)

    pstr = PauliString(nq, [:X, :Z, :Y], [1, 2, 3], 0.5)
    out = propagate(gate, pstr, gamma; min_abs_coeff=0.0)

    @test length(out) == 2
    @test getcoeff(out, [:X, :Z, :Y], [1, 2, 3]) ≈ 0.5 * (1 - gamma)
    @test getcoeff(out, [:X, :I, :Y], [1, 2, 3]) ≈ 0.5 * gamma
end


@testset "Test PauliNoise values" begin
    # coeff *= (1-lambda) on damped Paulis, untouched on I and non-damped Paulis,
    # checked on both the Dict (PauliSum) and array (VectorPauliSum) backends
    nq = 1
    c = 0.37

    noise_damped_paulis = (
        (DepolarizingNoise, (:X, :Y, :Z)),
        (PauliXNoise, (:Y, :Z)),
        (PauliYNoise, (:X, :Z)),
        (PauliZNoise, (:X, :Y)),
        (PauliPropagation.PauliXDamping, (:X,)),
        (PauliPropagation.PauliYDamping, (:Y,)),
        (PauliPropagation.PauliZDamping, (:Z,)),
    )

    for (NoiseType, damped) in noise_damped_paulis
        gate = NoiseType(1)
        for lambda in (0.0, 0.25, 0.6, 1.0)
            for pauli in (:I, :X, :Y, :Z)
                expected = pauli in damped ? c * (1 - lambda) : c

                for PS in (PauliSum, VectorPauliSum)
                    out = propagate(gate, PS(PauliString(nq, pauli, 1, c)), lambda; min_abs_coeff=0.0)
                    @test length(out) == 1
                    @test getcoeff(out, pauli, 1) ≈ expected
                end
            end
        end
    end
end


@testset "Test noise strength validation" begin
    nq = 2
    pstr = PauliString(nq, :X, 1)

    noise_types = (
        DepolarizingNoise, DephasingNoise, PauliXNoise, PauliYNoise, PauliZNoise, AmplitudeDampingNoise,
        PauliPropagation.PauliXDamping, PauliPropagation.PauliYDamping, PauliPropagation.PauliZDamping,
    )

    for NoiseType in noise_types
        # constructing a frozen gate validates the noise strength immediately
        @test_throws ArgumentError NoiseType(1, -0.1)
        @test_throws ArgumentError NoiseType(1, 1.1)

        # boundary values 0.0 and 1.0 are valid
        @test NoiseType(1, 0.0) isa FrozenGate
        @test NoiseType(1, 1.0) isa FrozenGate

        # the same bounds are enforced at propagation time for the un-frozen gate
        gate = NoiseType(1)
        @test_throws ArgumentError propagate(gate, pstr, -0.1)
        @test_throws ArgumentError propagate(gate, pstr, 1.1)
        @test propagate(gate, pstr, 0.0) isa PauliSum
        @test propagate(gate, pstr, 1.0) isa PauliSum
    end
end


@testset "Test noise channel qind validation" begin
    nq = 3
    pstr = PauliString(nq, :X, 1)

    noise_types = (
        DepolarizingNoise, DephasingNoise, PauliXNoise, PauliYNoise, PauliZNoise, AmplitudeDampingNoise,
        PauliPropagation.PauliXDamping, PauliPropagation.PauliYDamping, PauliPropagation.PauliZDamping,
    )

    for NoiseType in noise_types
        # qind must be a positive integer at construction time
        @test_throws ArgumentError NoiseType(0)
        @test_throws ArgumentError NoiseType(-1)

        # qind is only checked against the number of qubits once propagated
        gate = NoiseType(nq + 1)
        @test_throws ArgumentError propagate(gate, pstr, 0.1)
    end
end