using Test
using LinearAlgebra
using Random
using Yao: X, Y, Z, H, Rx, Rz, Ry, I2, chain, put, control, zero_state, expect, apply, rot, mat, matblock, swap, SWAP, time_evolve, kron
using PauliPropagation

# Comparison tests against Yao.jl.
# The Yao side (circuits and observables) is built by `paulipropagation2yao`,
# so the bespoke Yao constructors that used to live here are gone.

rng = MersenneTwister()
println("Global Yao.jl comparison test seed: $(rng.seed)")

# --- PauliPropagation-side gate inversion (used by the round-trip tests) ---

function _register_inverse!(symbol::Symbol)
    inv_symbol = Symbol(symbol, "_inv")
    if !haskey(clifford_map, inv_symbol)
        clifford_map[inv_symbol] = transposecliffordmap(clifford_map[symbol])
    end
    return inv_symbol
end

function _invert_gates(gates::Vector{<:Any}, θs::Vector{Float64})
    inverted_gates = []
    inverted_θs = []
    for gate in reverse(gates)
        if gate isa PauliRotation
            push!(inverted_gates, gate)
            push!(inverted_θs, -pop!(θs))
        elseif gate isa CliffordGate
            inv_sym = _register_inverse!(gate.symbol)
            push!(inverted_gates, CliffordGate(inv_sym, gate.qinds))
        else
            throw(ArgumentError("Cannot invert $(typeof(gate))"))
        end
    end
    return inverted_gates, inverted_θs
end

# Append a random gate to the PauliPropagation circuit (and its angle, for rotations).
# The Yao side is built later by paulipropagation2yao(custom_gates, nqubits, θs).
function _insert_gate!(gate_type, nqubits, rng, custom_gates, θs)
    rand_qubit() = rand(rng, 1:nqubits)
    rand_angle() = rand(rng) * π / 2
    function distinct_pair()
        c = rand_qubit()
        t = rand_qubit()
        while t == c
            t = rand_qubit()
        end
        return (c, t)
    end

    if gate_type == :H
        push!(custom_gates, CliffordGate(:H, [rand_qubit()]))
    elseif gate_type == :X
        push!(custom_gates, CliffordGate(:X, [rand_qubit()]))
    elseif gate_type == :Y
        push!(custom_gates, CliffordGate(:Y, [rand_qubit()]))
    elseif gate_type == :Z
        push!(custom_gates, CliffordGate(:Z, [rand_qubit()]))
    elseif gate_type == :RX
        push!(custom_gates, PauliRotation(:X, rand_qubit()))
        push!(θs, rand_angle())
    elseif gate_type == :RY
        push!(custom_gates, PauliRotation(:Y, rand_qubit()))
        push!(θs, rand_angle())
    elseif gate_type == :RZ
        push!(custom_gates, PauliRotation(:Z, rand_qubit()))
        push!(θs, rand_angle())
    elseif gate_type == :CNOT
        if nqubits ≥ 2
            c, t = distinct_pair()
            push!(custom_gates, CliffordGate(:CNOT, [c, t]))
        end
    elseif gate_type == :SWAP
        if nqubits ≥ 2
            q1, q2 = distinct_pair()
            push!(custom_gates, CliffordGate(:SWAP, [q1, q2]))
        end
    elseif gate_type == :PauliRotation
        k = rand(rng, 1:min(3, nqubits))
        qs = sort(unique(rand(rng, 1:nqubits, k)))
        paulis = rand(rng, [:X, :Y, :Z], length(qs))
        push!(custom_gates, PauliRotation(paulis, qs))
        push!(θs, rand_angle())
    else
        error("Unsupported gate type: $gate_type")
    end
    return nothing
end

const all_clifford_gates = collect(keys(PauliPropagation._default_clifford_map))
const single_obs = [inttosymbol(i) for i in 1:3]
const two_obs = [Tuple(inttosymbol(p, 2)) for p in 0:15]

# Test Clifford Gates on All Observables

@testset "Clifford Gate Propagation" begin
    for gate in all_clifford_gates
        n = gate in (:CNOT, :CZ, :SWAP, :ZZpihalf) ? 2 : 1
        qubits = 1:n
        @testset "$gate on Single Qubit Observables" begin
            yao_gate = paulipropagation2yao([CliffordGate(gate, qubits)], n)
            for obs in single_obs
                circ = [CliffordGate(gate, qubits)]
                pauli_obs = PauliSum(n)
                add!(pauli_obs, [obs], [1], 1)
                propagated = propagate(circ, pauli_obs)
                vector_propagated = propagate(circ, VectorPauliSum(pauli_obs))
                test_val = overlapwithzero(propagated)
                vector_test_val = overlapwithzero(vector_propagated)
                @test isapprox(test_val, vector_test_val, atol=1e-10)

                state = zero_state(n)
                evolved = apply(state, yao_gate)
                yao_obs = paulipropagation2yao(PauliString(n, [obs], [1]))
                ref_val = real(expect(yao_obs, evolved))

                @test isapprox(test_val, ref_val, atol=1e-10)
            end
        end

        n == 2 && @testset "$gate on Two Qubit Observables" begin
            yao_gate = paulipropagation2yao([CliffordGate(gate, qubits)], n)
            for (obs1, obs2) in two_obs
                circ = [CliffordGate(gate, qubits)]
                pauli_obs = PauliSum(2)
                add!(pauli_obs, [obs1, obs2], [1, 2], 1)
                propagated = propagate(circ, pauli_obs)
                vector_propagated = propagate(circ, VectorPauliSum(pauli_obs))
                test_val = overlapwithzero(propagated)
                vector_test_val = overlapwithzero(vector_propagated)
                @test isapprox(test_val, vector_test_val, atol=1e-10)

                state = zero_state(2)
                evolved = apply(state, yao_gate)
                yao_obs = paulipropagation2yao(PauliString(2, [obs1, obs2], [1, 2]))
                ref_val = real(expect(yao_obs, evolved))
                @test isapprox(test_val, ref_val, atol=1e-10)
            end
        end
    end
end

# Test PauliRotation Gates

@testset "PauliRotation Gates" begin
    @testset "Basic Properties" begin
        @test tomatrix(PauliRotation(:Z, 1), 0) ≈ Matrix(I, 2, 2) && tomatrix(PauliRotation(:Z, 1), π / 2) ≈ [exp(-im * π / 4) 0; 0 exp(im * π / 4)]
        @test tomatrix(PauliRotation(:Z, 1), π) ≈ -im * mat(Z) && tomatrix(PauliRotation(:X, 1), π) ≈ -im * mat(X) && tomatrix(PauliRotation(:Y, 1), π) ≈ -im * mat(Y)
    end
    @testset "Against Yao Rotations" begin
        for (axis, yao_rot) in [(:X, Rx), (:Y, Ry), (:Z, Rz)]
            θ = randn()
            yao_gate = put(1, 1 => yao_rot(θ))
            pr_gate = PauliRotation(axis, 1)
            @test mat(yao_gate) ≈ tomatrix(pr_gate, θ)
        end
    end
    @testset "Multi-Qubit Rotations" begin
        for (symbols, op) in [([:X, :X], kron(X, X)), ([:Y, :Y], kron(Y, Y)), ([:Z, :Z], kron(Z, Z))]
            θ = randn()
            pr = PauliRotation(symbols, [1, 2])
            yao = put(2, (1, 2) => time_evolve(op, θ / 2))
            @test tomatrix(pr, θ) ≈ mat(yao)
        end
    end
end

# Test Random Circuits with PauliRotations and Cliffords

@testset "Randomized PauliRotation & Clifford Tests" begin
    for trial in 1:10
        nqubits = rand(rng, 1:3)
        depth = rand(rng, 5:10)
        custom_gates = Any[]
        θs = Float64[]
        for _ in 1:depth
            gate_type = rand(rng, [:H, :X, :Y, :Z, :RX, :RY, :RZ, :CNOT, :SWAP, :PauliRotation])
            _insert_gate!(gate_type, nqubits, rng, custom_gates, θs)
        end
        if rand(rng) < 0.5
            k = rand(rng, 1:nqubits)
            obs_qubits = sort(unique(rand(rng, 1:nqubits, k)))
            obs_symbols = rand(rng, [:X, :Y, :Z], length(obs_qubits))
        else
            obs_symbols = [:Z]
            obs_qubits = [rand(rng, 1:nqubits)]
        end
        @testset "Trial $trial (n=$nqubits, depth=$depth)" begin
            obs = PauliSum(nqubits)
            add!(obs, obs_symbols, obs_qubits, 1.0)
            propagated = propagate(custom_gates, obs, θs)
            vector_propagated = propagate(custom_gates, VectorPauliSum(obs), θs)
            custom_val = overlapwithzero(propagated)
            vector_test_val = overlapwithzero(vector_propagated)
            @test isapprox(custom_val, vector_test_val, atol=1e-10)

            zero_st = zero_state(nqubits)
            evolved_state = apply(zero_st, paulipropagation2yao(custom_gates, nqubits, θs))
            yao_obs = paulipropagation2yao(PauliString(nqubits, obs_symbols, obs_qubits))
            yao_val = real(expect(yao_obs, evolved_state))
            @test isapprox(custom_val, yao_val; atol=1e-10)

            rev_gates, rev_θs = _invert_gates(custom_gates, θs)
            roundtrip_obs = propagate(rev_gates, propagated, rev_θs)
            vector_roundtrip_obs = propagate(rev_gates, vector_propagated, rev_θs)
            orig_val_zero = overlapwithzero(obs)
            roundtrip_val_zero = overlapwithzero(roundtrip_obs)
            vector_roundtrip_val_zero = overlapwithzero(vector_roundtrip_obs)
            @test isapprox(roundtrip_val_zero, orig_val_zero; atol=1e-10)
            @test isapprox(vector_roundtrip_val_zero, orig_val_zero; atol=1e-10)
        end
    end
end

# Test Model Hamiltonians

@testset "Transverse Field Ising Model" begin
    for nqubits in [2, 3, 4]
        J, h, dt, nsteps = 1.0, 0.5, 0.1, 3
        circ = tfitrottercircuit(nqubits, nsteps)
        θs = Float64[]
        for _ in 1:nsteps
            append!(θs, fill(-J * dt, nqubits - 1))
            append!(θs, fill(-h * dt, nqubits))
        end
        state = apply(zero_state(nqubits), paulipropagation2yao(circ, nqubits, θs))
        @testset "n = $nqubits" begin
            for q in 1:nqubits, p in [:X, :Y, :Z]
                obs = PauliSum(nqubits)
                add!(obs, [p], [q], 1.0)
                our_val = overlapwithzero(propagate(circ, obs, θs))
                yao_obs = paulipropagation2yao(PauliString(nqubits, [p], [q]))
                yao_val = real(expect(yao_obs, state))
                @test isapprox(our_val, yao_val; atol=1e-10)
            end
            if nqubits ≥ 2
                for q1 in 1:nqubits-1
                    obs = PauliSum(nqubits)
                    add!(obs, [:Z, :Z], [q1, q1 + 1], 1.0)
                    our_val = overlapwithzero(propagate(circ, obs, θs))
                    vector_propagated = propagate(circ, VectorPauliSum(obs), θs)
                    vector_test_val = overlapwithzero(vector_propagated)
                    yao_obs = paulipropagation2yao(PauliString(nqubits, [:Z, :Z], [q1, q1 + 1]))
                    yao_val = real(expect(yao_obs, state))
                    @test isapprox(our_val, yao_val; atol=1e-10)
                    @test isapprox(our_val, vector_test_val, atol=1e-10)
                end
            end
        end
    end
end

@testset "Heisenberg Model" begin
    for nqubits in [2, 3]
        Jx, Jy, Jz = 0.8, 0.9, 1.0
        dt = 0.05
        nsteps = 2
        circ = heisenbergtrottercircuit(nqubits, nsteps)
        θs = Float64[]
        for _ in 1:nsteps
            for _ in 1:nqubits-1
                append!(θs, [-Jx * dt, -Jy * dt, -Jz * dt])
            end
        end
        state = apply(zero_state(nqubits), paulipropagation2yao(circ, nqubits, θs))
        @testset "n = $nqubits" begin
            for q in 1:nqubits, p in [:X, :Y, :Z]
                obs = PauliSum(nqubits)
                add!(obs, [p], [q], 1.0)
                our_val = overlapwithzero(propagate(circ, obs, θs))
                vec_val = overlapwithzero(propagate(circ, VectorPauliSum(obs), θs))
                @test isapprox(our_val, vec_val; atol=1e-10)
                yao_obs = paulipropagation2yao(PauliString(nqubits, [p], [q]))
                yao_val = real(expect(yao_obs, state))
                # tightened from 1e-2: the Yao side is now the exact conversion of `circ`,
                # not a separately hand-built Trotter circuit, so they agree to machine precision
                @test isapprox(our_val, yao_val; atol=1e-10)
            end
            if nqubits ≥ 2
                for (p1, p2) in [(:X, :X), (:Y, :Y), (:Z, :Z)]
                    obs = PauliSum(nqubits)
                    add!(obs, [p1, p2], [1, 2], 1.0)
                    our_val = overlapwithzero(propagate(circ, obs, θs))
                    vec_val = overlapwithzero(propagate(circ, VectorPauliSum(obs), θs))
                    @test isapprox(our_val, vec_val; atol=1e-10)
                    yao_obs = paulipropagation2yao(PauliString(nqubits, [p1, p2], [1, 2]))
                    yao_val = real(expect(yao_obs, state))
                    @test isapprox(our_val, yao_val; atol=1e-10)
                end
            end
        end
    end
end

# Integration Tests

@testset "Integration Tests for Circuits" begin
    circs = [
        (nqubits=2, custom_gates=[PauliRotation(:X, 1)],                   obs=([:Z], [1])),
        (nqubits=2, custom_gates=[CliffordGate(:CNOT, [1, 2])],            obs=([:Z], [2])),
        (nqubits=2, custom_gates=[PauliRotation(:Z, 1)],                   obs=([:X], [1])),
        (nqubits=2, custom_gates=[PauliRotation([:X, :X], [1, 2])],        obs=([:Z, :Z], [1, 2])),
        (nqubits=2, custom_gates=[PauliRotation([:Y, :Y], [1, 2])],        obs=([:X, :X], [1, 2])),
        (nqubits=2, custom_gates=[PauliRotation([:Z, :Z], [1, 2])],        obs=([:Y, :Y], [1, 2])),
        (nqubits=3, custom_gates=[PauliRotation([:X, :Y, :Z], [1, 2, 3])], obs=([:Z, :Y, :X], [1, 2, 3])),
    ]
    for circ in circs
        @testset "nqubits=$(circ.nqubits), obs=$(circ.obs)" begin
            θs = fill(π / 4, count(g -> g isa PauliRotation, circ.custom_gates))
            obs = PauliSum(circ.nqubits)
            obs_symbols, obs_qubits = circ.obs
            add!(obs, obs_symbols, obs_qubits, 1.0)
            propagated = propagate(circ.custom_gates, obs, θs)
            vector_propagated = propagate(circ.custom_gates, VectorPauliSum(obs), θs)
            vector_test_val = overlapwithzero(vector_propagated)
            custom_val = overlapwithzero(propagated)
            @test isapprox(custom_val, vector_test_val; atol=1e-10)

            zero_st = zero_state(circ.nqubits)
            evolved_state = apply(zero_st, paulipropagation2yao(circ.custom_gates, circ.nqubits, θs))
            yao_obs = paulipropagation2yao(PauliString(circ.nqubits, obs_symbols, obs_qubits))
            yao_val = real(expect(yao_obs, evolved_state))
            @test isapprox(custom_val, yao_val; atol=1e-10)

            rev_gates, rev_θs = _invert_gates(circ.custom_gates, θs)
            roundtrip_obs = propagate(rev_gates, propagated, rev_θs)
            vector_roundtrip_obs = propagate(rev_gates, vector_propagated, rev_θs)
            @test PauliSum(vector_roundtrip_obs) == obs
            @test roundtrip_obs == obs
        end
    end
end