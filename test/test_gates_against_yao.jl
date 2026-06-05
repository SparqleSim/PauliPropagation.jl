using Test
using LinearAlgebra
using Random
using Yao: X, Y, Z, H, Rx, Rz, Ry, I2, chain, put, control, zero_state, expect, apply, rot, mat, matblock, swap, SWAP, time_evolve, kron
using PauliPropagation

# Gate Translation Functions

rng = MersenneTwister()
println("Global Yao.jl comparison test seed: $(rng.seed)")

_Rzz(θ) = put(2, 1:2 => matblock(Diagonal([exp(-im * θ / 2), exp(im * θ / 2), exp(im * θ / 2), exp(-im * θ / 2)])))

function _clifford_to_yao(g::CliffordGate)
    gate_dict = Dict(
        :X => X,
        :Y => Y,
        :Z => Z,
        :H => H,
        :S => chain(1, Rz(π / 2)),
        :SX => chain(1, Rx(π / 2)),
        :SY => chain(1, Ry(π / 2)),
        :CNOT => control(2, 1, 2 => X),
        :CZ => control(2, 1, 2 => Z),
        :SWAP => SWAP,
        :ZZpihalf => _Rzz(π / 2)
    )
    get(gate_dict, g.symbol) do
        error("Unsupported CliffordGate symbol: $(g.symbol)")
    end
end

function _build_yao_observable(symbols::Vector{Symbol}, qubits::Vector{Int}, nqubits::Int)
    length(symbols) == length(qubits) || throw(ArgumentError("Symbols and qubits must have same length"))
    all(1 ≤ q ≤ nqubits for q in qubits) || throw(ArgumentError("Qubit indices out of range"))
    pauli_map = Dict(
        :X => X,
        :Y => Y,
        :Z => Z,
        :I => I2
    )
    blocks = map(zip(symbols, qubits)) do (sym, q)
        op = get(pauli_map, sym) do
            throw(ArgumentError("Unsupported Pauli symbol: $sym. Use :X, :Y, :Z, or :I"))
        end
        put(nqubits, q => op)
    end
    return length(blocks) == 1 ? only(blocks) : chain(blocks...)
end

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

function _build_evolution_step!(circ, nqubits, ops::Pair{Symbol,Float64}...)
    for (op_symbol, coeff) in ops
        op = op_symbol == :X ? kron(X, X) : op_symbol == :Y ? kron(Y, Y) : op_symbol == :Z ? kron(Z, Z) : error("Unsupported operator")
        for i in 1:nqubits-1
            push!(circ, put(nqubits, (i, i + 1) => time_evolve(op, -coeff / 2)))
        end
    end
end

function _yao_heisenberg_circ(nqubits::Int, nsteps::Int, Jx::Float64, Jy::Float64, Jz::Float64, dt::Float64)
    circ = chain(nqubits)
    for _ in 1:nsteps
        _build_evolution_step!(circ, nqubits, :X => Jx * dt, :Y => Jy * dt, :Z => Jz * dt)
    end
    return circ
end

function _yao_tfi_circ(nqubits::Int, nsteps::Int, J::Float64, h::Float64, dt::Float64)
    circ = chain(nqubits)
    for _ in 1:nsteps
        _build_evolution_step!(circ, nqubits, :Z => J * dt)
        for i in 1:nqubits
            push!(circ, put(nqubits, i => Rx(-h * dt)))
        end
    end
    return circ
end

function _insert_gate!(gate_type, nqubits, rng, custom_gates, yao_ops, θs)
    rand_qubit() = rand(rng, 1:nqubits)
    rand_angle() = rand(rng) * π / 2
    distinct_pair() = (c = rand_qubit();
    t = rand_qubit();
    while t == c
        t = rand_qubit()
    end;
    (c, t))
    gate_handlers = Dict(
        :H => () -> let q = rand_qubit()
            push!(custom_gates, CliffordGate(:H, [q]))
            put(nqubits, q => H)
        end,
        :X => () -> let q = rand_qubit()
            push!(custom_gates, CliffordGate(:X, [q]))
            put(nqubits, q => X)
        end,
        :Y => () -> let q = rand_qubit()
            push!(custom_gates, CliffordGate(:Y, [q]))
            put(nqubits, q => Y)
        end,
        :Z => () -> let q = rand_qubit()
            push!(custom_gates, CliffordGate(:Z, [q]))
            put(nqubits, q => Z)
        end,
        :RX => () -> let q = rand_qubit(), θ = rand_angle()
            push!(custom_gates, PauliRotation(:X, q))
            push!(θs, θ)
            put(nqubits, q => Rx(θ))
        end,
        :RY => () -> let q = rand_qubit(), θ = rand_angle()
            push!(custom_gates, PauliRotation(:Y, q))
            push!(θs, θ)
            put(nqubits, q => Ry(θ))
        end,
        :RZ => () -> let q = rand_qubit(), θ = rand_angle()
            push!(custom_gates, PauliRotation(:Z, q))
            push!(θs, θ)
            put(nqubits, q => Rz(θ))
        end,
        :CNOT => () -> nqubits < 2 ? nothing : let (c, t) = distinct_pair()
            push!(custom_gates, CliffordGate(:CNOT, [c, t]))
            control(nqubits, c, t => X)
        end,
        :SWAP => () -> nqubits < 2 ? nothing : let (q1, q2) = distinct_pair()
            push!(custom_gates, CliffordGate(:SWAP, [q1, q2]))
            swap(nqubits, q1, q2)
        end,
        :PauliRotation => () -> let
            k = rand(rng, 1:min(3, nqubits))
            qs = sort(unique(rand(rng, 1:nqubits, k)))
            paulis = rand(rng, [:X, :Y, :Z], length(qs))
            θ = rand_angle()
            push!(custom_gates, PauliRotation(paulis, qs))
            push!(θs, θ)
            if length(qs) == 1
                q = qs[1]
                p = paulis[1]
                return put(nqubits, q => p == :X ? Rx(θ) : p == :Y ? Ry(θ) : Rz(θ))
            else
                blk = chain(nqubits)
                for (p, q) in zip(paulis, qs)
                    op = p == :X ? X : p == :Y ? Y : Z
                    push!(blk, put(nqubits, q => op))
                end
                return time_evolve(blk, θ / 2)
            end
        end
    )
    handler = get(gate_handlers, gate_type) do
        error("Unsupported gate type: $gate_type")
    end
    yao_op = handler()
    yao_op !== nothing && push!(yao_ops, yao_op)
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
            yao_gate = _clifford_to_yao(CliffordGate(gate, qubits))
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
                yao_obs = _build_yao_observable([obs], [1], n)
                ref_val = real(expect(yao_obs, evolved))

                @test isapprox(test_val, ref_val, atol=1e-10)
            end
        end

        n == 2 && @testset "$gate on Two Qubit Observables" begin
            yao_gate = _clifford_to_yao(CliffordGate(gate, qubits))
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
                yao_obs = _build_yao_observable([obs1, obs2], [1, 2], 2)
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
        yao_ops = Any[]
        θs = Float64[]
        for _ in 1:depth
            gate_type = rand(rng, [:H, :X, :Y, :Z, :RX, :RY, :RZ, :CNOT, :SWAP, :PauliRotation])
            _insert_gate!(gate_type, nqubits, rng, custom_gates, yao_ops, θs)
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
            evolved_state = apply(zero_st, chain(yao_ops...))
            yao_obs = _build_yao_observable(obs_symbols, obs_qubits, nqubits)
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
        yao_circ = _yao_tfi_circ(nqubits, nsteps, J, h, dt)
        state = zero_state(nqubits) |> yao_circ
        @testset "n = $nqubits" begin
            for q in 1:nqubits, p in [:X, :Y, :Z]
                obs = PauliSum(nqubits)
                add!(obs, [p], [q], 1.0)
                our_val = overlapwithzero(propagate(circ, obs, θs))
                yao_obs = _build_yao_observable([p], [q], nqubits)
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
                    yao_obs = _build_yao_observable([:Z, :Z], [q1, q1 + 1], nqubits)
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
        yao_circ = _yao_heisenberg_circ(nqubits, nsteps, Jx, Jy, Jz, dt)
        state = zero_state(nqubits) |> yao_circ
        @testset "n = $nqubits" begin
            for q in 1:nqubits, p in [:X, :Y, :Z]
                obs = PauliSum(nqubits)
                add!(obs, [p], [q], 1.0)
                our_val = overlapwithzero(propagate(circ, obs, θs))
                vec_val = overlapwithzero(propagate(circ, VectorPauliSum(obs), θs))
                @test isapprox(our_val, vec_val; atol=1e-10)
                yao_obs = _build_yao_observable([p], [q], nqubits)
                yao_val = real(expect(yao_obs, state))

                @test isapprox(our_val, yao_val; atol=1e-2)
            end
            if nqubits ≥ 2
                for (p1, p2) in [(:X, :X), (:Y, :Y), (:Z, :Z)]
                    obs = PauliSum(nqubits)
                    add!(obs, [p1, p2], [1, 2], 1.0)

                    our_val = overlapwithzero(propagate(circ, obs, θs))
                    vec_val = overlapwithzero(propagate(circ, VectorPauliSum(obs), θs))
                    @test isapprox(our_val, vec_val; atol=1e-10)
                    yao_obs = _build_yao_observable([p1, p2], [1, 2], nqubits)
                    yao_val = real(expect(yao_obs, state))
                    @test isapprox(our_val, yao_val; atol=1e-2)
                end
            end
        end
    end
end

# Integration Tests

@testset "Integration Tests for Circuits" begin
    XX = kron(X, X)
    YY = kron(Y, Y)
    ZZ = kron(Z, Z)
    XYZ = kron(X, kron(Y, Z))
    circs = [
        (
            nqubits=2,
            custom_gates=[PauliRotation(:X, 1)],
            yao_circ=chain(put(2, 1 => Rx(π / 4))),
            obs=([:Z], [1])
        ),
        (
            nqubits=2,
            custom_gates=[CliffordGate(:CNOT, [1, 2])],
            yao_circ=chain(control(2, 1, 2 => X)),
            obs=([:Z], [2])
        ),
        (
            nqubits=2,
            custom_gates=[PauliRotation(:Z, 1)],
            yao_circ=chain(put(2, 1 => Rz(π / 4))),
            obs=([:X], [1])
        ),
        (
            nqubits=2,
            custom_gates=[PauliRotation([:X, :X], [1, 2])],
            yao_circ=chain(put(2, (1, 2) => time_evolve(XX, π / 8))),
            obs=([:Z, :Z], [1, 2])
        ),
        (
            nqubits=2,
            custom_gates=[PauliRotation([:Y, :Y], [1, 2])],
            yao_circ=chain(put(2, (1, 2) => time_evolve(YY, π / 8))),
            obs=([:X, :X], [1, 2])
        ),
        (
            nqubits=2,
            custom_gates=[PauliRotation([:Z, :Z], [1, 2])],
            yao_circ=chain(put(2, (1, 2) => time_evolve(ZZ, π / 8))),
            obs=([:Y, :Y], [1, 2])
        ),
        (
            nqubits=3,
            custom_gates=[PauliRotation([:X, :Y, :Z], [1, 2, 3])],
            yao_circ=chain(put(3, (1, 2, 3) => time_evolve(XYZ, π / 8))),
            obs=([:Z, :Y, :X], [1, 2, 3])
        ),
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
            evolved_state = apply(zero_st, circ.yao_circ)
            yao_obs = _build_yao_observable(obs_symbols, obs_qubits, circ.nqubits)
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
