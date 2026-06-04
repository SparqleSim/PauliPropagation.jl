# test/test_paulipropagation2yao.jl
#
# spec for `paulipropagation2yao()` issue #146.

using Test
using PauliPropagation
using Yao: X, Y, Z, I2, put, chain, zero_state, apply, expect, mat

# Kept local so this test file doesn't depend on helpers in other test files.
_paulisym(s::Symbol) = s === :X ? X : s === :Y ? Y : s === :Z ? Z : I2
function _expected_obs(syms, qs, nq)
    length(qs) == 1 && return put(nq, qs[1] => _paulisym(syms[1]))
    return chain(nq, (put(nq, q => _paulisym(s)) for (s, q) in zip(syms, qs))...)
end

@testset "paulipropagation2yao" begin

    @testset "PauliString -> Yao observable" begin
        # (symbols, qubits, nqubits)
        cases = [([:Z], [1], 1),
                 ([:X], [2], 3),
                 ([:Z, :Y], [1, 3], 4),
                 ([:X, :Y, :Z], [1, 2, 3], 3)]
        for (syms, qs, nq) in cases
            pstr = PauliString(nq, syms, qs)
            yao  = paulipropagation2yao(pstr)
            @test mat(yao) ≈ mat(_expected_obs(syms, qs, nq))
            paulipropagation2yao(pstr)
        end
    end

    @testset "PauliSum -> Yao observable (with coefficients)" begin
        nq = 3
        psum = PauliSum(nq)
        add!(psum, [:Z], [1], 0.5)
        add!(psum, [:X, :Y], [1, 2], 1.5)
        yao = paulipropagation2yao(psum)
        expected = 0.5 * mat(_expected_obs([:Z], [1], nq)) +
                   1.5 * mat(_expected_obs([:X, :Y], [1, 2], nq))
        @test mat(yao) ≈ expected
    end

    @testset "static circuit -> Yao circuit (expectation equivalence)" begin
        # all gates are static (Clifford + TGate) so propagate needs no angles
        cases = [
            (nq = 2, gates = [CliffordGate(:H, [1]), TGate(1), CliffordGate(:CNOT, [1, 2])], obs = ([:Z, :Z], [1, 2])),
            (nq = 1, gates = [CliffordGate(:H, [1]), TGate(1), CliffordGate(:H, [1])],        obs = ([:Z], [1])),
            (nq = 3, gates = [CliffordGate(:H, [1]), CliffordGate(:CNOT, [1, 2]), TGate(3)],  obs = ([:X], [1])),
        ]
        for c in cases
            syms, qs = c.obs
            # PauliPropagation reference value
            obs = PauliSum(c.nq)
            add!(obs, syms, qs, 1.0)
            pp_val = overlapwithzero(propagate(c.gates, obs))
            # converted Yao circuit, evaluated against the trusted observable
            yao_circ = paulipropagation2yao(c.gates, c.nq)
            yao_val  = real(expect(_expected_obs(syms, qs, c.nq), apply(zero_state(c.nq), yao_circ)))
            @test isapprox(pp_val, yao_val; atol = 1e-10)
        end
    end
end
@testset "paulipropagation2yao — parametrised circuits" begin
    cases = [
        (nq=2, gates=[PauliRotation(:X, 1)],              θs=[π/4],     obs=([:Z],[1])),
        (nq=2, gates=[PauliRotation(:Z, 1)],              θs=[π/3],     obs=([:X],[1])),
        (nq=2, gates=[PauliRotation([:X,:X],[1,2])],      θs=[π/4],     obs=([:Z,:Z],[1,2])),
        (nq=3, gates=[PauliRotation([:X,:Y,:Z],[1,2,3])], θs=[π/8],     obs=([:Z,:Y,:X],[1,2,3])),
        (nq=2, gates=[CliffordGate(:H,[1]), PauliRotation(:Z,1), TGate(1),
                      CliffordGate(:CNOT,[1,2]), PauliRotation([:X,:Y],[1,2])],
                                                          θs=[0.7,1.1], obs=([:Z,:Z],[1,2])),
    ]
    for c in cases
        syms, qs = c.obs
        @testset "n=$(c.nq) obs=$syms" begin
            obs = PauliSum(c.nq); add!(obs, syms, qs, 1.0)
            pp_val  = overlapwithzero(propagate(c.gates, obs, c.θs))
            yao     = paulipropagation2yao(c.gates, c.nq, c.θs)
            yao_val = real(expect(_expected_obs(syms, qs, c.nq), apply(zero_state(c.nq), yao)))
            @test isapprox(pp_val, yao_val; atol=1e-10)
        end
    end
end

@testset "paulipropagation2yao — TFIM Trotter circuit" begin
    for nq in [2, 3]
        nsteps, J, h, dt = 3, 1.0, 0.5, 0.1
        circ = tfitrottercircuit(nq, nsteps)
        θs = Float64[]
        for _ in 1:nsteps
            append!(θs, fill(-J*dt, nq-1))
            append!(θs, fill(-h*dt, nq))
        end
        state = apply(zero_state(nq), paulipropagation2yao(circ, nq, θs))
        @testset "n=$nq" begin
            for q in 1:nq, p in (:X, :Y, :Z)
                obs = PauliSum(nq); add!(obs, [p], [q], 1.0)
                pp_val  = overlapwithzero(propagate(circ, obs, θs))
                yao_val = real(expect(_expected_obs([p],[q],nq), state))
                @test isapprox(pp_val, yao_val; atol=1e-10)
            end
        end
    end
end