using Test
using PauliPropagation
using YaoBlocks
using YaoBlocks.ConstGate: X, Y, Z
using YaoArrayRegister: expect, zero_state

function _pauli_sum_equal(a::PauliSum, b::PauliSum)
    a.nqubits == b.nqubits || return false
    length(a.terms) == length(b.terms) || return false
    for (k, v) in a.terms
        haskey(b.terms, k) || return false
        isapprox(v, b.terms[k]) || return false
    end
    return true
end

@testset "paulipropagation2yao observables" begin
    @testset "PauliString" begin
        n = 5
        pstr = PauliString(n, :Z, 3)
        @test paulipropagation2yao(pstr) == put(n, 3 => Z)

        pstr_xy = PauliString(n, [:X, :Z], [1, 3], 2.5im)
        @test paulipropagation2yao(pstr_xy) == Scale(2.5im, kron(n, 1 => X, 3 => Z))
    end

    @testset "PauliSum" begin
        n = 4
        psum = PauliSum([PauliString(n, :X, 1), PauliString(n, :Z, 2, 0.5)])
        obs = paulipropagation2yao(psum)
        @test obs isa YaoBlocks.Add
        @test isapprox(overlapwithzero(psum), real(expect(obs, zero_state(n))); atol=1e-10)
    end

    @testset "numeric expectation on |0⟩" begin
        nq = 6
        pstr = PauliString(nq, :Z, 3)
        yao_obs = paulipropagation2yao(pstr)
        @test isapprox(overlapwithzero(pstr), real(expect(yao_obs, zero_state(nq))); atol=1e-10)
    end

    @testset "errors" begin
        @test_throws ArgumentError paulipropagation2yao(PauliSum(3))
    end

    if isdefined(YaoBlocks, :yao2paulipropagation)
        @testset "YaoBlocks round-trip (extension loaded)" begin
            n = 5
            pstr = PauliString(n, :Y, 2)
            obs = paulipropagation2yao(pstr)
            pc = YaoBlocks.yao2paulipropagation(chain(n); observable=obs)
            @test _pauli_sum_equal(pc.observable, PauliSum([pstr]))
        end
    end
end
