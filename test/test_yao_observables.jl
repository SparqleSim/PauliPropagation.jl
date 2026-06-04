using Test
using PauliPropagation
using YaoBlocks
using YaoBlocks.ConstGate: X, Y, Z, I2
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

# Inverse of paulipropagation2yao for round-trip tests (local to tests; Yao ext may be unavailable on PP 0.7).
function _yao_obs_to_paulisum(obs)
    obs = YaoBlocks.Optimise.eliminate_nested(obs)
    if obs isa YaoBlocks.Add
        return PauliSum([_yao_obs_to_paulisum(b) for b in YaoBlocks.subblocks(obs)])
    elseif obs isa YaoBlocks.Scale
        return _yao_obs_to_paulisum(obs.content) * obs.alpha
    elseif obs isa YaoBlocks.KronBlock
        n = YaoBlocks.nqubits(obs)
        syms = [_yao_gate_to_symbol(b) for b in obs.blocks]
        locs = [loc[1] for loc in obs.locs]
        return PauliString(n, syms, locs)
    elseif obs isa YaoBlocks.PutBlock
        n = YaoBlocks.nqubits(obs)
        return PauliString(n, _yao_gate_to_symbol(obs.content), obs.locs[1])
    else
        error("Unsupported Yao observable type: $(typeof(obs))")
    end
end

const _yao_symbol_map = Dict(
    I2 => :I,
    X => :X,
    Y => :Y,
    Z => :Z,
)

_yao_gate_to_symbol(g) = _yao_symbol_map[g]

@testset "paulipropagation2yao observables" begin
    @testset "PauliString" begin
        n = 5
        pstr = PauliString(n, :Z, 3)
        obs = paulipropagation2yao(pstr)
        @test obs == put(n, 3 => Z)

        pstr_xy = PauliString(n, [:X, :Z], [1, 3], 2.5im)
        obs_xy = paulipropagation2yao(pstr_xy)
        @test obs_xy == Scale(2.5im, kron(n, 1 => X, 3 => Z))
        @test _yao_obs_to_paulisum(obs_xy) == pstr_xy
    end

    @testset "PauliSum" begin
        n = 4
        psum = PauliSum([PauliString(n, :X, 1), PauliString(n, :Z, 2, 0.5)])
        obs = paulipropagation2yao(psum)
        @test obs isa YaoBlocks.Add
        @test _pauli_sum_equal(_yao_obs_to_paulisum(obs), psum)
    end

    @testset "numeric expectation on |0⟩" begin
        nq = 6
        pstr = PauliString(nq, :Z, 3)
        yao_obs = paulipropagation2yao(pstr)
        reg = zero_state(nq)
        pp_val = overlapwithzero(pstr)
        yao_val = real(expect(yao_obs, reg))
        @test isapprox(pp_val, yao_val; atol=1e-10)
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
