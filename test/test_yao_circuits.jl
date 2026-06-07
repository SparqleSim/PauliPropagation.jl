using Test
using PauliPropagation
using YaoBlocks
using YaoArrayRegister: expect, zero_state, density_matrix
using YaoBlocks: apply!

function _pp_yao_overlap(circuit, pstr, thetas, reg)
    pp_val = overlapwithzero(propagate(circuit, pstr, thetas; min_abs_coeff=0))
    yao_val = real(expect(paulipropagation2yao(pstr), reg))
    return pp_val, yao_val
end

@testset "paulipropagation2yao circuits" begin
    @testset "parametric Clifford + rotations" begin
        nq = 6
        nl = 2
        circuit = tfitrottercircuit(nq, nl)
        thetas = randn(countparameters(circuit)) * 0.4
        pstr = PauliString(nq, :Z, 3)
        yao_circ = paulipropagation2yao(nq, circuit, thetas)
        reg = apply!(copy(zero_state(nq)), yao_circ)
        pp_val, yao_val = _pp_yao_overlap(circuit, pstr, thetas, reg)
        @test isapprox(pp_val, yao_val; atol=1e-6)
    end

    @testset "hardware efficient with noise" begin
        nq = 4
        nl = 2
        topo = bricklayertopology(nq; periodic=false)
        circuit = hardwareefficientcircuit(nq, nl; topology=topo)
        m = countparameters(circuit)
        circuit = deepcopy(circuit)
        insert!(circuit, 1, DepolarizingNoise(2))
        thetas = rand(m)
        insert!(thetas, 1, 0.05)
        pstr = PauliString(nq, :Z, 2)
        yao_circ = paulipropagation2yao(nq, circuit, thetas)
        reg = apply!(copy(zero_state(nq) |> density_matrix), yao_circ)
        pp_val, yao_val = _pp_yao_overlap(circuit, pstr, thetas, reg)
        @test isapprox(pp_val, yao_val; atol=1e-6)
    end

    @testset "parameter count" begin
        circuit = [PauliRotation([:X], [1]), CliffordGate(:H, 2)]
        @test_throws ArgumentError paulipropagation2yao(3, circuit, Float64[])
        @test paulipropagation2yao(3, circuit, [0.2]) isa ChainBlock
        @test_throws MethodError paulipropagation2yao(3, circuit)
    end
end
