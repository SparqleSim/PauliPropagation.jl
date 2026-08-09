using Test
using PauliPropagation
using YaoBlocks
using YaoBlocks.ConstGate: X, Y, Z
using YaoArrayRegister: expect, zero_state, density_matrix
using YaoBlocks: apply!

function _pp_yao_overlap(circuit, pstr, thetas, reg)
    pp_val = overlapwithzero(propagate(circuit, pstr, thetas; min_abs_coeff=0))
    yao_val = real(expect(paulipropagation2yao(pstr), reg))
    return pp_val, yao_val
end

@testset "PauliPropagationYao extension" begin
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
                @test pc.observable == PauliSum([pstr])
            end
        end
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
            base = hardwareefficientcircuit(nq, nl; topology=topo)
            m = countparameters(base)
            pstr = PauliString(nq, :Z, 2)
            noises = (DepolarizingNoise, PauliXNoise, PauliYNoise, PauliZNoise, AmplitudeDampingNoise)
            for noise in noises
                @testset "$noise" begin
                    circuit = deepcopy(base)
                    insert!(circuit, 1, noise(2))
                    thetas = rand(m)
                    insert!(thetas, 1, 0.3)
                    yao_circ = paulipropagation2yao(nq, circuit, thetas)
                    reg = apply!(copy(zero_state(nq) |> density_matrix), yao_circ)
                    pp_val, yao_val = _pp_yao_overlap(circuit, pstr, thetas, reg)
                    @test isapprox(pp_val, yao_val; atol=1e-6)
                end
            end
        end

        @testset "multi-qubit rotation with unsorted qubit indices" begin
            nq = 3
            circuit = PauliPropagation.Gate[PauliRotation(:Y, q) for q in 1:nq]
            push!(circuit, PauliRotation([:X, :Y, :Z], [3, 1, 2]))
            thetas = [0.31, 0.52, 0.73, 0.63]
            yao_circ = paulipropagation2yao(nq, circuit, thetas)
            reg = apply!(copy(zero_state(nq)), yao_circ)
            for pstr in (PauliString(nq, :X, 1), PauliString(nq, :Z, 1), PauliString(nq, [:Z, :Z], [1, 2]))
                pp_val, yao_val = _pp_yao_overlap(circuit, pstr, thetas, reg)
                @test isapprox(pp_val, yao_val; atol=1e-10)
            end
        end

        @testset "parameter count" begin
            circuit = [PauliRotation([:X], [1]), CliffordGate(:H, 2)]
            @test_throws ArgumentError paulipropagation2yao(3, circuit, Float64[])
            @test paulipropagation2yao(3, circuit, [0.2]) isa ChainBlock
            @test_throws MethodError paulipropagation2yao(3, circuit)
        end
    end
end
