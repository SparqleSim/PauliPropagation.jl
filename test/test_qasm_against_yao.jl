using Test
using LinearAlgebra
using PauliPropagation
using Yao

include("qasm_yao_common.jl")

function _u3_matrix(theta::Real, phi::Real, lambda::Real)
    cos_half = cos(theta / 2)
    sin_half = sin(theta / 2)
    return [
        cos_half                 -exp(im * lambda) * sin_half;
        exp(im * phi) * sin_half  exp(im * (phi + lambda)) * cos_half
    ]
end

"""
State-preparation probes shared by PP and Yao.
Each entry is `(name, prep_pp, prep_thetas, prep_yao)`.
"""
function _state_prep_probes(nq::Int)
    probes = Any[
        ("zero", Any[], Float64[], nothing),
    ]
    if nq >= 1
        push!(probes, ("H on q1", Any[CliffordGate(:H, 1)], Float64[], put(nq, 1 => H)))
    end
    if nq >= 2
        push!(probes, ("X on q2", Any[CliffordGate(:X, 2)], Float64[], put(nq, 2 => X)))
        push!(
            probes,
            (
                "H then X",
                Any[CliffordGate(:H, 1), CliffordGate(:X, 2)],
                Float64[],
                chain(nq, put(nq, 1 => H), put(nq, 2 => X)),
            ),
        )
    end
    return probes
end

function _assert_qasm_vs_yao_with_preps(
    nq::Int,
    gate_line::AbstractString,
    yao_circuit;
    atol=1e-8,
)
    for (name, prep_pp, prep_thetas, prep_yao) in _state_prep_probes(nq)
        @testset "$name" begin
            _assert_qasm_pp_matches_yao_circuit(
                nq,
                gate_line,
                yao_circuit;
                prep_pp,
                prep_thetas,
                prep_yao,
                atol,
            )
        end
    end
end

@testset "OpenQASM vs Yao" begin
    # Released YaoBlocks may not ship OpenQASMExt/`parseblock`. When available,
    # compare PP and Yao on the same QASM string; otherwise use hand-built Yao oracles.

    @testset "parseblock parity (when OpenQASMExt is available)" begin
        if !_yao_qasm_ext_available()
            @test_skip "YaoBlocks.OpenQASMExt is not available in this environment"
        else
            parseblock_cases = [
                (1, "u2(0.5, -0.3) q[0];"),
                (1, "u3(0.4, 0.2, -0.1) q[0];"),
                (2, "cy q[0], q[1];"),
                (2, "ch q[0], q[1];"),
                (3, "ccx q[0], q[1], q[2];"),
                (2, "crz($(pi / 5)) q[0], q[1];"),
                (2, "cu1($(pi / 4)) q[0], q[1];"),
                (2, "cp($(pi / 6)) q[0], q[1];"),
                (2, "cu3(0.5, -0.3, 0.7) q[0], q[1];"),
            ]
            for (nq, gate_line) in parseblock_cases
                @testset "$gate_line" begin
                    for (name, prep_pp, prep_thetas, prep_yao) in _state_prep_probes(nq)
                        @testset "$name" begin
                            _assert_qasm_pp_matches_yao_parseblock(
                                nq,
                                gate_line;
                                prep_pp,
                                prep_thetas,
                                prep_yao,
                                atol=1e-8,
                            )
                        end
                    end
                end
            end
        end
    end

    @testset "hand-built Yao oracles" begin
        @testset "u2" begin
            phi, lambda = 0.5, -0.3
            yao = put(1, 1 => matblock(_u3_matrix(pi / 2, phi, lambda)))
            _assert_qasm_vs_yao_with_preps(1, "u2($phi, $lambda) q[0];", yao)
        end

        @testset "u3" begin
            theta, phi, lambda = 0.4, 0.2, -0.1
            yao = put(1, 1 => matblock(_u3_matrix(theta, phi, lambda)))
            _assert_qasm_vs_yao_with_preps(1, "u3($theta, $phi, $lambda) q[0];", yao)
        end

        @testset "cy" begin
            yao = control(2, 1, 2 => Y)
            _assert_qasm_vs_yao_with_preps(2, "cy q[0], q[1];", yao)
        end

        @testset "ch" begin
            yao = control(2, 1, 2 => H)
            _assert_qasm_vs_yao_with_preps(2, "ch q[0], q[1];", yao)
        end

        @testset "ccx" begin
            yao = control(3, (1, 2), 3 => X)
            _assert_qasm_vs_yao_with_preps(3, "ccx q[0], q[1], q[2];", yao)
        end

        @testset "crz" begin
            theta = pi / 5
            yao = control(2, 1, 2 => Rz(theta))
            _assert_qasm_vs_yao_with_preps(2, "crz($theta) q[0], q[1];", yao)
        end

        @testset "cu1" begin
            lambda = pi / 4
            yao = control(2, 1, 2 => shift(lambda))
            _assert_qasm_vs_yao_with_preps(2, "cu1($lambda) q[0], q[1];", yao)
        end

        @testset "cp" begin
            lambda = pi / 6
            yao = control(2, 1, 2 => shift(lambda))
            _assert_qasm_vs_yao_with_preps(2, "cp($lambda) q[0], q[1];", yao)
        end

        @testset "cu3" begin
            theta, phi, lambda = 0.5, -0.3, 0.7
            yao = control(2, 1, 2 => matblock(_u3_matrix(theta, phi, lambda)))
            _assert_qasm_vs_yao_with_preps(2, "cu3($theta, $phi, $lambda) q[0], q[1];", yao)
        end

        @testset "sxdg" begin
            # Match PP's sxdg translation: S then H then S.
            yao = chain(1, put(1, 1 => ConstGate.S), put(1, 1 => H), put(1, 1 => ConstGate.S))
            _assert_qasm_vs_yao_with_preps(1, "sxdg q[0];", yao)
        end

        @testset "cswap" begin
            yao = control(3, 1, (2, 3) => SWAP)
            _assert_qasm_vs_yao_with_preps(3, "cswap q[0], q[1], q[2];", yao)
        end

        @testset "csx" begin
            # Controlled SX: use matblock of SX on the target.
            half = 1 / 2
            U_sx = half * [1 + im  1 - im; 1 - im  1 + im]
            yao = control(2, 1, 2 => matblock(U_sx))
            _assert_qasm_vs_yao_with_preps(2, "csx q[0], q[1];", yao)
        end

        @testset "crx" begin
            theta = pi / 3
            yao = control(2, 1, 2 => Rx(theta))
            _assert_qasm_vs_yao_with_preps(2, "crx($theta) q[0], q[1];", yao)
        end

        @testset "cry" begin
            theta = pi / 5
            yao = control(2, 1, 2 => Ry(theta))
            _assert_qasm_vs_yao_with_preps(2, "cry($theta) q[0], q[1];", yao)
        end

        @testset "cu" begin
            theta, phi, lambda, gamma = 0.4, 0.2, -0.1, 0.3
            # OpenQASM cu applies a control-phase gamma and controlled-U3.
            yao = chain(
                2,
                put(2, 1 => shift(gamma)),
                control(2, 1, 2 => matblock(_u3_matrix(theta, phi, lambda))),
            )
            _assert_qasm_vs_yao_with_preps(
                2,
                "cu($theta, $phi, $lambda, $gamma) q[0], q[1];",
                yao,
            )
        end
    end
end
