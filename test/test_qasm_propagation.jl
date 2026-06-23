using Test
using PauliPropagation

include("qasm_common.jl")

@testset "OpenQASM propagation equivalence" begin
    @testset "composite gates match totransfermap reference" begin
        @testset "sxdg" begin
            _assert_qasm_propagation_matches_transfermap(
                1,
                "sxdg q[0];",
                observables=[PauliString(1, sym, 1) for sym in (:I, :X, :Y, :Z)],
            )
        end

        @testset "cy" begin
            _assert_qasm_propagation_matches_transfermap(2, "cy q[0], q[1];")
        end

        @testset "ch" begin
            _assert_qasm_propagation_matches_transfermap(2, "ch q[0], q[1];")
        end

        @testset "ccx" begin
            _assert_qasm_propagation_matches_transfermap(3, "ccx q[0], q[1], q[2];")
        end

        @testset "cswap" begin
            _assert_qasm_propagation_matches_transfermap(3, "cswap q[0], q[1], q[2];")
        end

        @testset "csx" begin
            _assert_qasm_propagation_matches_transfermap(2, "csx q[0], q[1];")
        end

        @testset "crx" begin
            theta = pi / 3
            _assert_qasm_propagation_matches_transfermap(2, "crx($theta) q[0], q[1];")
        end

        @testset "cry" begin
            theta = pi / 5
            _assert_qasm_propagation_matches_transfermap(2, "cry($theta) q[0], q[1];")
        end

        @testset "crz" begin
            theta = pi / 7
            _assert_qasm_propagation_matches_transfermap(2, "crz($theta) q[0], q[1];")
        end

        @testset "cu1" begin
            lambda = pi / 4
            _assert_qasm_propagation_matches_transfermap(2, "cu1($lambda) q[0], q[1];")
        end

        @testset "cp" begin
            lambda = pi / 6
            _assert_qasm_propagation_matches_transfermap(2, "cp($lambda) q[0], q[1];")
        end

        @testset "cu3" begin
            theta, phi, lambda = 0.5, -0.3, 0.7
            _assert_qasm_propagation_matches_transfermap(2, "cu3($theta, $phi, $lambda) q[0], q[1];")
        end

        @testset "cu" begin
            theta, phi, lambda, gamma = 0.4, 0.2, -0.1, 0.3
            _assert_qasm_propagation_matches_transfermap(
                2,
                "cu($theta, $phi, $lambda, $gamma) q[0], q[1];",
            )
        end

        @testset "rzz" begin
            theta = pi / 6
            _assert_qasm_propagation_matches_transfermap(2, "rzz($theta) q[0], q[1];")
        end

        @testset "rzz pi/2 Clifford shortcut" begin
            _assert_qasm_propagation_matches_transfermap(2, "rzz($(pi / 2)) q[0], q[1];")
        end
    end

    @testset "Heisenberg roundtrip on composite decomposition" begin
        _, circuit, thetas = _readqasm_single_gate(2, "cry($(pi / 5)) q[0], q[1];")
        obs = PauliString(2, [:X, :Z], [1, 2])
        min_abs_coeff = 1e-12
        forward = propagate(circuit, obs, thetas; heisenberg=false, min_abs_coeff)
        roundtrip = propagate(circuit, forward, thetas; heisenberg=true, min_abs_coeff)
        @test roundtrip ≈ PauliSum(obs)
    end

    @testset "multi-gate QASM circuit" begin
        nq = 3
        qasm_content = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[$nq];
        rx(0.4) q[0];
        cx q[0], q[1];
        ccx q[0], q[1], q[2];
        """
        _, circuit_multi, thetas_multi = _readqasm_program(qasm_content)

        _, circuit_rx, thetas_rx = _readqasm_single_gate(nq, "rx(0.4) q[0];")
        _, circuit_cx, thetas_cx = _readqasm_single_gate(nq, "cx q[0], q[1];")
        _, circuit_ccx, thetas_ccx = _readqasm_single_gate(nq, "ccx q[0], q[1], q[2];")
        circuit_stitched = vcat(circuit_rx, circuit_cx, circuit_ccx)
        thetas_stitched = vcat(thetas_rx, thetas_cx, thetas_ccx)

        obs = PauliString(nq, [:Y, :Z, :X], [1, 2, 3])
        psum_multi = propagate(circuit_multi, obs, thetas_multi; min_abs_coeff=0)
        psum_stitched = propagate(circuit_stitched, obs, thetas_stitched; min_abs_coeff=0)
        @test psum_multi == psum_stitched
    end
end
