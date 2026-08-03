using Test
using PauliPropagation

include("qasm_common.jl")

@testset "OpenQASM propagation integration" begin
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
