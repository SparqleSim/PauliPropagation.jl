using Test
using PauliPropagation
using PauliPropagation.OpenQASMInterface
import PauliPropagation: TransferMapGate, TGate

include("qasm_common.jl")

function _assert_clifford_mapping(nq::Int, gate_line::AbstractString, symbol::Symbol, qinds)
    nq_parsed, circuit, thetas = _readqasm_single_gate(nq, gate_line)
    @test nq_parsed == nq
    @test length(circuit) == 1
    @test isempty(thetas)
    @test only(circuit) isa CliffordGate
    @test only(circuit).symbol == symbol
    @test only(circuit).qinds == qinds
end

function _assert_pauli_rotation_mapping(
    nq::Int,
    gate_line::AbstractString,
    symbols,
    qinds,
    theta,
)
    nq_parsed, circuit, thetas = _readqasm_single_gate(nq, gate_line)
    @test nq_parsed == nq
    @test length(circuit) == 1
    @test length(thetas) == 1
    @test only(circuit) isa PauliRotation
    @test only(circuit).symbols == symbols
    @test only(circuit).qinds == qinds
    @test only(thetas) ≈ theta
end

function _assert_tgate_mapping(nq::Int, gate_line::AbstractString, qind::Int)
    nq_parsed, circuit, thetas = _readqasm_single_gate(nq, gate_line)
    @test nq_parsed == nq
    @test length(circuit) == 1
    @test isempty(thetas)
    @test only(circuit) isa TGate
    @test only(circuit).qind == qind
end

@testset "OpenQASM Interface" begin
    @testset "Native CliffordGate mappings" begin
        clifford_cases = [
            (1, "h q[0];", :H, [1]),
            (2, "x q[1];", :X, [2]),
            (3, "y q[2];", :Y, [3]),
            (1, "z q[0];", :Z, [1]),
            (2, "s q[1];", :S, [2]),
            (3, "sx q[2];", :SX, [3]),
            (2, "cx q[0], q[1];", :CNOT, [1, 2]),
            (3, "cx q[1], q[2];", :CNOT, [2, 3]),
            (3, "cz q[0], q[2];", :CZ, [1, 3]),
            (3, "swap q[1], q[2];", :SWAP, [2, 3]),
        ]
        for (nq, gate_line, symbol, qinds) in clifford_cases
            @testset "$gate_line" begin
                _assert_clifford_mapping(nq, gate_line, symbol, qinds)
            end
        end
    end

    @testset "Native PauliRotation mappings" begin
        rotation_cases = [
            (1, "rx(1.23) q[0];", [:X], [1], 1.23),
            (2, "ry($(pi / 2)) q[1];", [:Y], [2], pi / 2),
            (1, "rz($(pi / 4)) q[0];", [:Z], [1], pi / 4),
            (1, "p($(pi / 2)) q[0];", [:Z], [1], pi / 2),
            (1, "u1($(pi / 4)) q[0];", [:Z], [1], pi / 4),
            (3, "rxx($(pi / 3)) q[1], q[2];", [:X, :X], [2, 3], pi / 3),
            (2, "rzz($(pi / 4)) q[0], q[1];", [:Z, :Z], [1, 2], pi / 4),
            (1, "sdg q[0];", [:Z], [1], -pi / 2),
            (2, "tdg q[1];", [:Z], [2], -pi / 4),
        ]
        for (nq, gate_line, symbols, qinds, theta) in rotation_cases
            @testset "$gate_line" begin
                _assert_pauli_rotation_mapping(nq, gate_line, symbols, qinds, theta)
            end
        end
    end

    @testset "Native TGate mapping" begin
        _assert_tgate_mapping(2, "t q[1];", 2)
    end

    @testset "Qubit indexing and theta ordering" begin
        nq, circuit, thetas = _readqasm_program("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        rx(1.23) q[0];
        cx q[0], q[1];
        ry($(pi / 2)) q[2];
        """)

        @test nq == 3
        @test length(circuit) == 3
        @test length(thetas) == 2

        @test circuit[1] isa PauliRotation
        @test circuit[1].symbols == [:X]
        @test circuit[1].qinds == [1]
        @test thetas[1] ≈ 1.23

        @test circuit[2] isa CliffordGate
        @test circuit[2].symbol == :CNOT
        @test circuit[2].qinds == [1, 2]

        @test circuit[3] isa PauliRotation
        @test circuit[3].symbols == [:Y]
        @test circuit[3].qinds == [3]
        @test thetas[2] ≈ pi / 2
    end

    @testset "expression parameters" begin
        nq, circuit, thetas = _readqasm_program("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        ry(0.8*pi) q[3];
        u3(0.1, 0.2*pi, pi/2) q[0];
        """)

        @test nq == 4
        @test length(circuit) == 2
        @test length(thetas) == 1

        @test circuit[1] isa PauliRotation
        @test circuit[1].symbols == [:Y]
        @test circuit[1].qinds == [4]
        @test thetas[1] ≈ 0.8 * pi

        @test circuit[2] isa TransferMapGate
        @test circuit[2].qinds == [1]
    end

    @testset "Unsupported gates" begin
        unsupported_cases = [
            (1, "h q[0];\nreset q[0];"),
            (2, "creg c[4];\nh q[0];\nmeasure q[1] -> c[1];"),
            (1, "u0(1.0) q[0];"),
            (4, "c3x q[0], q[1], q[2], q[3];"),
            (4, "c3sqrtx q[0], q[1], q[2], q[3];"),
            (4, "rc3x q[0], q[1], q[2], q[3];"),
            (3, "rccx q[0], q[1], q[2];"),
        ]
        for (nq, body) in unsupported_cases
            @testset "$body" begin
                @test_throws ErrorException _readqasm_program("""
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[$nq];
                $body
                """)
            end
        end
    end

    @testset "Special cases" begin
        @testset "id is skipped" begin
            nq, circuit, thetas = _readqasm_program("""
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            id q[0];
            x q[1];
            id q[0];
            """)
            @test nq == 2
            @test length(circuit) == 1
            @test isempty(thetas)
            @test only(circuit) isa CliffordGate
            @test only(circuit).symbol == :X
            @test only(circuit).qinds == [2]
        end

        @testset "barrier is skipped" begin
            nq, circuit, thetas = _readqasm_program("""
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            barrier q[0];
            h q[0];
            """)
            @test nq == 1
            @test length(circuit) == 1
            @test isempty(thetas)
            @test only(circuit) isa CliffordGate
            @test only(circuit).symbol == :H
            @test only(circuit).qinds == [1]
        end

        @testset "rzz(pi/2) maps to ZZpihalf" begin
            _assert_clifford_mapping(2, "rzz($(pi / 2)) q[0], q[1];", :ZZpihalf, [1, 2])
        end
    end

    @testset "u3 gate support" begin
        nq_u3, circuit_u3, thetas_u3 = _readqasm_single_gate(1, "u3(0.1, 0.2, 0.3) q[0];")
        @test nq_u3 == 1
        @test length(circuit_u3) == 1
        @test isempty(thetas_u3)
        @test only(circuit_u3) isa TransferMapGate
        @test only(circuit_u3).qinds == [1]

        nq_mixed, circuit_mixed, thetas_mixed = _readqasm_program("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        u3($(pi / 2), $(pi / 4), $(pi / 8)) q[1];
        cx q[0], q[1];
        """)
        @test nq_mixed == 2
        @test length(circuit_mixed) == 3
        @test isempty(thetas_mixed)
        @test circuit_mixed[1] isa CliffordGate && circuit_mixed[1].symbol == :H
        @test circuit_mixed[2] isa TransferMapGate && circuit_mixed[2].qinds == [2]
        @test circuit_mixed[3] isa CliffordGate && circuit_mixed[3].symbol == :CNOT
    end

    @testset "u2 gate support" begin
        nq_u2, circuit_u2, thetas_u2 = _readqasm_single_gate(1, "u2(0.1, 0.2) q[0];")
        @test nq_u2 == 1
        @test length(circuit_u2) == 1
        @test isempty(thetas_u2)
        @test only(circuit_u2) isa TransferMapGate
        @test only(circuit_u2).qinds == [1]

        nq_u2_mixed, circuit_u2_mixed, thetas_u2_mixed = _readqasm_program("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        u2(0.5, -0.3) q[1];
        cx q[0], q[1];
        """)
        @test nq_u2_mixed == 2
        @test length(circuit_u2_mixed) == 3
        @test isempty(thetas_u2_mixed)
        @test circuit_u2_mixed[1] isa CliffordGate && circuit_u2_mixed[1].symbol == :H
        @test circuit_u2_mixed[2] isa TransferMapGate && circuit_u2_mixed[2].qinds == [2]
        @test circuit_u2_mixed[3] isa CliffordGate && circuit_u2_mixed[3].symbol == :CNOT
    end
end
