using Test
using PauliPropagation.OpenQASMInterface

@testset "OpenQASM Interface" begin
    # Define the content for a temporary QASM file
    qasm_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    rx(1.23) q[0];
    cx q[0], q[1];
    """

    # Write the content to a file in the current directory
    test_filepath = "test_circ.qasm"
    write(test_filepath, qasm_content)

    # Call your new function to parse the file
    nq, circuit, thetas = readqasm(test_filepath)

    # --- Assertions: Check if the output is correct ---
    @test nq == 2
    @test length(circuit) == 2
    @test length(thetas) == 1

    # Check the first gate (rx)
    @test circuit[1] isa PauliRotation
    @test circuit[1].symbols[1] == :X    # CORRECT FIELD: .symbols, not .gate_type
    @test circuit[1].qinds[1] == 1        # CORRECT FIELD: .qinds, not .qubit
    @test thetas[1] ≈ 1.23

    # Check the second gate (cx)
    @test circuit[2] isa CliffordGate
    @test circuit[2].symbol == :CNOT      # CORRECT FIELD: .symbol, not .gate_type
    @test circuit[2].qinds == [1, 2]      # CORRECT FIELD: .qinds, not .qubits

    # Clean up the temporary file
    rm(test_filepath)

    # Test for unsupported gates
    unsupported_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    u3(0.1,0.2,0.3) q[0];
    """
    unsupported_filepath = "unsupported.qasm"
    write(unsupported_filepath, unsupported_content)

    @test_throws ErrorException readqasm(unsupported_filepath)

    rm(unsupported_filepath)
end