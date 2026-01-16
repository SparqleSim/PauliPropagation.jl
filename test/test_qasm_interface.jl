using Test
using PauliPropagation
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

    # Test for reset gate (not supported - non-unitary operation)
    reset_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    h q[0];
    reset q[0];
    """
    reset_filepath = "reset_test.qasm"
    write(reset_filepath, reset_content)
    @test_throws ErrorException readqasm(reset_filepath)
    rm(reset_filepath)

    # Test for measure gate (not supported - non-unitary operation)
    measure_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[4];
    h q[0];
    measure q[1] -> c[1];
    """
    measure_filepath = "measure_test.qasm"
    write(measure_filepath, measure_content)
    @test_throws ErrorException readqasm(measure_filepath)
    rm(measure_filepath)

    # Test for u0, u1, u2 gates (not supported)
    u0_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    u0(1.0) q[0];
    """
    u0_filepath = "u0_test.qasm"
    write(u0_filepath, u0_content)
    @test_throws ErrorException readqasm(u0_filepath)
    rm(u0_filepath)

    u1_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    u1(0.5) q[0];
    """
    u1_filepath = "u1_test.qasm"
    write(u1_filepath, u1_content)
    @test_throws ErrorException readqasm(u1_filepath)
    rm(u1_filepath)

    u2_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    u2(0.5, 0.3) q[0];
    """
    u2_filepath = "u2_test.qasm"
    write(u2_filepath, u2_content)
    @test_throws ErrorException readqasm(u2_filepath)
    rm(u2_filepath)

    u_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    u(0.1, 0.2, 0.3) q[0];
    """
    u_filepath = "u_test.qasm"
    write(u_filepath, u_content)
    @test_throws ErrorException readqasm(u_filepath)
    rm(u_filepath)

    # cu3 is not supported (depends on general u3 gate)
    cu3_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    cu3(0.1, 0.2, 0.3) q[0], q[1];
    """
    cu3_filepath = "cu3_test.qasm"
    write(cu3_filepath, cu3_content)
    @test_throws ErrorException readqasm(cu3_filepath)
    rm(cu3_filepath)

    # cu is not supported (depends on general u gate)
    cu_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    cu(0.1, 0.2, 0.3, 0.4) q[0], q[1];
    """
    cu_filepath = "cu_test.qasm"
    write(cu_filepath, cu_content)
    @test_throws ErrorException readqasm(cu_filepath)
    rm(cu_filepath)

    # Test composite gates: sxdg, cy, ch, ccx
    sxdg_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    sxdg q[0];
    """
    sxdg_filepath = "sxdg_test.qasm"
    write(sxdg_filepath, sxdg_content)
    nq_sxdg, circuit_sxdg, thetas_sxdg = readqasm(sxdg_filepath)
    @test nq_sxdg == 1
    @test length(circuit_sxdg) == 3  # s, h, s
    @test length(thetas_sxdg) == 0
    @test circuit_sxdg[1] isa CliffordGate && circuit_sxdg[1].symbol == :S
    @test circuit_sxdg[2] isa CliffordGate && circuit_sxdg[2].symbol == :H
    @test circuit_sxdg[3] isa CliffordGate && circuit_sxdg[3].symbol == :S
    rm(sxdg_filepath)

    cy_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    cy q[0], q[1];
    """
    cy_filepath = "cy_test.qasm"
    write(cy_filepath, cy_content)
    nq_cy, circuit_cy, thetas_cy = readqasm(cy_filepath)
    @test nq_cy == 2
    @test length(circuit_cy) == 3  # sdg, cx, s
    @test length(thetas_cy) == 1  # only sdg has theta
    @test circuit_cy[1] isa PauliRotation && circuit_cy[1].symbols == [:Z]
    @test circuit_cy[2] isa CliffordGate && circuit_cy[2].symbol == :CNOT
    @test circuit_cy[3] isa CliffordGate && circuit_cy[3].symbol == :S
    @test thetas_cy[1] ≈ -pi/2
    rm(cy_filepath)

    ch_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    ch q[0], q[1];
    """
    ch_filepath = "ch_test.qasm"
    write(ch_filepath, ch_content)
    nq_ch, circuit_ch, thetas_ch = readqasm(ch_filepath)
    @test nq_ch == 2
    @test length(circuit_ch) == 11
    @test length(thetas_ch) == 1  # only sdg has theta
    rm(ch_filepath)

    ccx_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    ccx q[0], q[1], q[2];
    """
    ccx_filepath = "ccx_test.qasm"
    write(ccx_filepath, ccx_content)
    nq_ccx, circuit_ccx, thetas_ccx = readqasm(ccx_filepath)
    @test nq_ccx == 3
    @test length(circuit_ccx) == 15
    @test length(thetas_ccx) == 3  # three tdg gates
    @test all(t ≈ -pi/4 for t in thetas_ccx)
    rm(ccx_filepath)

    # Test cswap (controlled-SWAP / Fredkin gate)
    cswap_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    cswap q[0], q[1], q[2];
    """
    cswap_filepath = "cswap_test.qasm"
    write(cswap_filepath, cswap_content)
    nq_cswap, circuit_cswap, thetas_cswap = readqasm(cswap_filepath)
    @test nq_cswap == 3
    @test length(circuit_cswap) == 17  # cx + ccx(15 gates) + cx
    @test length(thetas_cswap) == 3  # three tdg gates from ccx
    @test all(t ≈ -pi/4 for t in thetas_cswap)
    rm(cswap_filepath)

    # Test crx (controlled RX rotation)
    crx_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    crx(1.5707963267948966) q[0], q[1];
    """
    crx_filepath = "crx_test.qasm"
    write(crx_filepath, crx_content)
    nq_crx, circuit_crx, thetas_crx = readqasm(crx_filepath)
    @test nq_crx == 2
    @test length(circuit_crx) == 5  # rz, cx, ry, cx, rx
    @test length(thetas_crx) == 3  # pi/2, -lambda/2, lambda/2
    @test thetas_crx[1] ≈ pi/2
    @test thetas_crx[2] ≈ -1.5707963267948966/2  # -lambda/2
    @test thetas_crx[3] ≈ 1.5707963267948966/2   # lambda/2
    @test circuit_crx[1] isa PauliRotation && circuit_crx[1].symbols == [:Z]
    @test circuit_crx[2] isa CliffordGate && circuit_crx[2].symbol == :CNOT
    @test circuit_crx[3] isa PauliRotation && circuit_crx[3].symbols == [:Y]
    @test circuit_crx[4] isa CliffordGate && circuit_crx[4].symbol == :CNOT
    @test circuit_crx[5] isa PauliRotation && circuit_crx[5].symbols == [:X]
    rm(crx_filepath)

    # Test cry (controlled RY rotation)
    cry_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    cry(1.5707963267948966) q[0], q[1];
    """
    cry_filepath = "cry_test.qasm"
    write(cry_filepath, cry_content)
    nq_cry, circuit_cry, thetas_cry = readqasm(cry_filepath)
    @test nq_cry == 2
    @test length(circuit_cry) == 4  # ry, cx, ry, cx
    @test length(thetas_cry) == 2  # lambda/2, -lambda/2
    @test thetas_cry[1] ≈ 1.5707963267948966/2
    @test thetas_cry[2] ≈ -1.5707963267948966/2
    @test circuit_cry[1] isa PauliRotation && circuit_cry[1].symbols == [:Y]
    @test circuit_cry[2] isa CliffordGate && circuit_cry[2].symbol == :CNOT
    @test circuit_cry[3] isa PauliRotation && circuit_cry[3].symbols == [:Y]
    @test circuit_cry[4] isa CliffordGate && circuit_cry[4].symbol == :CNOT
    rm(cry_filepath)

    # Test crz (controlled RZ rotation)
    crz_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    crz(1.5707963267948966) q[0], q[1];
    """
    crz_filepath = "crz_test.qasm"
    write(crz_filepath, crz_content)
    nq_crz, circuit_crz2, thetas_crz2 = readqasm(crz_filepath)
    @test nq_crz == 2
    @test length(circuit_crz2) == 4  # rz, cx, rz, cx
    @test length(thetas_crz2) == 2  # lambda/2, -lambda/2
    @test thetas_crz2[1] ≈ 1.5707963267948966/2
    @test thetas_crz2[2] ≈ -1.5707963267948966/2
    @test circuit_crz2[1] isa PauliRotation && circuit_crz2[1].symbols == [:Z]
    @test circuit_crz2[2] isa CliffordGate && circuit_crz2[2].symbol == :CNOT
    @test circuit_crz2[3] isa PauliRotation && circuit_crz2[3].symbols == [:Z]
    @test circuit_crz2[4] isa CliffordGate && circuit_crz2[4].symbol == :CNOT
    rm(crz_filepath)

    # Test cu1 (controlled phase via u1)
    cu1_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    cu1(1.5707963267948966) q[0], q[1];
    """
    cu1_filepath = "cu1_test.qasm"
    write(cu1_filepath, cu1_content)
    nq_cu1, circuit_cu1, thetas_cu1 = readqasm(cu1_filepath)
    @test nq_cu1 == 2
    @test length(circuit_cu1) == 5  # rz(a), cx, rz(b), cx, rz(b)
    @test length(thetas_cu1) == 3  # lambda/2, -lambda/2, lambda/2
    @test thetas_cu1[1] ≈ 1.5707963267948966/2
    @test thetas_cu1[2] ≈ -1.5707963267948966/2
    @test thetas_cu1[3] ≈ 1.5707963267948966/2
    @test circuit_cu1[1] isa PauliRotation && circuit_cu1[1].symbols == [:Z] && circuit_cu1[1].qinds == [1]
    @test circuit_cu1[2] isa CliffordGate && circuit_cu1[2].symbol == :CNOT
    @test circuit_cu1[3] isa PauliRotation && circuit_cu1[3].symbols == [:Z] && circuit_cu1[3].qinds == [2]
    @test circuit_cu1[4] isa CliffordGate && circuit_cu1[4].symbol == :CNOT
    @test circuit_cu1[5] isa PauliRotation && circuit_cu1[5].symbols == [:Z] && circuit_cu1[5].qinds == [2]
    rm(cu1_filepath)

    # Test cp (controlled phase via p)
    cp_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    cp(1.5707963267948966) q[0], q[1];
    """
    cp_filepath = "cp_test.qasm"
    write(cp_filepath, cp_content)
    nq_cp, circuit_cp, thetas_cp = readqasm(cp_filepath)
    @test nq_cp == 2
    @test length(circuit_cp) == 5  # same pattern as cu1
    @test length(thetas_cp) == 3  # lambda/2, -lambda/2, lambda/2
    @test thetas_cp[1] ≈ 1.5707963267948966/2
    @test thetas_cp[2] ≈ -1.5707963267948966/2
    @test thetas_cp[3] ≈ 1.5707963267948966/2
    @test circuit_cp[1] isa PauliRotation && circuit_cp[1].symbols == [:Z] && circuit_cp[1].qinds == [1]
    @test circuit_cp[2] isa CliffordGate && circuit_cp[2].symbol == :CNOT
    @test circuit_cp[3] isa PauliRotation && circuit_cp[3].symbols == [:Z] && circuit_cp[3].qinds == [2]
    @test circuit_cp[4] isa CliffordGate && circuit_cp[4].symbol == :CNOT
    @test circuit_cp[5] isa PauliRotation && circuit_cp[5].symbols == [:Z] && circuit_cp[5].qinds == [2]
    rm(cp_filepath)

    # Test rccx gate (relative-phase Toffoli)
    rccx_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    rccx q[0], q[1], q[2];
    """
    rccx_filepath = "rccx_test_circ.qasm"
    write(rccx_filepath, rccx_content)

    nq_rccx, circuit_rccx, thetas_rccx = readqasm(rccx_filepath)

    @test nq_rccx == 3
    @test length(circuit_rccx) == 9   # h, t, cx, tdag, cx, t, cx, tdag, h
    @test length(thetas_rccx) == 2    # two -pi/4 rotations
    @test all(t ≈ -pi/4 for t in thetas_rccx)

    rm(rccx_filepath)

    # Test rc3x gate (relative-phase 3-controlled X)
    rc3x_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    rc3x q[0], q[1], q[2], q[3];
    """
    rc3x_filepath = "rc3x_test_circ.qasm"
    write(rc3x_filepath, rc3x_content)

    nq_rc3x, circuit_rc3x, thetas_rc3x = readqasm(rc3x_filepath)

    @test nq_rc3x == 4
    @test length(circuit_rc3x) == 18   # sequence from definition
    @test length(thetas_rc3x) == 4     # four -pi/4 rotations
    @test all(t ≈ -pi/4 for t in thetas_rc3x)

    rm(rc3x_filepath)

    # c3x is not supported (requires non-relative-phase decomposition)
    c3x_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    c3x q[0], q[1], q[2], q[3];
    """
    c3x_filepath = "c3x_test.qasm"
    write(c3x_filepath, c3x_content)
    @test_throws ErrorException readqasm(c3x_filepath)
    rm(c3x_filepath)

    # c3sqrtx is not supported (built from unsupported c3x)
    c3sqrtx_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    c3sqrtx q[0], q[1], q[2], q[3];
    """
    c3sqrtx_filepath = "c3sqrtx_test.qasm"
    write(c3sqrtx_filepath, c3sqrtx_content)
    @test_throws ErrorException readqasm(c3sqrtx_filepath)
    rm(c3sqrtx_filepath)

    # Test csx gate (controlled sqrt(X))
    csx_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    csx q[0], q[1];
    """
    csx_filepath = "csx_test_circ.qasm"
    write(csx_filepath, csx_content)

    nq_csx, circuit_csx, thetas_csx = readqasm(csx_filepath)

    @test nq_csx == 2
    @test length(circuit_csx) == 7  # h, rz(a), cx, rz(b), cx, rz(b), h
    @test length(thetas_csx) == 3   # lambda/2, -lambda/2, lambda/2 with lambda = pi/2

    # h b
    @test circuit_csx[1] isa CliffordGate
    @test circuit_csx[1].symbol == :H
    @test circuit_csx[1].qinds == [2]

    # rz(pi/4) a
    @test circuit_csx[2] isa PauliRotation
    @test circuit_csx[2].symbols == [:Z]
    @test circuit_csx[2].qinds == [1]
    @test thetas_csx[1] ≈ pi/4

    # cx a,b
    @test circuit_csx[3] isa CliffordGate
    @test circuit_csx[3].symbol == :CNOT
    @test circuit_csx[3].qinds == [1, 2]

    # rz(-pi/4) b
    @test circuit_csx[4] isa PauliRotation
    @test circuit_csx[4].symbols == [:Z]
    @test circuit_csx[4].qinds == [2]
    @test thetas_csx[2] ≈ -pi/4

    # cx a,b
    @test circuit_csx[5] isa CliffordGate
    @test circuit_csx[5].symbol == :CNOT
    @test circuit_csx[5].qinds == [1, 2]

    # rz(pi/4) b
    @test circuit_csx[6] isa PauliRotation
    @test circuit_csx[6].symbols == [:Z]
    @test circuit_csx[6].qinds == [2]
    @test thetas_csx[3] ≈ pi/4

    # h b
    @test circuit_csx[7] isa CliffordGate
    @test circuit_csx[7].symbol == :H
    @test circuit_csx[7].qinds == [2]

    rm(csx_filepath)

    
    # Extended test: larger circuit using additional supported gates
    
    extended_qasm_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    h q[0];
    x q[1];
    y q[2];
    cx q[0], q[1];
    cx q[1], q[2];
    s q[1];
    sx q[2];
    cz q[0], q[2];
    swap q[1], q[2];
    rzz(1.5707963267948966) q[0], q[1];
    """

    extended_filepath = "extended_test_circ.qasm"
    write(extended_filepath, extended_qasm_content)

    nq_ext, circuit_ext, thetas_ext = readqasm(extended_filepath)

    @test nq_ext == 3
    @test length(circuit_ext) == 10
    @test length(thetas_ext) == 0  # rzz(π/2) becomes ZZpihalf Clifford gate, so no theta

    # h q[0]
    @test circuit_ext[1] isa CliffordGate
    @test circuit_ext[1].symbol == :H
    @test circuit_ext[1].qinds == [1]

    # x q[1]
    @test circuit_ext[2] isa CliffordGate
    @test circuit_ext[2].symbol == :X
    @test circuit_ext[2].qinds == [2]

    # y q[2]
    @test circuit_ext[3] isa CliffordGate
    @test circuit_ext[3].symbol == :Y
    @test circuit_ext[3].qinds == [3]

    # cx q[0], q[1]
    @test circuit_ext[4] isa CliffordGate
    @test circuit_ext[4].symbol == :CNOT
    @test circuit_ext[4].qinds == [1, 2]

    # cx q[1], q[2]
    @test circuit_ext[5] isa CliffordGate
    @test circuit_ext[5].symbol == :CNOT
    @test circuit_ext[5].qinds == [2, 3]

    # s q[1]
    @test circuit_ext[6] isa CliffordGate
    @test circuit_ext[6].symbol == :S
    @test circuit_ext[6].qinds == [2]

    # sx q[2]
    @test circuit_ext[7] isa CliffordGate
    @test circuit_ext[7].symbol == :SX
    @test circuit_ext[7].qinds == [3]

    # cz q[0], q[2]
    @test circuit_ext[8] isa CliffordGate
    @test circuit_ext[8].symbol == :CZ
    @test circuit_ext[8].qinds == [1, 3]

    # swap q[1], q[2]
    @test circuit_ext[9] isa CliffordGate
    @test circuit_ext[9].symbol == :SWAP
    @test circuit_ext[9].qinds == [2, 3]

    # rzz(1.5708...) q[0], q[1] - should be ZZpihalf Clifford gate since angle is π/2
    @test circuit_ext[10] isa CliffordGate
    @test circuit_ext[10].symbol == :ZZpihalf
    @test circuit_ext[10].qinds == [1, 2]
    @test length(thetas_ext) == 0  # No theta since it's a Clifford gate

    rm(extended_filepath)

    rotations_qasm_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    ry(1.5707963267948966) q[1];
    rx(0.39269908169872414) q[0];
    rxx(1.0471975511965976) q[1], q[2];
    rzz(0.7853981633974483) q[0], q[1];
    """

    rotations_filepath = "rotations_test_circ.qasm"
    write(rotations_filepath, rotations_qasm_content)

    nq_rot, circuit_rot, thetas_rot = readqasm(rotations_filepath)

    @test nq_rot == 3
    @test length(circuit_rot) == 4
    @test length(thetas_rot) == 4

    # ry(pi/2) q[1]
    @test circuit_rot[1] isa PauliRotation
    @test circuit_rot[1].symbols == [:Y]
    @test circuit_rot[1].qinds == [2]
    @test thetas_rot[1] ≈ 1.5707963267948966

    # rx(pi/8) q[0]
    @test circuit_rot[2] isa PauliRotation
    @test circuit_rot[2].symbols == [:X]
    @test circuit_rot[2].qinds == [1]
    @test thetas_rot[2] ≈ 0.39269908169872414

    # rxx(pi/3) q[1], q[2]
    @test circuit_rot[3] isa PauliRotation
    @test circuit_rot[3].symbols == [:X, :X]
    @test circuit_rot[3].qinds == [2, 3]
    @test thetas_rot[3] ≈ 1.0471975511965976

    # rzz(pi/4) q[0], q[1]
    @test circuit_rot[4] isa PauliRotation
    @test circuit_rot[4].symbols == [:Z, :Z]
    @test circuit_rot[4].qinds == [1, 2]
    @test thetas_rot[4] ≈ 0.7853981633974483

    rm(rotations_filepath)

    # Test z and t gates
    zt_qasm_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    z q[0];
    t q[1];
    """

    zt_filepath = "zt_test_circ.qasm"
    write(zt_filepath, zt_qasm_content)

    nq_zt, circuit_zt, thetas_zt = readqasm(zt_filepath)

    @test nq_zt == 3
    @test length(circuit_zt) == 2
    @test length(thetas_zt) == 0  # Both z and t are static gates

    # z q[0]
    @test circuit_zt[1] isa CliffordGate
    @test circuit_zt[1].symbol == :Z
    @test circuit_zt[1].qinds == [1]

    # t q[1]
    @test circuit_zt[2] isa TGate
    @test circuit_zt[2].qind == 2

    rm(zt_filepath)

    id_qasm_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    id q[0];
    x q[1];
    id q[0];
    """
    id_filepath = "id_test_circ.qasm"
    write(id_filepath, id_qasm_content)

    nq_id, circuit_id, thetas_id = readqasm(id_filepath)

    @test nq_id == 2
    @test length(circuit_id) == 1  # Only x gate, id gates are ignored
    @test length(thetas_id) == 0

    # Only the x gate should be present
    @test circuit_id[1] isa CliffordGate
    @test circuit_id[1].symbol == :X
    @test circuit_id[1].qinds == [2]

    rm(id_filepath)

    # Test barrier gate (should be completely ignored)
    barrier_qasm_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    barrier q[0];
    h q[0];
    """
    barrier_filepath = "barrier_test_circ.qasm"
    write(barrier_filepath, barrier_qasm_content)

    nq_barrier, circuit_barrier, thetas_barrier = readqasm(barrier_filepath)

    @test nq_barrier == 1
    @test length(circuit_barrier) == 1  # Only h gate, barrier is ignored
    @test length(thetas_barrier) == 0

    # Only the h gate should be present
    @test circuit_barrier[1] isa CliffordGate
    @test circuit_barrier[1].symbol == :H
    @test circuit_barrier[1].qinds == [1]

    rm(barrier_filepath)

    rz_qasm_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    rz(0.7853981633974483) q[0];
    x q[1];
    """
    rz_filepath = "rz_test_circ.qasm"
    write(rz_filepath, rz_qasm_content)

    nq_rz, circuit_rz, thetas_rz = readqasm(rz_filepath)

    @test nq_rz == 2
    @test length(circuit_rz) == 2
    @test length(thetas_rz) == 1

    # rz(pi/4) q[0]
    @test circuit_rz[1] isa PauliRotation
    @test circuit_rz[1].symbols == [:Z]
    @test circuit_rz[1].qinds == [1]
    @test thetas_rz[1] ≈ 0.7853981633974483

    # x q[1]
    @test circuit_rz[2] isa CliffordGate
    @test circuit_rz[2].symbol == :X
    @test circuit_rz[2].qinds == [2]

    rm(rz_filepath)

    # Test p gate (phase gate - equivalent to rz)
    p_qasm_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    p(1.5707963267948966) q[0];
    x q[1];
    """
    p_filepath = "p_test_circ.qasm"
    write(p_filepath, p_qasm_content)

    nq_p, circuit_p, thetas_p = readqasm(p_filepath)

    @test nq_p == 2
    @test length(circuit_p) == 2
    @test length(thetas_p) == 1

    # p(pi/2) q[0] - should be equivalent to rz(pi/2)
    @test circuit_p[1] isa PauliRotation
    @test circuit_p[1].symbols == [:Z]
    @test circuit_p[1].qinds == [1]
    @test thetas_p[1] ≈ 1.5707963267948966

    # x q[1]
    @test circuit_p[2] isa CliffordGate
    @test circuit_p[2].symbol == :X
    @test circuit_p[2].qinds == [2]

    rm(p_filepath)

    # Test sdg and tdg gates (conjugate gates)
    sdg_tdg_qasm_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    sdg q[0];
    tdg q[1];
    """
    sdg_tdg_filepath = "sdg_tdg_test_circ.qasm"
    write(sdg_tdg_filepath, sdg_tdg_qasm_content)

    nq_sdg_tdg, circuit_sdg_tdg, thetas_sdg_tdg = readqasm(sdg_tdg_filepath)

    @test nq_sdg_tdg == 2
    @test length(circuit_sdg_tdg) == 2
    @test length(thetas_sdg_tdg) == 2

    # sdg q[0] - should be RZ(-pi/2)
    @test circuit_sdg_tdg[1] isa PauliRotation
    @test circuit_sdg_tdg[1].symbols == [:Z]
    @test circuit_sdg_tdg[1].qinds == [1]
    @test thetas_sdg_tdg[1] ≈ -pi/2

    # tdg q[1] - should be RZ(-pi/4)
    @test circuit_sdg_tdg[2] isa PauliRotation
    @test circuit_sdg_tdg[2].symbols == [:Z]
    @test circuit_sdg_tdg[2].qinds == [2]
    @test thetas_sdg_tdg[2] ≈ -pi/4

    rm(sdg_tdg_filepath)
end