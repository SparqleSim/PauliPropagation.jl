using Test
using LinearAlgebra
using PauliPropagation
using PauliPropagation.OpenQASMInterface
import PauliPropagation: TransferMapGate, ParametrizedGate, TGate, tomatrix

# Embed a gate's local unitary into full 2^nq × 2^nq space.
# Qubit ordering for the big system matches NumPy's kron convention:
# basis |q1 q2 … q_nq⟩ with q1 the most-significant bit, and
# index = 1 + ∑_{j=1}^{nq} b_j * 2^(nq-j), where b_j is the bit for qubit j.
function _embed_gate_matrix(nq::Int, qinds::Vector{Int}, U_local::AbstractMatrix)
    dim = size(U_local, 1)
    k = length(qinds)
    @assert dim == 2^k

    # If the gate already acts on all qubits, don't re-embed: keep the local 2^nq×2^nq matrix as-is.
    if k == nq && sort(qinds) == collect(1:nq)
        return Matrix{ComplexF64}(U_local)
    end

    # Use sorted physical qubits for the local basis; tomatrix(gate) encodes
    # control/target orientation via gate.qinds, while the embedding only
    # decides where in the big system the k-qubit block sits.
    q_sorted = sort(qinds)
    other_qubits = sort(setdiff(1:nq, q_sorted))
    U_full = zeros(ComplexF64, 2^nq, 2^nq)

    for r in 1:(2^nq)
        rr = r - 1
        for c in 1:(2^nq)
            cc = c - 1
            # Check other qubits match (same bits on all qubits not in q_sorted)
            match = true
            for j in other_qubits
                big_bit = nq - j
                if ((rr >> big_bit) & 1) != ((cc >> big_bit) & 1)
                    match = false
                    break
                end
            end
            match || continue

            # Build local row/col indices in the sorted-wire basis.
            gate_r = 1
            gate_c = 1
            for p in 1:k
                j = q_sorted[p]
                big_bit = nq - j          # bit position in the global index
                local_bit = k - p         # p=1 → high bit, p=k → low bit
                gate_r += ((rr >> big_bit) & 1) * (1 << local_bit)
                gate_c += ((cc >> big_bit) & 1) * (1 << local_bit)
            end
            U_full[r, c] = U_local[gate_r, gate_c]
        end
    end
    return U_full
end

@testset "embed_gate_matrix X on 2 qubits" begin
    # Local X on a single qubit
    X = [0.0 1.0; 1.0 0.0]
    I2 = Matrix{Float64}(I, 2, 2)

    # X acting on qubit 1 vs qubit 2, matching NumPy's kron(X, I) / kron(I, X) convention:
    U_X_on_1 = _embed_gate_matrix(2, [1], X)  # X on first (left) qubit
    U_X_on_2 = _embed_gate_matrix(2, [2], X)  # X on second (right) qubit

    @test U_X_on_1 ≈ kron(X, I2)
    @test U_X_on_2 ≈ kron(I2, X)
end

"""
    circuit_to_matrix(nq, circuit, thetas)

Compute the full 2^nq × 2^nq unitary for a circuit. Gates are applied in order (first gate first).
Parametrized gates consume one element of `thetas` in order.
"""
function circuit_to_matrix(nq::Int, circuit, thetas)
    theta_idx = 1
    U_total = Matrix{ComplexF64}(I, 2^nq, 2^nq)
    for gate in circuit
        if gate isa ParametrizedGate
            theta = thetas[theta_idx]
            theta_idx += 1
            U_local = tomatrix(gate, theta)
        else
            U_local = tomatrix(gate)
        end
        qinds = gate isa TGate ? [gate.qind] : gate.qinds
        U_full = _embed_gate_matrix(nq, qinds, U_local)
        U_total = U_full * U_total
    end
    return U_total
end

# Convenience helper: build the unitary for a single-gate OpenQASM program.
function _single_gate_unitary(nq::Int, gate_line::AbstractString)
    qasm_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[$nq];
    $gate_line
    """
    filepath = "single_gate_matrix_test.qasm"
    write(filepath, qasm_content)
    nq_parsed, circuit, thetas = readqasm(filepath)
    U = circuit_to_matrix(nq_parsed, circuit, thetas)
    rm(filepath)
    return U
end

# Compare two unitaries up to global phase (and optional tolerance).
function _approx_equal_up_to_phase(U::AbstractMatrix, V::AbstractMatrix; atol=1e-10)
    idx = findfirst(!iszero, U)
    idx === nothing && return iszero(V)
    phase = V[idx] / U[idx]
    return isapprox(U * phase, V; atol=atol)
end

# Reference 2×2 Pauli matrices (computational basis).
_reference_x_2x2() = [0.0 1.0; 1.0 0.0]
_reference_y_2x2() = [0.0 -1.0im; 1.0im 0.0]
_reference_z_2x2() = [1.0 0.0; 0.0 -1.0]

function _reference_h_2x2()
    invsqrt2 = 1 / sqrt(2)
    return invsqrt2 * [1.0 1.0; 1.0 -1.0]
end
_reference_s_2x2() = [1.0 0.0; 0.0 1.0im]
_reference_sdg_2x2() = [1.0 0.0; 0.0 -1.0im]
function _reference_sx_2x2()
    half = 1 / 2
    return half * [1.0 + 1.0im   1.0 - 1.0im;
                   1.0 - 1.0im   1.0 + 1.0im]
end
_reference_sxdg_2x2() = _reference_sx_2x2()'

_reference_t_2x2() = [1.0 0.0; 0.0 exp(1.0im * pi / 4)]
_reference_tdg_2x2() = [1.0 0.0; 0.0 exp(-1.0im * pi / 4)]
function _reference_rx_2x2(theta::Real)
    c, s = cos(theta / 2), sin(theta / 2)
    return [c -1.0im*s; -1.0im*s c]
end
function _reference_ry_2x2(theta::Real)
    c, s = cos(theta / 2), sin(theta / 2)
    return [c -s; s c]
end
function _reference_rz_2x2(theta::Real)
    return [exp(-1.0im * theta / 2) 0.0; 0.0 exp(1.0im * theta / 2)]
end

function _reference_u3_2x2(theta::Real, phi::Real, lambda::Real)
    cos_half_theta = cos(theta / 2)
    sin_half_theta = sin(theta / 2)
    exp_i_lambda = exp(1.0im * lambda)
    exp_i_phi = exp(1.0im * phi)
    exp_i_phi_lambda = exp(1.0im * (phi + lambda))
    return [
        cos_half_theta                 -exp_i_lambda * sin_half_theta;
        exp_i_phi * sin_half_theta      exp_i_phi_lambda * cos_half_theta
    ]
end

_reference_u2_2x2(phi::Real, lambda::Real) = _reference_u3_2x2(pi/2, phi, lambda)
_reference_u1_2x2(lambda::Real) = _reference_p_2x2(lambda)

function _reference_p_2x2(lambda::Real)
    return [
        1.0               0.0;
        0.0   exp(1.0im * lambda)
    ]
end

function _reference_cx()
    return [
        1.0  0.0  0.0  0.0;
        0.0  1.0  0.0  0.0;
        0.0  0.0  0.0  1.0;
        0.0  0.0  1.0  0.0
    ]
end

function _reference_cx_swapped()
    return [
        1.0  0.0  0.0  0.0;
        0.0  0.0  0.0  1.0;
        0.0  0.0  1.0  0.0;
        0.0  1.0  0.0  0.0
    ]
end

function _reference_cz()
    return [
        1.0  0.0  0.0  0.0;
        0.0  1.0  0.0  0.0;
        0.0  0.0  1.0  0.0;
        0.0  0.0  0.0 -1.0
    ]
end

function _reference_cy()
    return [
        1.0  0.0   0.0   0.0;
        0.0  1.0   0.0   0.0;
        0.0  0.0   0.0  -1.0im;
        0.0  0.0   1.0im 0.0
    ]
end

function _reference_cy_swapped()
    # CY with control = qubit 2, target = qubit 1
    S = _reference_swap()
    return S * _reference_cy() * S
end

function _reference_swap()
    return [
        1.0  0.0  0.0  0.0;
        0.0  0.0  1.0  0.0;
        0.0  1.0  0.0  0.0;
        0.0  0.0  0.0  1.0
    ]
end

function _reference_ccx()
    return [
        1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
        0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0;
        0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0;
        0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0;
        0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0;
        0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0;
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0;
        0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0
    ]
end

function _reference_ccx_target2()
    return [
        1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
        0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0;
        0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0;
        0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0;
        0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0;
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0;
        0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0;
        0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0
    ]
end

function _reference_crx(theta::Real)
    R = _reference_rx_2x2(theta)
    return [
        1.0  0.0        0.0        0.0;
        0.0  1.0        0.0        0.0;
        0.0  0.0  R[1, 1]  R[1, 2];
        0.0  0.0  R[2, 1]  R[2, 2]
    ]
end

function _reference_crx_swapped(theta::Real)
    # CRX with control = qubit 2, target = qubit 1
    S = _reference_swap()
    return S * _reference_crx(theta) * S
end

function _reference_cry(theta::Real)
    R = _reference_ry_2x2(theta)
    return [
        1.0  0.0        0.0        0.0;
        0.0  1.0        0.0        0.0;
        0.0  0.0  R[1, 1]  R[1, 2];
        0.0  0.0  R[2, 1]  R[2, 2]
    ]
end

function _reference_cry_swapped(theta::Real)
    # CRY with control = qubit 2, target = qubit 1
    S = _reference_swap()
    return S * _reference_cry(theta) * S
end

function _reference_crz(theta::Real)
    R = _reference_rz_2x2(theta)
    return [
        1.0  0.0        0.0        0.0;
        0.0  1.0        0.0        0.0;
        0.0  0.0  R[1, 1]  R[1, 2];
        0.0  0.0  R[2, 1]  R[2, 2]
    ]
end

function _reference_crz_swapped(theta::Real)
    # CRZ with control = qubit 2, target = qubit 1
    S = _reference_swap()
    return S * _reference_crz(theta) * S
end

function _reference_cswap()
    return [
        1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
        0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0;
        0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0;
        0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0;
        0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0;
        0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0;
        0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0;
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
    ]
end

function _reference_cswap_control2()
    return [
        1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
        0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0;
        0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0;
        0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0;
        0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0;
        0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0;
        0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0;
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
    ]
end

function _reference_cu1(lambda::Real)
    phase = exp(1.0im * lambda)
    return [
        1.0  0.0  0.0    0.0;
        0.0  1.0  0.0    0.0;
        0.0  0.0  1.0    0.0;
        0.0  0.0  0.0  phase
    ]
end

_reference_cp(lambda::Real) = _reference_cu1(lambda)

function _reference_cu1_swapped(lambda::Real)
    # CU1 with control = qubit 2, target = qubit 1
    S = _reference_swap()
    return S * _reference_cu1(lambda) * S
end

_reference_cp_swapped(lambda::Real) = _reference_cu1_swapped(lambda)

function _reference_cu3_swapped(theta::Real, phi::Real, lambda::Real)
    # CU3 with control = qubit 2, target = qubit 1
    S = _reference_swap()
    return S * _reference_cu3(theta, phi, lambda) * S
end

function _reference_cu3(theta::Real, phi::Real, lambda::Real)
    U = _reference_u3_2x2(theta, phi, lambda)
    return [
        1.0  0.0      0.0       0.0;
        0.0  1.0      0.0       0.0;
        0.0  0.0  U[1, 1]   U[1, 2];
        0.0  0.0  U[2, 1]   U[2, 2]
    ]
end

function _reference_csx()
    SX = _reference_sx_2x2()
    return [
        1.0  0.0        0.0        0.0;
        0.0  1.0        0.0        0.0;
        0.0  0.0  SX[1, 1]  SX[1, 2];
        0.0  0.0  SX[2, 1]  SX[2, 2]
    ]
end

function _reference_csx_swapped()
    # CSX with control = qubit 2, target = qubit 1
    S = _reference_swap()
    return S * _reference_csx() * S
end

function _reference_rxx(theta::Real)
    c = cos(theta / 2)
    s = sin(theta / 2)
    return [
        c           0.0         0.0           -1.0im * s;
        0.0         c   -1.0im * s            0.0;
        0.0  -1.0im * s         c             0.0;
        -1.0im * s   0.0        0.0           c
    ]
end

function _reference_rzz(theta::Real)
    p = exp(-1.0im * theta / 2)
    q = exp(1.0im * theta / 2)
    return [
        p   0.0  0.0  0.0;
        0.0 q    0.0  0.0;
        0.0 0.0  q    0.0;
        0.0 0.0  0.0  p
    ]
end

function _reference_ch()
    invsqrt2 = 1 / sqrt(2)
    return [
        1.0   0.0   0.0       0.0;
        0.0   1.0   0.0       0.0;
        0.0   0.0   invsqrt2  invsqrt2;
        0.0   0.0   invsqrt2 -invsqrt2
    ]
end

function _reference_ch_swapped()
    # CH with control = qubit 2, target = qubit 1
    S = _reference_swap()
    return S * _reference_ch() * S
end

@testset "Composite gate matrix verification" begin
    @testset "X gate" begin
        U_x = _single_gate_unitary(1, "x q[0];")
        @test _approx_equal_up_to_phase(U_x, _reference_x_2x2())
    end

    @testset "Y gate" begin
        U_y = _single_gate_unitary(1, "y q[0];")
        @test _approx_equal_up_to_phase(U_y, _reference_y_2x2())
    end

    @testset "Z gate" begin
        U_z = _single_gate_unitary(1, "z q[0];")
        @test _approx_equal_up_to_phase(U_z, _reference_z_2x2())
    end

    @testset "H gate" begin
        U_h = _single_gate_unitary(1, "h q[0];")
        @test _approx_equal_up_to_phase(U_h, _reference_h_2x2())
    end

    @testset "S gate" begin
        U_s = _single_gate_unitary(1, "s q[0];")
        @test _approx_equal_up_to_phase(U_s, _reference_s_2x2())
    end

    @testset "sdg gate" begin
        U_sdg = _single_gate_unitary(1, "sdg q[0];")
        @test _approx_equal_up_to_phase(U_sdg, _reference_sdg_2x2())
    end

    @testset "SX gate" begin
        U_sx = _single_gate_unitary(1, "sx q[0];")
        @test _approx_equal_up_to_phase(U_sx, _reference_sx_2x2())
    end

    @testset "sxdg gate" begin
        U_sxdg = _single_gate_unitary(1, "sxdg q[0];")
        @test _approx_equal_up_to_phase(U_sxdg, _reference_sxdg_2x2())
    end

    @testset "T gate" begin
        U_t = _single_gate_unitary(1, "t q[0];")
        @test _approx_equal_up_to_phase(U_t, _reference_t_2x2())
    end

    @testset "tdg gate" begin
        U_tdg = _single_gate_unitary(1, "tdg q[0];")
        @test _approx_equal_up_to_phase(U_tdg, _reference_tdg_2x2())
    end

    @testset "rx gate" begin
        theta_rx = pi / 3
        U_rx = _single_gate_unitary(1, "rx($theta_rx) q[0];")
        @test _approx_equal_up_to_phase(U_rx, _reference_rx_2x2(theta_rx))
    end

    @testset "ry gate" begin
        theta_ry = pi / 5
        U_ry = _single_gate_unitary(1, "ry($theta_ry) q[0];")
        @test _approx_equal_up_to_phase(U_ry, _reference_ry_2x2(theta_ry))
    end

    @testset "rz gate" begin
        theta_rz = pi / 7
        U_rz = _single_gate_unitary(1, "rz($theta_rz) q[0];")
        @test _approx_equal_up_to_phase(U_rz, _reference_rz_2x2(theta_rz))
    end

    @testset "P gate" begin
        lambda = pi / 5
        U_p = _single_gate_unitary(1, "p($lambda) q[0];")
        @test _approx_equal_up_to_phase(U_p, _reference_p_2x2(lambda))
    end

    @testset "u1 gate" begin
        lambda = -0.9
        U_u1 = _single_gate_unitary(1, "u1($lambda) q[0];")
        @test _approx_equal_up_to_phase(U_u1, _reference_u1_2x2(lambda))
    end

    @testset "u2 gate" begin
        phi = 0.4
        lambda = -0.6
        U_u2 = _single_gate_unitary(1, "u2($phi, $lambda) q[0];")
        @test _approx_equal_up_to_phase(U_u2, _reference_u2_2x2(phi, lambda))
    end

    @testset "u3 gate" begin
        theta = 0.7
        phi = 0.3
        lambda = -0.2
        U_u3 = _single_gate_unitary(1, "u3($theta, $phi, $lambda) q[0];")
        @test _approx_equal_up_to_phase(U_u3, _reference_u3_2x2(theta, phi, lambda))
    end

    @testset "u gate (alias of u3)" begin
        theta = -0.8
        phi = 0.2
        lambda = 0.9
        U_u = _single_gate_unitary(1, "u($theta, $phi, $lambda) q[0];")
        @test _approx_equal_up_to_phase(U_u, _reference_u3_2x2(theta, phi, lambda))
    end

    @testset "CU gate" begin
        theta = 0.5
        phi = -0.3
        lambda = 0.7
        gamma = 0.0  
        U_cu = _single_gate_unitary(2, "cu($theta, $phi, $lambda, $gamma) q[0], q[1];")
        @test _approx_equal_up_to_phase(U_cu, _reference_cu3(theta, phi, lambda))
    end

    @testset "CX gate" begin
        U_cx = _single_gate_unitary(2, "cx q[0], q[1];")
        @test _approx_equal_up_to_phase(U_cx, _reference_cx())
    end

    @testset "CX gate (swapped qubits)" begin
        U_cx_swapped = _single_gate_unitary(2, "cx q[1], q[0];")
        @test _approx_equal_up_to_phase(U_cx_swapped, _reference_cx_swapped())
    end

    @testset "CZ gate" begin
        U_cz = _single_gate_unitary(2, "cz q[0], q[1];")
        @test _approx_equal_up_to_phase(U_cz, _reference_cz())
    end

    @testset "CZ gate (swapped qubits)" begin
        U_cz_swapped = _single_gate_unitary(2, "cz q[1], q[0];")
        @test _approx_equal_up_to_phase(U_cz_swapped, _reference_cz())
    end

    @testset "CY gate" begin
        U_cy = _single_gate_unitary(2, "cy q[0], q[1];")
        @test _approx_equal_up_to_phase(U_cy, _reference_cy())
    end

    @testset "CY gate (swapped qubits)" begin
        U_cy_swapped = _single_gate_unitary(2, "cy q[1], q[0];")
        @test _approx_equal_up_to_phase(U_cy_swapped, _reference_cy_swapped())
    end

    @testset "CRX gate" begin
        theta_crx = pi / 3
        U_crx = _single_gate_unitary(2, "crx($theta_crx) q[0], q[1];")
        @test _approx_equal_up_to_phase(U_crx, _reference_crx(theta_crx))
    end

    @testset "CRX gate (swapped qubits)" begin
        theta_crx = pi / 3
        U_crx_swapped = _single_gate_unitary(2, "crx($theta_crx) q[1], q[0];")
        @test _approx_equal_up_to_phase(U_crx_swapped, _reference_crx_swapped(theta_crx))
    end

    @testset "CRY gate" begin
        theta_cry = pi / 5
        U_cry = _single_gate_unitary(2, "cry($theta_cry) q[0], q[1];")
        @test _approx_equal_up_to_phase(U_cry, _reference_cry(theta_cry))
    end

    @testset "CRY gate (swapped qubits)" begin
        theta_cry = pi / 5
        U_cry_swapped = _single_gate_unitary(2, "cry($theta_cry) q[1], q[0];")
        @test _approx_equal_up_to_phase(U_cry_swapped, _reference_cry_swapped(theta_cry))
    end

    @testset "CRZ gate" begin
        theta_crz = pi / 7
        U_crz = _single_gate_unitary(2, "crz($theta_crz) q[0], q[1];")
        @test _approx_equal_up_to_phase(U_crz, _reference_crz(theta_crz))
    end

    @testset "CRZ gate (swapped qubits)" begin
        theta_crz = pi / 7
        U_crz_swapped = _single_gate_unitary(2, "crz($theta_crz) q[1], q[0];")
        @test _approx_equal_up_to_phase(U_crz_swapped, _reference_crz_swapped(theta_crz))
    end

    @testset "RXX gate" begin
        theta_rxx = pi / 4
        U_rxx = _single_gate_unitary(2, "rxx($theta_rxx) q[0], q[1];")
        @test _approx_equal_up_to_phase(U_rxx, _reference_rxx(theta_rxx))
    end

    @testset "RXX gate (swapped qubits)" begin
        theta_rxx = pi / 4
        U_rxx_swapped = _single_gate_unitary(2, "rxx($theta_rxx) q[1], q[0];")
        @test _approx_equal_up_to_phase(U_rxx_swapped, _reference_rxx(theta_rxx))
    end

    @testset "RZZ gate" begin
        theta_rzz = pi / 6
        U_rzz = _single_gate_unitary(2, "rzz($theta_rzz) q[0], q[1];")
        @test _approx_equal_up_to_phase(U_rzz, _reference_rzz(theta_rzz))
    end

    @testset "RZZ gate (swapped qubits)" begin
        theta_rzz = pi / 6
        U_rzz_swapped = _single_gate_unitary(2, "rzz($theta_rzz) q[1], q[0];")
        @test _approx_equal_up_to_phase(U_rzz_swapped, _reference_rzz(theta_rzz))
    end

    @testset "CSX gate" begin
        U_csx = _single_gate_unitary(2, "csx q[0], q[1];")
        @test _approx_equal_up_to_phase(U_csx, _reference_csx())
    end

    @testset "CSX gate (swapped qubits)" begin
        U_csx_swapped = _single_gate_unitary(2, "csx q[1], q[0];")
        @test _approx_equal_up_to_phase(U_csx_swapped, _reference_csx_swapped())
    end

    @testset "CU1 gate" begin
        lambda_cu1 = 0.9
        U_cu1 = _single_gate_unitary(2, "cu1($lambda_cu1) q[0], q[1];")
        @test _approx_equal_up_to_phase(U_cu1, _reference_cu1(lambda_cu1))
    end

    @testset "CU1 gate (swapped qubits)" begin
        lambda_cu1 = 0.9
        U_cu1_swapped = _single_gate_unitary(2, "cu1($lambda_cu1) q[1], q[0];")
        @test _approx_equal_up_to_phase(U_cu1_swapped, _reference_cu1_swapped(lambda_cu1))
    end

    @testset "CP gate" begin
        lambda_cp = -0.4
        U_cp = _single_gate_unitary(2, "cp($lambda_cp) q[0], q[1];")
        @test _approx_equal_up_to_phase(U_cp, _reference_cp(lambda_cp))
    end

    @testset "CP gate (swapped qubits)" begin
        lambda_cp = -0.4
        U_cp_swapped = _single_gate_unitary(2, "cp($lambda_cp) q[1], q[0];")
        @test _approx_equal_up_to_phase(U_cp_swapped, _reference_cp_swapped(lambda_cp))
    end

    @testset "CU3 gate (swapped qubits)" begin
        theta_cu3 = 0.5
        phi_cu3 = -0.3
        lambda_cu3 = 0.7
        U_cu3_swapped = _single_gate_unitary(2, "cu3($theta_cu3, $phi_cu3, $lambda_cu3) q[1], q[0];")
        @test _approx_equal_up_to_phase(U_cu3_swapped, _reference_cu3_swapped(theta_cu3, phi_cu3, lambda_cu3))
    end

    @testset "SWAP gate" begin
        U_swap = _single_gate_unitary(2, "swap q[0], q[1];")
        @test _approx_equal_up_to_phase(U_swap, _reference_swap())
    end

    @testset "CCX gate" begin
        U_ccx = _single_gate_unitary(3, "ccx q[0], q[1], q[2];")
        @test _approx_equal_up_to_phase(U_ccx, _reference_ccx())
    end

    @testset "CCX gate (target qubit 2)" begin
        # Controls: q[0], q[2]; target: q[1]
        U_ccx_t2 = _single_gate_unitary(3, "ccx q[0], q[2], q[1];")
        @test _approx_equal_up_to_phase(U_ccx_t2, _reference_ccx_target2())
    end

    @testset "CSWAP gate" begin
        U_cswap = _single_gate_unitary(3, "cswap q[0], q[1], q[2];")
        @test _approx_equal_up_to_phase(U_cswap, _reference_cswap())
    end

    @testset "CSWAP gate (control qubit 2)" begin
        # Control: q[1]; swapped targets: q[0] and q[2]
        U_cswap_c2 = _single_gate_unitary(3, "cswap q[1], q[0], q[2];")
        @test _approx_equal_up_to_phase(U_cswap_c2, _reference_cswap_control2())
    end

    @testset "CH gate" begin
        U_ch = _single_gate_unitary(2, "ch q[0], q[1];")
        @test _approx_equal_up_to_phase(U_ch, _reference_ch())
    end

    @testset "CH gate (swapped qubits)" begin
        U_ch_swapped = _single_gate_unitary(2, "ch q[1], q[0];")
        @test _approx_equal_up_to_phase(U_ch_swapped, _reference_ch_swapped())
    end
end
