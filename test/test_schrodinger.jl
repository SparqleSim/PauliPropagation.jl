## Test heisenberg=true and heisenberg=false equivalence for implemented gates
using Random

@testset "Test propagate heisenberg vs schrodinger equivalence" begin
    nq = 5
    nl1 = 2
    nl2 = 3
    nl_half = max(nl1, nl2)

    min_abs_coeff = 0

    pstr_beginning = PauliString(nq, :Z, 2)
    pstr_end = PauliString(nq, [:X, :Y], [1, 3])

    topology = staircasetopology(nq; periodic=true)

    # contains Pauli rotations and CNOTs
    circ1 = hardwareefficientcircuit(nq, nl1; topology=topology)
    circ2 = tiltedtfitrottercircuit(nq, nl2; topology=topology)

    # tests frozen gates and noise
    noise_layer = Gate[DepolarizingNoise(i, 0.1) for i in 1:nq]

    circ = vcat(circ1, noise_layer, circ2)

    thetas1 = randn(countparameters(circ1))
    thetas2 = randn(countparameters(circ2))
    thetas = vcat(thetas1, thetas2)

    psum_beginning = propagate(circ, pstr_beginning, thetas; heisenberg=false, min_abs_coeff)
    val1 = scalarproduct(psum_beginning, pstr_end)

    psum_end = propagate!(circ, pstr_end, thetas; heisenberg=true, min_abs_coeff)
    val2 = scalarproduct(pstr_beginning, psum_end)

    psum_beginning = propagate(circ1, pstr_beginning, thetas1; heisenberg=false, min_abs_coeff)
    psum_beginning = propagate!(noise_layer, psum_beginning; heisenberg=false, min_abs_coeff)
    psum_end = propagate(circ2, pstr_end, thetas2; heisenberg=true, min_abs_coeff)
    val3 = scalarproduct(psum_beginning, psum_end)

    @test val1 ≈ val2 ≈ val3

end

@testset "Test inversion" begin
    nx = 3
    ny = 2
    nq = nx * ny
    nl = 3
    topology = rectangletopology(nx, ny; periodic=false)

    min_abs_coeff = 1e-12

    circ = hardwareefficientcircuit(nq, nl; topology=topology)
    thetas = randn(countparameters(circ))

    fcirc = freeze(circ, thetas)

    pstr = PauliString(nq, rand([:X, :Y, :Z], nq), 1:nq)
    psum_forward = propagate(fcirc, pstr; heisenberg=false, min_abs_coeff)
    psum_backward = propagate(fcirc, psum_forward; heisenberg=true, min_abs_coeff)
    @test PauliSum(pstr) ≈ psum_backward

    # flip direction
    pstr = PauliString(nq, rand([:X, :Y, :Z], nq), 1:nq)
    psum_forward = propagate(fcirc, pstr; heisenberg=true, min_abs_coeff)
    psum_backward = propagate(fcirc, psum_forward; heisenberg=false, min_abs_coeff)
    @test PauliSum(pstr) ≈ psum_backward

end

@testset "Test conversions" begin
    # The Heisenberg conversions are implicitly already performed above and in other tests.

    nq = 4
    # Pauli Rotation 
    gate = PauliRotation(rand([:I, :X, :Y, :Z], nq), 1:nq)
    θ = randn()
    gate_schrodinger, param_schrodinger = toschrodinger(gate, θ)
    @test gate_schrodinger == gate
    @test param_schrodinger == -θ

    # FrozenGate if PauliRotation
    gate = freeze(gate, θ)
    gate_schrodinger = toschrodinger(gate)
    @test gate_schrodinger.gate == gate.gate
    @test gate_schrodinger.parameter == -θ

    # CliffordGate
    cliff_symb = rand(keys(clifford_map))
    transfer_map = clifford_map[cliff_symb]
    active_qubits = Int(log(4, length(transfer_map)))
    gate = CliffordGate(cliff_symb, 1:active_qubits)
    gate_schrodinger = toschrodinger(gate)
    @test gate_schrodinger.symbol == Symbol(cliff_symb, :_transpose)
    @test gate_schrodinger.qinds == gate.qinds


    # PauliNoise
    gate = rand([DepolarizingNoise, DephasingNoise, PauliXNoise, PauliYNoise])(3)
    p = 0.1
    gate_schrodinger, param_schrodinger = toschrodinger(gate, p)
    @test gate_schrodinger == gate
    @test param_schrodinger == p

    # AmplitudeDampingNoise should throw in Schrödinger
    gate = AmplitudeDampingNoise(2)
    p = 0.1
    @test_throws ErrorException toschrodinger(gate, p)

    # TransferMapGate should throw in Schrodinger
    ptm = randn(4, 4)
    gate = TransferMapGate(ptm, 1)
    @test_throws ErrorException toschrodinger(gate)

    # ImaginaryPauliRotation should throw in Heisenberg
    gate = ImaginaryPauliRotation(rand([:I, :X, :Y, :Z], nq), 1:nq)
    τ = randn()
    @test_throws ErrorException toheisenberg(gate, τ)

end

