using LinearAlgebra
using PauliPropagation
using Random
using Test

@testset "Unitaries PTM Tests" begin
    """Test the PTM for unitary matrices."""
    tol = 1e-12

    # Test using single-qubit PauliRotation gate
    @testset "PauliRotation Y" begin
        pauligate = PauliRotation(:Y, 1)

        theta = Random.rand()
        U = tomatrix(pauligate, theta)

        ptm = calculateptm(U)
        ptm_schrodinger = calculateptm(U, heisenberg=false)

        expected_ptm = [
            [1 0 0 0];
            [0 cos(theta) 0 -sin(theta)];
            [0 0 1 0];
            [0 sin(theta) 0 cos(theta)]
        ]

        @test LinearAlgebra.norm(ptm - expected_ptm) < tol
        @test LinearAlgebra.norm(ptm_schrodinger - transpose(expected_ptm)) < tol
    end

    # Test using T gate
    @testset "TGate" begin
        tgate = TGate(1)
        matrix = tomatrix(tgate)

        ptmmap = calculateptm(matrix)
        ptm_schrodinger = calculateptm(matrix, heisenberg=false)

        expected_ptm = [
            [1 0 0 0];
            [0 1 / sqrt(2) 1 / sqrt(2) 0];
            [0 -1 / sqrt(2) 1 / sqrt(2) 0];
            [0 0 0 1]
        ]

        @test LinearAlgebra.norm(ptmmap - expected_ptm) < tol
        @test LinearAlgebra.norm(ptm_schrodinger - transpose(expected_ptm)) < tol
    end
end

@testset "Test PTM simulation" begin
    pauli_rotation = PauliRotation(:X, 1)
    θ = π / 4
    U = tomatrix(pauli_rotation, θ)
    ptm = calculateptm(U)
    transfer_map = totransfermap(ptm)
    @test transfer_map isa TransferMap
    @test length(transfer_map) == length(transfer_map.entries)
    @test nqubits(transfer_map) == 1
    @test occursin("4 columns", sprint(show, transfer_map))
    @test occursin("max 2 mapped terms/column", sprint(show, transfer_map))
    @test transfer_map[0][1][1] == 0
    @test_throws BoundsError transfer_map[-1]
    @test_throws BoundsError transfer_map[4]
    @test length(collect(transfer_map)) == 4
    transfer_map_gate = TransferMapGate(transfer_map, 1)
    @test transfer_map_gate.transfer_map isa TransferMap
    @test_throws ArgumentError TransferMap([[(UInt8(0), 1.0)], [(UInt8(1), 1.0)]])
    @test_throws ArgumentError TransferMap(ones(3, 3); ptm=false)
    @test_throws ArgumentError TransferMap(ones(3, 3); ptm=true)
    @test_throws ArgumentError TransferMapGate(transfer_map, [1, 2])

    for symb in [:I, :X, :Y, :Z]
        pstr = PauliString(1, symb, 1)
        psum1 = propagate(pauli_rotation, pstr, θ)
        psum2 = propagate(transfer_map_gate, pstr)

        # it should be zero but we are seeing numerical imprecisions
        @test PauliPropagation.norm(psum1 - psum2) < 1e-15
    end

end


@testset "Test Transfer Maps and TransferMapGates" begin
    nq = 2
    gate = PauliRotation(:X, 2)
    theta = 0.3
    # get the PTmap on qubit 1 but apply on qubit 2
    pauli_rotation = PauliRotation(:X, 1)
    ptmap = totransfermap(1, pauli_rotation, theta)

    g = TransferMapGate(ptmap, 2)
    for symb in [:I, :X, :Y, :Z]
        pstr = PauliString(nq, symb, 2)
        psum1 = propagate(g, pstr)
        psum2 = propagate(gate, pstr, theta)
        @test psum1 == psum2
    end

    # test the matrix constructors
    U = tomatrix(pauli_rotation, theta)
    ptm = calculateptm(U)
    tmap = TransferMap(U; ptm=false)
    gU = TransferMapGate(U, 2)
    gptm = TransferMapGate(ptm, 2)
    gtmap = TransferMapGate(tmap, [2])
    glegacy = TransferMapGate([[item for item in column] for column in g.transfer_map], 2)
    # equality check
    @test tmap isa TransferMap
    @test all((t1[1] == t2[1]) && (t1[2] ≈ t2[2]) for (v1, v2) in zip(g.transfer_map, gU.transfer_map) for (t1, t2) in zip(v1, v2))
    @test all((t1[1] == t2[1]) && (t1[2] ≈ t2[2]) for (v1, v2) in zip(g.transfer_map, gptm.transfer_map) for (t1, t2) in zip(v1, v2))
    @test all((t1[1] == t2[1]) && (t1[2] ≈ t2[2]) for (v1, v2) in zip(g.transfer_map, gtmap.transfer_map) for (t1, t2) in zip(v1, v2))
    @test length(glegacy.transfer_map) == length(g.transfer_map)
    @test map(collect, glegacy.transfer_map) == map(collect, g.transfer_map)
    @test gU.qinds == g.qinds
    @test gptm.qinds == g.qinds
    @test gtmap.qinds == g.qinds

    # Issue #144 API example: construct from a computational-basis unitary,
    # use non-contiguous qubits, and propagate through TransferMapGate.
    issue_tmap = TransferMap(Matrix{ComplexF64}(I, 4, 4); ptm=false)
    issue_gate = TransferMapGate(issue_tmap, [1, 3])
    issue_pstr = PauliString(4, [:Z, :Y], [1, 3])
    @test propagate(issue_gate, issue_pstr) == PauliSum(issue_pstr)

    noncontiguous_gate = CliffordGate(:CNOT, [1, 3])
    noncontiguous_tmap = totransfermap(2, [CliffordGate(:CNOT, [1, 2])])
    noncontiguous_tmap_gate = TransferMapGate(noncontiguous_tmap, [1, 3])
    noncontiguous_pstr = PauliString(4, [:X, :Y, :Z], [1, 3, 4])
    noncontiguous_psum = propagate(noncontiguous_tmap_gate, noncontiguous_pstr)
    expected_noncontiguous_psum = propagate(noncontiguous_gate, noncontiguous_pstr)
    @test noncontiguous_psum == expected_noncontiguous_psum



    nq = 2

    circuit = [CliffordGate(:CNOT, [1, 2]), CliffordGate(:X, 1), CliffordGate(:H, 2), TGate(1), TGate(2)]
    ptmap = totransfermap(nq, circuit)
    g = TransferMapGate(ptmap, (1, 2))

    pstr = PauliString(nq, [:Y, :X], [1, 2])
    psum1 = propagate(g, pstr)
    psum2 = propagate(circuit, pstr)
    @test psum1 == psum2


    nq = 5

    circuit = Gate[]
    rxlayer!(circuit, nq)
    rzlayer!(circuit, nq)
    ryylayer!(circuit, bricklayertopology(nq))

    thetas = randn(countparameters(circuit))
    static_circuit = freeze(circuit, thetas)

    ptmap = totransfermap(nq, circuit, thetas)
    static_ptmap = totransfermap(nq, static_circuit)
    @test ptmap == static_ptmap

    g = TransferMapGate(ptmap, 1:nq)

    pstr = PauliString(nq, [rand((:X, :Y, :Z)) for _ in 1:nq], 1:nq)
    psum1 = propagate(g, pstr)
    psum2 = propagate(circuit, pstr, thetas)
    @test psum1 == psum2

end
