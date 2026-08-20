using Test

@testset "Test Circuit Utils" begin
    nq = 3
    nl = 2

    circuit = tfitrottercircuit(nq, nl)

    nparams = countparameters(circuit)
    @test nparams == length(circuit)
    @test_throws MethodError getparameterindices(circuit, StaticGate)
    @test getparameterindices(circuit, ParametrizedGate) == 1:length(circuit)
    @test getparameterindices(circuit, PauliRotation) == 1:length(circuit)
    @test_throws MethodError getparameterindices(circuit, PauliRotation, :X)
    @test getparameterindices(circuit, PauliRotation, [:X]) == [3, 4, 5, 8, 9, 10]
    @test getparameterindices(circuit, PauliRotation, [:X], [1]) == [3, 8]

end


@testset "Test Topologies" begin
    nq = rand(1:100)

    open_brick_topo = bricklayertopology(nq)
    periodic_brick_topo = bricklayertopology(nq; periodic=true)
    @test length(open_brick_topo) == (nq - 1)
    @test nq <= 2 ? length(periodic_brick_topo) == (nq - 1) : length(periodic_brick_topo) == nq
    @test length(staircasetopology(nq)) == length(open_brick_topo)

    topo = staircasetopology2d(rand(1:10), rand(1:10))
    @test topo == unique(topo)
    topo = rectangletopology(rand(1:10), rand(1:10))
    @test topo == unique(topo)

    # the periodic wraparound must not self-loop or duplicate a pair at small nqubits
    @test staircasetopology(1; periodic=true) == Tuple{Int,Int}[]
    @test staircasetopology(2; periodic=true) == [(1, 2)]

end


@testset "Test rectanglebricktopology" begin
    # small grid dimensions must not produce self-loops or duplicate (possibly reversed) pairs
    for nx in 1:6, ny in 1:6, periodic_x in (false, true), periodic_y in (false, true)
        topo = rectanglebricktopology(nx, ny; periodic_x=periodic_x, periodic_y=periodic_y)
        @test all(pair -> pair[1] != pair[2], topo)

        normalized = [pair[1] < pair[2] ? pair : (pair[2], pair[1]) for pair in topo]
        @test normalized == unique(normalized)
    end

    # with full periodicity, it connects the same pairs as rectangletopology
    for nx in 1:6, ny in 1:6
        brick_topo = Set(pair[1] < pair[2] ? pair : (pair[2], pair[1]) for pair in rectanglebricktopology(nx, ny; periodic_x=true, periodic_y=true))
        grid_topo = Set(rectangletopology(nx, ny; periodic=true))
        @test brick_topo == grid_topo
    end

    # without periodicity, it connects the same pairs as the full open grid
    nx, ny = rand(2:8), rand(2:8)
    brick_topo = Set(pair[1] < pair[2] ? pair : (pair[2], pair[1]) for pair in rectanglebricktopology(nx, ny))
    grid_topo = Set(rectangletopology(nx, ny))
    @test brick_topo == grid_topo
end


@testset "Test Circuit Builders" begin
    nq = rand(1:100)
    nl = rand(1:100)

    @test length(hardwareefficientcircuit(nq, nl)) == length(hardwareefficientcircuit(nq, nl; topology=bricklayertopology(nq)))
    @test length(efficientsu2circuit(nq, nl)) == length(efficientsu2circuit(nq, nl; topology=bricklayertopology(nq)))
    @test length(tfitrottercircuit(nq, nl)) == length(tfitrottercircuit(nq, nl; topology=bricklayertopology(nq)))
    @test length(tiltedtfitrottercircuit(nq, nl)) == length(tiltedtfitrottercircuit(nq, nl; topology=bricklayertopology(nq)))
    @test length(heisenbergtrottercircuit(nq, nl)) == length(heisenbergtrottercircuit(nq, nl; topology=bricklayertopology(nq)))
    @test length(su4circuit(nq, nl)) == length(su4circuit(nq, nl; topology=bricklayertopology(nq)))
    @test length(qcnncircuit(nq)) == length(qcnncircuit(nq; periodic=false))

end
