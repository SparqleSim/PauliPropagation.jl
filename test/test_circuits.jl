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
