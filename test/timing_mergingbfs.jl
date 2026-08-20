using PauliPropagation
using PauliPropagation.Performance
using BenchmarkTools
using Random
using Base.Threads


function timingnumericalPP()
    nq = 8
    nl = 4
    max_weight = Inf
    min_abs_coeff = 0

    pstr = PauliString(nq, :Z, round(Int, nq / 2))
    psum = PauliSum(nq, pstr)
    vpsum = VectorPauliSum(nq, pstr)

    topo = bricklayertopology(nq; periodic=false)
    # topo = get2dtopology(4, 4)
    circ = hardwareefficientcircuit(nq, nl; topology=topo)

    m = length(circ)

    Random.seed!(42)
    thetas = randn(m)

    res1 = propagate(circ, psum, thetas; max_weight, min_abs_coeff)
    res2 = propagate(circ, vpsum, thetas; max_weight, min_abs_coeff)
    res3 = Performance.propagate(circ, vpsum, thetas; max_weight, min_abs_coeff)
    @show overlapwithzero(res1), overlapwithzero(res2), overlapwithzero(res3)
    
    println("\nPauliSum")
    @btime propagate($circ, $psum, $thetas; $max_weight, $min_abs_coeff)
    println("\nVectorPauliSum, nthreads=$(Threads.nthreads())")
    @btime propagate($circ, $vpsum, $thetas; $max_weight, $min_abs_coeff)
    println("\nVectorPauliSum Performance, nthreads=$(Threads.nthreads())")
    @btime Performance.propagate($circ, $vpsum, $thetas; $max_weight, $min_abs_coeff)

    return
end

# 55.345 ms (354 allocations: 3.79 MiB)
timingnumericalPP()