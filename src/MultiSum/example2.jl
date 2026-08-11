using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using PauliPropagation
using PauliPropagation.Performance
using Base.Threads

include("MultiPSum.jl")
using .MultiPSum

println("Threads: $(nthreads())")

# 6x6 tilted field Ising
nx = 6
ny = 6
nq = nx * ny

dt = 0.05
h = 1.0
J = 0.5

topology = rectangletopology(nx, ny)
layer = PauliRotation[]
rxlayer!(layer, nq)
rzlayer!(layer, nq)
rzzlayer!(layer, topology)

thetas = ones(countparameters(layer)) * dt * 2

x_inds = getparameterindices(layer, PauliRotation, [:X])
z_inds = getparameterindices(layer, PauliRotation, [:Z])
zz_inds = getparameterindices(layer, PauliRotation, [:Z, :Z])
thetas[x_inds] .*= h
thetas[z_inds] .*= h
thetas[zz_inds] .*= J

pstr = PauliString(nq, [:Z, :Z], [21, 22])

min_abs_coeff = 2.0^(-20)

layers = parse(Int, get(ARGS, 1, "18"))
zone_counts = length(ARGS) > 1 ? parse.(Int, split(ARGS[2], ",")) : [2, 4, 8]
repetitions = parse(Int, get(ARGS, 3, "3"))

capacity_hint = 9_000_000

function timeruns(setup, run)
    run(setup())  # warm up

    best = Inf
    local target
    for _ in 1:repetitions
        target = setup()
        elapsed = @elapsed for _ in 1:layers
            run(target)
        end
        best = min(best, elapsed)
    end

    println("Took $(round(best, digits=3)) s (best of $repetitions). " *
            "Overlap: $(overlapwithzero(target)), length: $(length(target))")
    return best, target
end

function vectorcache()
    cache = PropagationCache(VectorPauliSum(pstr))
    resize!(cache, capacity_hint)
    return cache
end

function multipsum(n_zones)
    mpsum = MultiVectorPauliSum(pstr, n_zones)
    resize!(mpsum, capacity_hint)
    return mpsum
end

println("\nVectorPauliSum, fused, thread=false")
serial_time, _ = timeruns(vectorcache,
    cache -> Performance.propagate!(layer, cache, thetas; min_abs_coeff, thread=false))

println("\nVectorPauliSum, fused, thread=true")
threaded_time, _ = timeruns(vectorcache,
    cache -> Performance.propagate!(layer, cache, thetas; min_abs_coeff))

for n_zones in zone_counts
    println("\nMultiVectorPauliSum, $n_zones zones")
    multi_time, mpsum = timeruns(() -> multipsum(n_zones),
        mpsum -> propagate!(layer, mpsum, thetas; min_abs_coeff))

    println("Speedup over fused thread=false: $(round(serial_time / multi_time, digits=2))x, " *
            "over fused thread=true: $(round(threaded_time / multi_time, digits=2))x")
    println("Zone sizes: ", zonesizes(mpsum))
end
