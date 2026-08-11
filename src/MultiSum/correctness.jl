using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using PauliPropagation
using Random
using Base.Threads

include("MultiPSum.jl")
using .MultiPSum

function referencesum(circuit, pstr, thetas; kwargs...)
    psum = PauliSum(pstr)
    propagate!(circuit, psum, thetas; kwargs...)
    return psum
end

function multisum(circuit, pstr, thetas, n_zones; kwargs...)
    mpsum = MultiVectorPauliSum(pstr, n_zones)
    propagate!(circuit, mpsum, thetas; kwargs...)
    return PauliSum(mpsum)
end

function maxdeviation(psum1, psum2)
    deviation = 0.0
    for pstr in union(keys(psum1.terms), keys(psum2.terms))
        deviation = max(deviation, abs(getcoeff(psum1, pstr) - getcoeff(psum2, pstr)))
    end
    return deviation
end

function report(name, circuit, pstr, thetas, n_zones; kwargs...)
    expected = referencesum(circuit, pstr, thetas; kwargs...)
    got = multisum(circuit, pstr, thetas, n_zones; kwargs...)
    deviation = maxdeviation(expected, got)
    println(rpad(name, 34), " zones=", n_zones,
        " terms=", lpad(length(expected), 8), "/", lpad(length(got), 8),
        " maxdev=", deviation)
    return deviation < 1e-12 && length(expected) == length(got)
end

Random.seed!(42)

nq = 10
topology = bricklayertopology(nq)

rotations = PauliRotation[]
rxlayer!(rotations, nq)
rzlayer!(rotations, nq)
rzzlayer!(rotations, topology)
append!(rotations, [PauliRotation([:X, :Y, :Z], [1, 4, 7]), PauliRotation([:Y, :Y], [2, 9])])

cliffords = [CliffordGate(:CNOT, pair) for pair in topology]
append!(cliffords, [CliffordGate(:H, [q]) for q in 1:nq])

noises = vcat([DepolarizingNoise(q) for q in 1:nq], [DephasingNoise(q) for q in 1:nq])

mixed = Gate[]
for _ in 1:3
    append!(mixed, rotations)
    append!(mixed, cliffords)
    append!(mixed, noises)
end

pstr = PauliString(nq, [:Z, :Z], [3, 4])

allpassed = true
for n_zones in (1, 2, 4, 8)
    thetas = randn(countparameters(rotations))
    global allpassed &= report("rotations", rotations, pstr, thetas, n_zones; min_abs_coeff=0.0)
    global allpassed &= report("rotations, truncated", rotations, pstr, thetas, n_zones; min_abs_coeff=1e-3, max_weight=5)
    global allpassed &= report("rotations, then cliffords", vcat(Gate[], rotations, cliffords), pstr, thetas, n_zones; min_abs_coeff=0.0)

    params = [gate isa PauliNoise ? 0.05 : randn() for gate in mixed if gate isa ParametrizedGate]
    global allpassed &= report("rotations+cliffords+noise", mixed, pstr, params, n_zones; min_abs_coeff=1e-8)
end

println(allpassed ? "\nAll checks passed." : "\nFAILURES above.")
