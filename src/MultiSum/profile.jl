using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using PauliPropagation
using PauliPropagation.Performance
using Base.Threads

include("MultiPSum.jl")
using .MultiPSum

const PF = PauliPropagation.Performance
const PB = PauliPropagation.PropagationBase

println("Threads: $(nthreads())")

nx, ny = 6, 6
nq = nx * ny
dt, h, J = 0.05, 1.0, 0.5

topology = rectangletopology(nx, ny)
layer = PauliRotation[]
rxlayer!(layer, nq)
rzlayer!(layer, nq)
rzzlayer!(layer, topology)

thetas = ones(countparameters(layer)) * dt * 2
thetas[getparameterindices(layer, PauliRotation, [:X])] .*= h
thetas[getparameterindices(layer, PauliRotation, [:Z])] .*= h
thetas[getparameterindices(layer, PauliRotation, [:Z, :Z])] .*= J

const pstr = PauliString(nq, [:Z, :Z], [21, 22])
const min_abs_coeff = 2.0^(-20)

layers = parse(Int, get(ARGS, 1, "18"))
zone_counts = length(ARGS) > 1 ? parse.(Int, split(ARGS[2], ",")) : [1, 2, 4, 8]

# per (phase, zone) busy time of the last gate, accumulated across gates
mutable struct PhaseLog
    branch_busy::Matrix{Float64}
    merge_busy::Matrix{Float64}
    branch_wall::Vector{Float64}
    merge_wall::Vector{Float64}
    gate::Int
end

PhaseLog(n_zones, n_gates) = PhaseLog(zeros(n_gates, n_zones), zeros(n_gates, n_zones),
    zeros(n_gates), zeros(n_gates), 0)

# mirrors MultiPSum.applygate! for PauliRotation, timing each zone in each of the two passes
function timedapplygate!(gate::PauliRotation, mpsum, theta, log::PhaseLog; thread::Bool=true)
    log.gate += 1
    gg = log.gate

    xor_mask = symboltoint(paulitype(mpsum), gate.symbols, gate.qinds)
    gate_mask = PF._gatemask(xor_mask, paulis(mainsum(mpsum.zones[1])))
    zone_shift = MultiPSum._zonebits(xor_mask, mpsum.zonemasks)
    kept_val, new_val = cos(theta), sin(theta)
    truncfunc(pstr, coeff) = PF._coefftruncfunc(pstr, coeff; min_abs_coeff, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing)

    t0 = time_ns()
    MultiPSum._eachzone(nzones(mpsum), thread) do source
        t = time_ns()
        MultiPSum._branchintooutbox!(mpsum, source, MultiPSum._partner(source, zone_shift),
            gate_mask, kept_val, new_val, Inf)
        log.branch_busy[gg, source] += (time_ns() - t) * 1e-9
    end
    log.branch_wall[gg] += (time_ns() - t0) * 1e-9

    t0 = time_ns()
    MultiPSum._eachzone(nzones(mpsum), thread) do owner
        t = time_ns()
        MultiPSum._mergefromoutbox!(mpsum, owner, MultiPSum._partner(owner, zone_shift), xor_mask, truncfunc)
        log.merge_busy[gg, owner] += (time_ns() - t) * 1e-9
    end
    log.merge_wall[gg] += (time_ns() - t0) * 1e-9

    return mpsum
end

function runlayers(n_zones, thread)
    mpsum = MultiVectorPauliSum(pstr, n_zones)
    foreach(zone -> resize!(zone, cld(9_000_000, n_zones)), mpsum.zones)

    log = PhaseLog(n_zones, length(layer))
    heisenberg_layer, heisenberg_thetas = PauliPropagation._preparecircuit(layer, thetas, true)

    total = @elapsed for _ in 1:layers
        log.gate = 0
        for (gate, theta) in zip(heisenberg_layer, heisenberg_thetas)
            timedapplygate!(gate, mpsum, theta, log; thread)
        end
    end

    return total, log, mpsum
end

function report(n_zones, thread, total, log)
    branch_wall = sum(log.branch_wall)
    merge_wall = sum(log.merge_wall)

    # per gate, the zone that finished last sets the wall time; the rest wait at the barrier
    branch_busy = sum(maximum(log.branch_busy, dims=2))
    merge_busy = sum(maximum(log.merge_busy, dims=2))
    branch_mean = sum(sum(log.branch_busy, dims=2)) / n_zones
    merge_mean = sum(sum(log.merge_busy, dims=2)) / n_zones

    label = thread ? "$n_zones zones, threaded" : "$n_zones zones, serial  "
    println(rpad(label, 26),
        " total=", lpad(round(total, digits=3), 7),
        "  branch=", lpad(round(branch_wall, digits=3), 6),
        "  merge=", lpad(round(merge_wall, digits=3), 6),
        "  outside=", lpad(round(total - branch_wall - merge_wall, digits=3), 6))
    if thread
        println(rpad("", 26),
            " straggler loss: branch ", round(100 * (1 - branch_mean / branch_busy), digits=1), "%",
            ", merge ", round(100 * (1 - merge_mean / merge_busy), digits=1), "%",
            "; barrier+spawn ", round(100 * (1 - (branch_busy + merge_busy) / (branch_wall + merge_wall)), digits=1), "%")
    end
    return (branch=branch_wall, merge=merge_wall, total=total)
end

runlayers(2, true)  # warm up

println("\n$layers layers")
serial = Dict{Int,Any}()
for n_zones in zone_counts
    total, log, mpsum = runlayers(n_zones, false)
    serial[n_zones] = report(n_zones, false, total, log)
    println(rpad("", 26), " terms=", length(mpsum))
end

println()
for n_zones in zone_counts
    total, log, _ = runlayers(n_zones, true)
    threaded = report(n_zones, true, total, log)
    ref = serial[n_zones]
    println(rpad("", 26),
        " parallel efficiency: branch ", round(100 * ref.branch / (n_zones * threaded.branch), digits=1), "%",
        ", merge ", round(100 * ref.merge / (n_zones * threaded.merge), digits=1), "%",
        ", overall ", round(100 * ref.total / (n_zones * threaded.total), digits=1), "%")
end
