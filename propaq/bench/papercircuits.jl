###
##
# Runs the propagation paths over the circuits that `propaq-benchmarks` saves as JSON, so that the
# library, this prototype and Propaq itself are measured on one and the same gate list.
#
#   cd propaq-benchmarks/experiments/ising_trotter && python3 generate_circuits.py
#   julia --project=. -t8 propaq/bench/papercircuits.jl fused indexedmulti -- circuits/6x6_steps14.json
#
# Under the benchmark repo's own environment, which pins PauliPropagation v0.7.3, only the default
# path exists and the script offers only that:
#
#   julia --project=propaq-benchmarks/julia_env -t8 propaq/bench/papercircuits.jl default -- circuits/6x6_steps14.json
#
# Every line of output is one JSON record, with the process's peak RSS so far, so a run per process
# measures one path's footprint. `CUTOFF`, `ZONES` and `REPEATS` come from the environment.
##
###

using PauliPropagation
using JSON
using Printf

const HAS_PERFORMANCE = isdefined(PauliPropagation, :Performance)

if HAS_PERFORMANCE
    include(joinpath(@__DIR__, "..", "src", "IndexedPropagation.jl"))
    using .IndexedPropagation
    using PauliPropagation.PropagationBase: defaultnzones
end

"""
    loadproblem(path)

The circuit, parameters, observable and Trotter step count of one saved `propaq-benchmarks` problem.
The gate names are the ones the benchmark repo's circuits use, including the Clifford+T set of
its Clifford deferral experiment.
"""
function loadproblem(path::AbstractString)
    d = JSON.parsefile(path)
    nq = d["n_qubits"]

    circuit = Gate[]
    thetas = Float64[]
    for g in d["gates"]
        name = g["name"]
        qinds = [q + 1 for q in g["qubits"]]
        if name == "rzz"
            push!(circuit, PauliRotation([:Z, :Z], qinds)); push!(thetas, g["angle"])
        elseif name in ("rx", "ry", "rz")
            push!(circuit, PauliRotation(Symbol(uppercase(name[2:2])), qinds[1])); push!(thetas, g["angle"])
        elseif name == "t"
            push!(circuit, PauliRotation(:Z, qinds[1])); push!(thetas, pi / 4)
        elseif name == "cx"
            push!(circuit, CliffordGate(:CNOT, qinds))
        elseif name in ("h", "x", "y", "z", "s")
            push!(circuit, CliffordGate(Symbol(uppercase(name)), qinds[1]))
        elseif name == "sdg"
            push!(circuit, CliffordGate(:S, qinds[1])); push!(circuit, CliffordGate(:Z, qinds[1]))
        else
            throw(ArgumentError("gate '$name' is outside the benchmark basis"))
        end
    end

    # Qiskit labels read right to left
    obs = PauliSum(nq)
    for (label, coeff) in zip(d["observable"]["paulis"], d["observable"]["coeffs"])
        symbols = Symbol[]
        qinds = Int[]
        for (k, ch) in enumerate(label)
            ch == 'I' && continue
            push!(symbols, Symbol(ch))
            push!(qinds, nq - k + 1)
        end
        add!(obs, PauliString(nq, symbols, qinds, Float64(coeff)))
    end

    return circuit, thetas, obs, d["params"]["steps"]
end

const PATHS = Dict{String,Function}(
    "default" => (c, o, t, cut, nz) -> propagate(c, VectorPauliSum(o), t; min_abs_coeff=cut),
)

if HAS_PERFORMANCE
    const PERF = PauliPropagation.Performance
    PATHS["fused"] = (c, o, t, cut, nz) -> PERF.propagate(c, VectorPauliSum(o), t; min_abs_coeff=cut)
    PATHS["fusedmulti"] = (c, o, t, cut, nz) -> PERF.propagate(c, MultiPauliSum(VectorPauliSum(o), nz), t; min_abs_coeff=cut)
    PATHS["indexed"] = (c, o, t, cut, nz) -> IndexedPropagation.propagate(c, VectorPauliSum(o), t; min_abs_coeff=cut)
    PATHS["indexedmulti"] = (c, o, t, cut, nz) -> IndexedPropagation.propagate(c, VectorPauliSum(o), t; n_zones=nz, min_abs_coeff=cut)
end

# Best of `repeats`, plus whatever the last of them returned.
function best(run, args...; repeats::Int=parse(Int, get(ENV, "REPEATS", "3")))
    elapsed = Inf
    result = nothing
    for _ in 1:repeats
        GC.gc()
        t = @elapsed result = run(args...)
        elapsed = min(elapsed, t)
    end
    return elapsed, result
end

function main(names, files; cutoff::Float64, n_zones::Int)
    # compile every path on the smallest circuit before timing anything
    let (circuit, thetas, obs, _) = loadproblem(first(files))
        for name in names
            PATHS[name](circuit, obs, thetas, cutoff, n_zones)
        end
    end

    for file in files
        circuit, thetas, obs, steps = loadproblem(file)
        for name in names
            elapsed, psum = best(PATHS[name], circuit, obs, thetas, cutoff, n_zones)
            @printf("{\"path\":\"%s\",\"file\":\"%s\",\"steps\":%d,\"threads\":%d,\"zones\":%d,\"wall_s\":%.4f,\"n_terms\":%d,\"expectation_value\":%.12g,\"maxrss_gb\":%.3f}\n",
                name, basename(file), steps, Threads.nthreads(), n_zones, elapsed, length(psum), overlapwithzero(psum), Sys.maxrss() / 2^30)
            flush(stdout)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    sep = findfirst(==("--"), ARGS)
    names = ARGS[1:sep-1]
    files = ARGS[sep+1:end]
    cutoff = parse(Float64, get(ENV, "CUTOFF", "1e-6"))
    n_zones = HAS_PERFORMANCE ? parse(Int, get(ENV, "ZONES", string(defaultnzones()))) : 1
    main(names, files; cutoff, n_zones)
end
