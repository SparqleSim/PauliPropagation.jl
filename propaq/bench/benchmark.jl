###
##
# Runs the propagation paths of the library and this prototype over the same problem and prints a
# table. The problems are the ones Propaq benchmarks in its paper (arXiv:2609.07730), plus the 1D
# chain that `propaq-comparison.md` measured.
#
#   julia --project=. -t1 propaq/bench/benchmark.jl tfim2d 10 12 14
#   julia --project=. -t8 propaq/bench/benchmark.jl hubbard zones=32 4 6 8
#
# Thread counts come from `-t`. The zone counts of the split paths default to `defaultnzones()`.
##
###

using PauliPropagation
using PauliPropagation.PropagationBase: defaultnzones
using Printf

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "..", "src", "IndexedPropagation.jl"))
using .IndexedPropagation

const PERF = PauliPropagation.Performance

"""
    problem(name, nsteps)

The circuit, parameters, observable and coefficient cutoff of one benchmark problem.
"""
function problem(name::String, nsteps::Int)
    if name == "tfim2d"        # Propaq Fig. 1(a): 6 x 6 transverse-field Ising
        circuit, thetas = tfim2d(6, 6, nsteps)
        return circuit, thetas, tfim2dobservable(6, 6), 1e-6
    elseif name == "tfim2d100" # the same, on a 10 x 10 grid
        circuit, thetas = tfim2d(10, 10, nsteps)
        return circuit, thetas, tfim2dobservable(10, 10), 1e-6
    elseif name == "hubbard"   # Propaq Fig. 1(b): 3 x 3 Fermi-Hubbard. The paper's 1e-6 cutoff runs
                               # out of memory here within a Trotter step, so this uses 1e-5
        circuit, thetas = hubbard2d(3, 3, nsteps)
        return circuit, thetas, PauliString(18, :Z, 9), 1e-5
    elseif name == "tilted"    # the 6 x 6 tilted-field Ising circuit of examples/advanced_performance.ipynb
        circuit, thetas = tiltedising(6, 6, nsteps)
        return circuit, thetas, tiltedisingobservable(6, 6), 2.0^-20
    elseif name == "tfi1d"     # the 64-qubit chain of propaq-comparison.md
        circuit, thetas = tfi1d(64, nsteps)
        return circuit, thetas, PauliString(64, :Z, 32), 1e-8
    end
    throw(ArgumentError("unknown problem $name"))
end

const PATHS = (
    "PauliSum" => (c, o, t, cut, nz) -> propagate(c, PauliSum(o), t; min_abs_coeff=cut),
    "VectorPauliSum" => (c, o, t, cut, nz) -> propagate(c, VectorPauliSum(o), t; min_abs_coeff=cut),
    "fused" => (c, o, t, cut, nz) -> PERF.propagate(c, VectorPauliSum(o), t; min_abs_coeff=cut),
    "fused multi" => (c, o, t, cut, nz) -> PERF.propagate(c, MultiPauliSum(VectorPauliSum(o), nz), t; min_abs_coeff=cut),
    "indexed" => (c, o, t, cut, nz) -> IndexedPropagation.propagate(c, VectorPauliSum(o), t; min_abs_coeff=cut),
    "indexed multi" => (c, o, t, cut, nz) -> IndexedPropagation.propagate(c, VectorPauliSum(o), t; n_zones=nz, min_abs_coeff=cut),
)

# Best of `repeats`, plus whatever the last of them returned, so nothing is propagated twice.
function best(run, args...; repeats::Int=3)
    elapsed = Inf
    result = nothing
    for _ in 1:repeats
        GC.gc()
        t = @elapsed result = run(args...)
        elapsed = min(elapsed, t)
    end
    return elapsed, result
end

function main(name::String, steps; n_zones::Int=defaultnzones(), skip=("PauliSum",))
    @printf("%s, %d thread(s), %d zones\n\n", name, Threads.nthreads(), n_zones)
    @printf("%-7s %7s %10s", "steps", "gates", "terms")
    for (path, _) in PATHS
        path in skip || @printf(" %14s", path)
    end
    println()

    for nsteps in steps
        circuit, thetas, obs, cutoff = problem(name, nsteps)
        @printf("%-7d %7d", nsteps, length(circuit))

        nterms = 0
        for (path, run) in PATHS
            path in skip && continue
            elapsed, psum = best(run, circuit, obs, thetas, cutoff, n_zones)
            nterms == 0 && (nterms = length(psum); @printf(" %10d", nterms))
            # paths that sum a term's contributions in a different order can disagree on the very
            # last terms above the cutoff, so this only catches a path that truncates differently
            abs(length(psum) - nterms) <= 1e-3 * nterms ||
                error("$path returned $(length(psum)) terms, expected $nterms")
            @printf(" %12.3f s", elapsed)
            flush(stdout)
        end
        println()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = ARGS[2:end]
    n_zones = defaultnzones()
    if !isempty(args) && startswith(args[1], "zones=")
        n_zones = parse(Int, args[1][7:end])
        args = args[2:end]
    end
    main(ARGS[1], parse.(Int, args); n_zones)
end
