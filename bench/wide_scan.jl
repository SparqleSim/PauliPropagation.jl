# Propagation cost against qubit count. The light cone keeps the sum the same size at every qubit
# count, so the timings only reflect the width of the Pauli string.
#
# `stride` is the distance between the two qubits of a ZZ rotation. The connectivity is a chain either
# way, so the workload is identical; only the mask bits move. Stride 1 gives one contiguous run of
# mask bits, stride 2 gives two, which is the multi-digit case for the radix tail sort.
#
# usage: julia -tN bench/wide_scan.jl <project_path> <out.csv> <label> [nlayers] [qubits] [strides]
# Run through bench/capped.sh.
using Pkg
Pkg.activate(ARGS[1]; io=devnull)
using PauliPropagation
using PauliPropagation.Performance
using Base.Threads

const THETA = 0.1
const MIN_ABS_COEFF = 0.0

# each new width recompiles the pipeline, and that compile costs superlinear memory, not the sum
const MAX_BITS = 8192          # 4096 qubits

# checked every layer, not at the end: an untruncated sum can multiply by a thousand in one layer
const MAX_TERMS = 200_000

# set PP_LOCAL / PP_RADIX to 0 to time without one of the two optimizations
Performance.USE_LOCAL_GATES[] = get(ENV, "PP_LOCAL", "1") == "1"
Performance.USE_RADIX_TAILSORT[] = get(ENV, "PP_RADIX", "1") == "1"

label = ARGS[3]
nlayers = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 4
qubits = length(ARGS) >= 5 ? parse.(Int, split(ARGS[5], ",")) :
         [128, 256, 512, 768, 896, 1024, 1056, 1088, 1280, 1536, 2048]
strides = length(ARGS) >= 6 ? parse.(Int, split(ARGS[6], ",")) : [1, 2]

# a chain whose bonds join qubits `k` apart; relabelling a chain leaves the sum unchanged
stridedtopology(nq, k) = [(i, i + k) for i in 1:(nq-k)]

function runpoint(nq, k, reps)
    TT = getinttype(nq)
    layer = tfitrottercircuit(nq, 1; topology=stridedtopology(nq, k))
    thetas = ones(countparameters(layer)) * THETA
    obs = symboltoint(TT, [:Z], [nq ÷ 2])
    best = Inf
    local pc
    for rep in 1:(reps+1)   # the first pass warms up compilation and is not scored
        pc = PropagationCache(VectorPauliSum(nq, TT[obs], [1.0]))
        t = @timed for _ in 1:nlayers
            Performance.propagate!(layer, pc, thetas; min_abs_coeff=MIN_ABS_COEFF)
            length(pc) > MAX_TERMS && error("term cap exceeded at nq=$nq stride=$k: $(length(pc))")
        end
        rep > 1 && (best = min(best, t.time))
    end
    return (time=best, len=length(pc), overlap=overlapwithzero(pc))
end

rssgb() = Sys.maxrss() / 2^30

out = open(ARGS[2], "w")
println(out, "label,stride,nq,nlayers,threads,inttype,sizeof,alignedsizeof,time,len,overlap")

for k in strides, nq in qubits
    if 2 * nq > MAX_BITS
        println("stopping at nq=$nq: $(2 * nq) bits is past the $(MAX_BITS)-bit compile cap")
        break
    end
    r = runpoint(nq, k, 2)
    TT = getinttype(nq)
    println(out, "$label,$k,$nq,$nlayers,$(nthreads()),$TT,$(sizeof(TT)),$(Base.aligned_sizeof(TT)),$(r.time),$(r.len),$(r.overlap)")
    flush(out)
    println("stride=$k nq=$nq [$label t$(nthreads())] $(round(r.time, digits=4))s len=$(r.len) " *
            "$(sizeof(TT))B/$(Base.aligned_sizeof(TT))B rss=$(round(rssgb(), digits=2))GB")
    flush(stdout)
end
close(out)
