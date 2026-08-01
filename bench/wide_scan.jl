# Propagation cost against qubit count for wide Pauli strings, at a workload whose size does not
# depend on the qubit count.
#
# The observable is a single Z in the middle of the register and the circuit is shallow, so after
# `nlayers` layers the support is confined to a light cone narrower than every register scanned here.
# The surviving sum is then the same at every qubit count -- only the width of the Pauli string
# integer changes -- and any qubit dependence in the timings is the cost of carrying wider strings.
#
# `stride` sets how far apart the two qubits of a ZZ rotation are. The connectivity is a chain either
# way, so the workload is identical term for term; what changes is the layout of the gate's mask bits.
# At stride 1 the four bits form one contiguous run; at stride 2 they form two runs. The radix tail
# sort treats a run as one digit, so stride 2 is the multi-digit case, and comparing the two isolates
# that from everything else.
#
# usage: julia -tN bench/wide_scan.jl <project_path> <out.csv> <label> [nlayers] [qubits] [strides]
#
# Run it through bench/capped.sh. See that script for why an in-process memory check is not enough.
using Pkg
Pkg.activate(ARGS[1]; io=devnull)
using PauliPropagation
using PauliPropagation.Performance
using Base.Threads

const THETA = 0.1
const MIN_ABS_COEFF = 0.0

# Each qubit count asks `getinttype` for a width it has not seen, which defines a new primitive integer
# type and recompiles the whole propagation pipeline for it. That compile, not the Pauli sum, is the
# allocation that grows dangerously: expanding operations on an integer hundreds of machine words wide
# costs the compiler superlinear memory, and no term-count guard can see it coming.
const MAX_BITS = 8192          # 4096 qubits

# Checked after every layer, not after the circuit: an untruncated sum on a topology with more than
# nearest-neighbour connectivity can multiply by a thousand in a single layer, and a check that only
# runs at the end has already let that happen.
const MAX_TERMS = 200_000

label = ARGS[3]
nlayers = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 4
qubits = length(ARGS) >= 5 ? parse.(Int, split(ARGS[5], ",")) :
         [128, 256, 512, 768, 896, 1024, 1056, 1088, 1280, 1536, 2048]
strides = length(ARGS) >= 6 ? parse.(Int, split(ARGS[6], ",")) : [1, 2]

# A chain whose bonds join qubits `k` apart. Relabelling a chain does not change the propagation, so
# every stride gives the same sum; only the gate masks move.
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
