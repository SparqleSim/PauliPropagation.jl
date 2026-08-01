# Depth scan of the 6x6 tilted-field Ising setup.
#
# `fullword` rounds the Pauli string type up to a whole number of 64-bit words, which at 36 qubits
# turns UInt72 into UInt128. Both occupy the same 16-byte array slot, so only codegen changes.
#
# usage: julia -tN bench/depth_scan.jl <project_path> <out.csv> <label> [max_depth] [fullword]
# Run through bench/capped.sh.
using Pkg
Pkg.activate(ARGS[1]; io=devnull)
using PauliPropagation
using PauliPropagation.Performance
using Base.Threads

const NX, NY = 6, 6
const MIN_ABS_COEFF = 1e-6

# checked every layer so a blowup is reported rather than allocated
const MAX_TERMS = 12_000_000

label = ARGS[3]
max_depth = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 18
fullword = length(ARGS) >= 5 ? parse(Bool, ARGS[5]) : false

termtype(nq) = fullword ? getinttype(cld(2 * nq, 64) * 32) : getinttype(nq)

function tfimlayer(nx, ny; dt=0.05, h=1.0, J=0.5)
    nq = nx * ny
    layer = PauliRotation[]
    rxlayer!(layer, nq)
    rzlayer!(layer, nq)
    rzzlayer!(layer, rectangletopology(nx, ny))
    thetas = ones(countparameters(layer)) * dt * 2
    thetas[getparameterindices(layer, PauliRotation, [:X])] .*= h
    thetas[getparameterindices(layer, PauliRotation, [:Z])] .*= h
    thetas[getparameterindices(layer, PauliRotation, [:Z, :Z])] .*= J
    return layer, thetas
end

# a two-site observable at the middle of the lattice, as in example2.jl
function startsum(::Type{TT}) where {TT}
    q1 = (NY ÷ 2) * NX + (NX ÷ 2)
    return VectorPauliSum(NX * NY, TT[symboltoint(TT, [:Z, :Z], [q1, q1 + 1])], [1.0])
end

function runpoint(nlayers, ::Type{TT}, reps) where {TT}
    layer, thetas = tfimlayer(NX, NY)
    best = Inf
    local pc
    for rep in 1:(reps+1)   # the first pass warms up compilation and is not scored
        pc = PropagationCache(startsum(TT))
        t = @timed for _ in 1:nlayers
            Performance.propagate!(layer, pc, thetas; min_abs_coeff=MIN_ABS_COEFF)
            length(pc) > MAX_TERMS && error("term cap exceeded at nl=$nlayers: $(length(pc))")
        end
        rep > 1 && (best = min(best, t.time))
        GC.gc()
    end
    return (time=best, len=length(pc), overlap=overlapwithzero(pc))
end

rssgb() = Sys.maxrss() / 2^30

TT = termtype(NX * NY)
out = open(ARGS[2], "w")
println(out, "label,fullword,nl,nq,threads,inttype,sizeof,alignedsizeof,time,len,overlap")
println("$label: $(NX)x$(NY), $TT ($(sizeof(TT))B in a $(Base.aligned_sizeof(TT))B slot)")

for nl in 1:max_depth
    r = runpoint(nl, TT, nl >= 15 ? 1 : 2)   # deep points scatter ~8%, but cost too much to repeat
    println(out, "$label,$fullword,$nl,$(NX * NY),$(nthreads()),$TT,$(sizeof(TT)),$(Base.aligned_sizeof(TT)),$(r.time),$(r.len),$(r.overlap)")
    flush(out)
    println("nl=$(rpad(nl, 2)) $(rpad(round(r.time, digits=4), 9))s len=$(rpad(r.len, 9)) rss=$(round(rssgb(), digits=2))GB")
    flush(stdout)
end
close(out)
