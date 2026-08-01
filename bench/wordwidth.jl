# Cost of rounding a Pauli string type up to a whole number of 64-bit words, on the operations the
# gate loop runs. The array slot is unchanged by the rounding, so only codegen can differ.
#
# usage: julia bench/wordwidth.jl <project_path> <out.csv>
using Pkg
Pkg.activate(ARGS[1]; io=devnull)
using PauliPropagation

const BYTE_BUDGET = 1 << 20
const TARGET = 0.02
const SINK = Ref(UInt(0))

commutesum(a, b) = (c = 0; @inbounds for i in eachindex(a); c += commutes(a[i], b[i]); end; c)
weightsum(a, _) = (c = 0; @inbounds for i in eachindex(a); c += countweight(a[i]); end; c)
prodsum(a, b) = (s = zero(eltype(a)); @inbounds for i in eachindex(a); s ⊻= first(pauliprod(a[i], b[i])); end; s)
ltcount(a, b) = (c = 0; @inbounds for i in eachindex(a); c += a[i] < b[i]; end; c)
xorsum(a, b) = (s = zero(eltype(a)); @inbounds for i in eachindex(a); s ⊻= a[i] ⊻ b[i]; end; s)
hashsum(a, _) = (h = UInt(0); @inbounds for i in eachindex(a); h ⊻= hash(a[i]); end; h)

# `a` is perturbed each iteration against hoisting, and results are accumulated against dead-coding
@noinline function repeatop(f::F, a, b, iters) where {F}
    s = UInt(0)
    @inbounds for _ in 1:iters
        a[1] ⊻= one(eltype(a))
        s ⊻= f(a, b) % UInt
    end
    return s
end

function timeop(f::F, a, b) where {F}
    percall = @timed(repeatop(f, a, b, 1)).time
    iters = clamp(round(Int, TARGET / max(percall, 1e-9)), 1, 100_000)
    best = Inf
    for _ in 1:5
        t = @timed repeatop(f, a, b, iters)
        SINK[] ⊻= t.value
        best = min(best, t.time)
    end
    return best / (length(a) * iters) * 1e9   # ns per string
end

fullwordtype(nq) = getinttype(cld(2 * nq, 64) * 32)

const OPS = (("commutes", commutesum), ("countweight", weightsum), ("pauliprod", prodsum),
    ("lt", ltcount), ("xor", xorsum), ("hash", hashsum))

function measure(::Type{TT}) where {TT}
    n = max(64, BYTE_BUDGET ÷ Base.aligned_sizeof(TT))
    a = TT[rand(TT) for _ in 1:n]
    b = TT[rand(TT) for _ in 1:n]
    return [timeop(f, a, b) for (_, f) in OPS]
end

out = open(ARGS[2], "w")
println(out, "nq,kind,inttype,sizeof,alignedsizeof," * join(first.(OPS), ","))
println("qubits  type            slot    " * join(rpad.(first.(OPS), 12)))

for nq in (36, 40, 49, 56, 64, 100, 200, 400, 900)
    narrow = getinttype(nq)
    wide = fullwordtype(nq)
    for (kind, TT) in (("current", narrow), ("fullword", wide))
        r = measure(TT)
        println(out, "$nq,$kind,$TT,$(sizeof(TT)),$(Base.aligned_sizeof(TT))," * join(r, ","))
        println(rpad(nq, 7), " ", rpad(string(TT)[max(1, end - 13):end], 15), " ",
            rpad("$(Base.aligned_sizeof(TT))B", 7), join(rpad.(round.(r, digits=2), 12)))
        flush(stdout)
    end
    narrow === wide && println("        (already a whole number of words -- unchanged)")
end
close(out)
