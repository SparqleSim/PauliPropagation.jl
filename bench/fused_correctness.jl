# The fused vector path against the stock path, over qubit counts either side of the local-path
# threshold, masks splitting into one, two and three runs, and the radix sort on and off.
#
# Exact comparison needs no coefficient truncation, since the fused path truncates during gate
# application. Weight truncation is fine, depending only on the Pauli string.
#
# usage: julia bench/fused_correctness.jl <project_path>
using Pkg
Pkg.activate(ARGS[1]; io=devnull)
using PauliPropagation
using PauliPropagation.Performance
using PauliPropagation.PropagationBase

function sumsmatch(a, b)
    da = Dict(zip(PauliPropagation.terms(a), PauliPropagation.coefficients(a)))
    db = Dict(zip(PauliPropagation.terms(b), PauliPropagation.coefficients(b)))
    keys(da) == keys(db) || return false
    return all(isapprox(da[k], db[k]; atol=1e-12) for k in keys(da))
end

stridedtopology(nq, k) = [(i, i + k) for i in 1:(nq-k)]

fails = 0
checks = 0

for nq in (8, 40, 200, 600, 1056),          # below and above the 24-byte local-path threshold
    k in (1, 2, 3),                          # one, two and three runs of mask bits
    nl in (3, 4),
    radix in (true, false),
    maxw in (Inf, 6)

    k >= nq && continue
    Performance.USE_RADIX_TAILSORT[] = radix

    TT = getinttype(nq)
    circ = tfitrottercircuit(nq, nl; topology=stridedtopology(nq, k))
    thetas = ones(countparameters(circ)) * 0.1
    obs = symboltoint(TT, [:Z], [max(1, nq ÷ 2)])

    a = PropagationCache(VectorPauliSum(nq, TT[obs], [1.0]))
    Performance.propagate!(circ, a, thetas; min_abs_coeff=0.0, max_weight=maxw)
    b = PropagationCache(VectorPauliSum(nq, TT[obs], [1.0]))
    PauliPropagation.propagate!(circ, b, thetas; min_abs_coeff=0.0, max_weight=maxw)

    global checks += 1
    if !sumsmatch(activesum(a), activesum(b))
        global fails += 1
        println("MISMATCH  nq=$nq stride=$k nl=$nl radix=$radix max_weight=$maxw  " *
                "fused=$(length(a)) stock=$(length(b))")
    end
end

# these branch on the *commuting* condition, so an untruncated sum doubles per gate; keep it short
const IMAG_MAX_TERMS = 100_000

for nq in (40, 600), k in (1, 2), radix in (true, false)
    Performance.USE_RADIX_TAILSORT[] = radix
    TT = getinttype(nq)
    mid = nq ÷ 2
    circ = ImaginaryPauliRotation[]
    for i in mid:(mid+4)
        push!(circ, ImaginaryPauliRotation([:Z, :Z], [i, i + k]))
        push!(circ, ImaginaryPauliRotation([:X], [i]))
    end
    taus = ones(countparameters(circ)) * 0.05
    obs = symboltoint(TT, [:Z], [mid])

    a = PropagationCache(VectorPauliSum(nq, TT[obs], [1.0]))
    Performance.propagate!(circ, a, taus; min_abs_coeff=0.0, heisenberg=false)
    length(a) > IMAG_MAX_TERMS && error("imaginary sum grew to $(length(a)) terms at nq=$nq")
    b = PropagationCache(VectorPauliSum(nq, TT[obs], [1.0]))
    PauliPropagation.propagate!(circ, b, taus; min_abs_coeff=0.0, heisenberg=false)

    global checks += 1
    if !sumsmatch(activesum(a), activesum(b))
        global fails += 1
        println("MISMATCH imaginary  nq=$nq stride=$k radix=$radix  fused=$(length(a)) stock=$(length(b))")
    end
end

Performance.USE_RADIX_TAILSORT[] = true
println(fails == 0 ? "all $checks configurations match" : "$fails of $checks configurations FAILED")
exit(fails == 0 ? 0 : 1)
