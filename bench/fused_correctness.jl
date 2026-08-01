# The fused vector path against the stock path, over the cases where the two differ structurally.
#
# The byte-local mask only engages above `_MIN_LOCAL_BYTES`, so qubit counts on both sides of that
# are covered; the radix tail sort only engages for a tail past `_MIN_RADIX_TAIL` whose gate mask
# splits into few enough runs, so strides giving one, two and three runs are covered, with the tail
# sort forced on and off.
#
# Comparison is exact: the fused path truncates during gate application rather than after it, so the
# two agree term for term only when nothing is truncated by coefficient. Weight truncation is safe to
# compare, since it depends on the Pauli string alone.
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

# Imaginary rotations branch on the *commuting* condition, so nearly every term branches at every
# gate and an untruncated sum doubles per gate. The circuit is therefore kept to a handful of gates
# around the middle of the register, which is enough to exercise the shared gate loop, and the size is
# asserted rather than assumed.
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
