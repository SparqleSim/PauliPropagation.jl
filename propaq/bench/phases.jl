###
##
# Where the time of the library's fused vector path goes, gate by gate, on a large sum.
#
# Propagates a saved `propaq-benchmarks` circuit up to its last Trotter step, then times the last
# step one rotation at a time, split into the three passes the fused path makes: the scan that
# tests every term and appends the products, the XOR sort of the appended tail, and the merge that
# rewrites the head against it. The scan is what a transposed index replaces, and the merge is what
# a hash table replaces, so the split says what either is worth before it is built.
#
# It also counts what Propaq's emit precheck would have declined, `|c sin θ| < cutoff` before the
# product exists, and how many products land on a term already in the sum.
#
#   julia --project=. -t1 propaq/bench/phases.jl circuits/6x6_steps18.json
#
# `CUTOFF` comes from the environment.
##
###

using PauliPropagation
using PauliPropagation.PropagationBase
using PauliPropagation.PropagationBase: _mainauxarrays, _xorplan, _tailscratch, _xorsorttail!, _mergesortedhead!
using Printf

const PERF = PauliPropagation.Performance

include(joinpath(@__DIR__, "papercircuits.jl"))

# `xorsortedtailmerge!` with its two passes timed and the truncated outputs counted
function timedtailmerge!(prop_cache, xor_mask, truncfunc, thread::Bool)
    n_old = sortedprefix(mainsum(prop_cache))
    n_new = activesize(prop_cache)
    n_tail = n_new - n_old
    n_tail == 0 && return (0.0, 0.0, 0)

    main_terms, main_coeffs, aux_terms, aux_coeffs = _mainauxarrays(prop_cache)
    groups = _xorplan(xor_mask, main_terms)
    groups === nothing && error("the tail is not XOR-sortable here")

    a_terms = view(main_terms, n_old+1:n_new)
    a_coeffs = view(main_coeffs, n_old+1:n_new)
    buf_terms, buf_coeffs = _tailscratch(aux_terms, aux_coeffs, n_new, n_tail, main_terms, main_coeffs)
    b_terms = view(buf_terms, 1:n_tail)
    b_coeffs = view(buf_coeffs, 1:n_tail)

    t_sort = @elapsed tail_terms, tail_coeffs = _xorsorttail!(groups, a_terms, a_coeffs, b_terms, b_coeffs; thread)

    n_truncated = Threads.Atomic{Int}(0)
    counting(pstr, coeff) = (t = truncfunc(pstr, coeff); t && Threads.atomic_add!(n_truncated, 1); t)

    t_merge = @elapsed _mergesortedhead!(prop_cache, aux_terms, aux_coeffs, main_terms, main_coeffs,
        n_old, tail_terms, tail_coeffs, n_tail, counting, thread, Val(true))

    return t_sort, t_merge, n_truncated[]
end

function timedlayer(prop_cache, layer, thetas, cutoff::Float64, thread::Bool)
    t_scan = t_sort = t_merge = 0.0
    visited = branched = declined = collided = 0

    for (gate, theta) in zip(layer, thetas)
        n_old = activesize(prop_cache)
        gate_mask = PERF._gatemask(PauliPropagation.symboltoint(PauliPropagation.paulitype(prop_cache), gate.symbols, gate.qinds),
            PauliPropagation.terms(mainsum(prop_cache)))
        truncfunc(pstr, coeff) = PERF._coefftruncfunc(pstr, coeff; min_abs_coeff=cutoff, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing)

        t_scan += @elapsed PERF._fusedapplytruncaterotation!(prop_cache, gate_mask, cos(theta), sin(theta), Inf, Val(:PauliRotation); thread)
        n_tail = activesize(prop_cache) - n_old
        tail_coeffs = view(PauliPropagation.coefficients(mainsum(prop_cache)), n_old+1:n_old+n_tail)

        visited += n_old
        branched += n_tail
        declined += count(c -> abs(c) < cutoff, tail_coeffs)

        t_s, t_m, n_truncated = timedtailmerge!(prop_cache, PERF._plainmask(gate_mask), truncfunc, thread)
        t_sort += t_s
        t_merge += t_m
        collided += n_old + n_tail - activesize(prop_cache) - n_truncated
    end

    return (; t_scan, t_sort, t_merge, visited, branched, declined, collided)
end

function main(file::String; cutoff::Float64, thread::Bool)
    circuit, thetas, obs, steps = loadproblem(file)
    n_layer = length(circuit) ÷ steps
    # the circuit is applied back to front, so the last step timed is the first `n_layer` gates
    layer = circuit[1:n_layer]
    layer_thetas = thetas[1:n_layer]

    prop_cache = PropagationCache(VectorPauliSum(obs))
    PERF.propagate!(circuit[n_layer+1:end], prop_cache, thetas[n_layer+1:end]; min_abs_coeff=cutoff, thread)
    n_start = length(prop_cache)

    # compile on a copy of the first gate, then time the whole step
    timedlayer(deepcopy(prop_cache), layer[1:1], layer_thetas[1:1], cutoff, thread)
    GC.gc()
    r = timedlayer(prop_cache, layer, layer_thetas, cutoff, thread)

    total = r.t_scan + r.t_sort + r.t_merge
    @printf("%s, last step, %d gates, %d -> %d terms, %d thread(s)\n", basename(file), n_layer, n_start, length(prop_cache), thread ? Threads.nthreads() : 1)
    @printf("  scan and append  %7.3f s  %4.1f%%   %.2f ns per term visited\n", r.t_scan, 100r.t_scan / total, 1e9r.t_scan / r.visited)
    @printf("  XOR sort tail    %7.3f s  %4.1f%%   %.1f ns per product\n", r.t_sort, 100r.t_sort / total, 1e9r.t_sort / r.branched)
    @printf("  merge and rewrite%7.3f s  %4.1f%%   %.2f ns per term visited\n", r.t_merge, 100r.t_merge / total, 1e9r.t_merge / r.visited)
    @printf("  branching %.2f%% of visits; of the products %.1f%% are below the cutoff at emit, %.1f%% land on an existing term\n",
        100r.branched / r.visited, 100r.declined / r.branched, 100r.collided / r.branched)
    return r
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1]; cutoff=parse(Float64, get(ENV, "CUTOFF", "1e-6")), thread=Threads.nthreads() > 1)
end
