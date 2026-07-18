###
##
# Variants of `applymergetruncate!` for VectorPauliSum that truncate during gate application.
# This may yield slightly different results compared to normal functionality.
# This will only function for CPU, currently.
##
###

"""
    applymergetruncate!(gate::PauliRotation, prop_cache::VectorPauliPropagationCache, theta; fused::Bool=false, kwargs...)

Fused, task-partitioned overload of `applymergetruncate!` for `PauliRotation` -- see file header.
Only used when `fused=true`; otherwise falls through (via `invoke`) to default behavior.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.PauliRotation, prop_cache::PauliPropagation.VectorPauliPropagationCache, theta;
    fused::Bool=false,
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing,
    thread::Bool=true, kwargs...)

    # invoke function from library
    if !fused
        return invoke(PauliPropagation.applymergetruncate!,
            Tuple{PauliPropagation.PauliRotation,PauliPropagation.AbstractPauliPropagationCache,typeof(theta)},
            gate, prop_cache, theta;
            min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
    end

    PauliPropagation._check_qind_range(PauliPropagation.nqubits(prop_cache), gate.qinds)

    if PauliPropagation.activesize(prop_cache) == 0
        return prop_cache
    end

    gate_mask = PauliPropagation.symboltoint(PauliPropagation.paulitype(prop_cache), gate.symbols, gate.qinds)

    truncfunc(pstr, coeff) = _fusedtruncfunc(pstr, coeff; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc)

    # truncats during application of the gate to 1. make merge!() faster and 2. save on the extra truncation!() pass
    _fusedapplytruncaterotation!(prop_cache, gate_mask, cos(theta), sin(theta), PauliPropagation.paulirotationproduct, truncfunc, Val(:PauliRotation); thread)

    PauliPropagation.merge!(prop_cache; thread, truncfunc, kwargs...)

    return prop_cache
end

### Imaginary Pauli Rotation

"""
    applymergetruncate!(gate::ImaginaryPauliRotation, prop_cache::VectorPauliPropagationCache, tau; fused::Bool=false, normalize_coeffs=true, kwargs...)

Fused, task-partitioned overload of `applymergetruncate!` for `ImaginaryPauliRotation`.
Shares its branch-and-write core with the fused `PauliRotation` overload.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.ImaginaryPauliRotation, prop_cache::PauliPropagation.VectorPauliPropagationCache, tau;
    fused::Bool=false, normalize_coeffs::Bool=true,
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing,
    thread::Bool=true, kwargs...)

    if !fused
        return invoke(PauliPropagation.applymergetruncate!,
            Tuple{PauliPropagation.ImaginaryPauliRotation,PauliPropagation.AbstractPauliPropagationCache,typeof(tau)},
            gate, prop_cache, tau;
            normalize_coeffs, min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
    end

    PauliPropagation._check_qind_range(PauliPropagation.nqubits(prop_cache), gate.qinds)

    if PauliPropagation.activesize(prop_cache) == 0
        return prop_cache
    end

    gate_mask = PauliPropagation.symboltoint(PauliPropagation.paulitype(prop_cache), gate.symbols, gate.qinds)

    truncfunc(pstr, coeff) = _fusedtruncfunc(pstr, coeff; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc)

    _fusedapplytruncaterotation!(prop_cache, gate_mask, cosh(tau), sinh(tau), PauliPropagation.imaginarypaulirotationproduct, truncfunc, Val(:ImaginaryPauliRotation); thread)

    PauliPropagation.merge!(prop_cache; thread, truncfunc, kwargs...)

    # This gate assumes we are working in the Schrödinger picture evolving states
    # we normalize by the coefficient of the identity Pauli string for numerical stability
    # getcoeff sums over any duplicates and binary-searches the sorted prefix, so it is both
    # correct and fast here without relying on merge!() having fully deduplicated everything
    if normalize_coeffs
        PauliPropagation.mult!(prop_cache, 1 / PauliPropagation.getcoeff(PauliPropagation.activesum(prop_cache), 0))
    end

    return prop_cache
end

# Shared backend for the two-branching rotation gates
# does a first pass computing the number of new terms, and a second writing them 
function _fusedapplytruncaterotation!(prop_cache::PauliPropagation.VectorPauliPropagationCache, gate_mask::TT, kept_val, new_val, productfunc::PF, truncfunc, ::Val{GateType};
    thread::Bool=true) where {TT,PF,GateType}

    n_old = activesize(prop_cache)

    # Only the first `old_sortedprefix` elements are guaranteed sorted going in (some gates, e.g.
    # CliffordGate, reset it without merging). Compaction is order-preserving, so kept survivors
    # from that sorted prefix stay contiguous at the front of the output -- counting them below
    # gives the new sortedprefix directly, with no separate re-derivation needed.
    old_sortedprefix = sortedprefix(mainsum(prop_cache))

    task_partitioner, n_tasks = PropagationBase._preparetasks(n_old, thread)
    main_terms, main_coeffs, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(prop_cache)

    kept_counts = Vector{Int}(undef, n_tasks)
    new_counts = Vector{Int}(undef, n_tasks)
    sorted_kept_counts = Vector{Int}(undef, n_tasks)

    # dry run: each task counts its own kept (passthrough or surviving kept-branch) and new (surviving
    # branch) output sizes, without writing
    AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
        rng = task_partitioner[task_id]
        kept_counts[task_id], new_counts[task_id], sorted_kept_counts[task_id] = _fusedbranchwrite!(aux_terms, aux_coeffs, 1, aux_terms, aux_coeffs, 1,
            main_terms, main_coeffs, rng.start, rng.stop, gate_mask, kept_val, new_val, productfunc, truncfunc, old_sortedprefix, Val(GateType), Val(false))
    end

    # small serial prefix sums over just the per-task counts (mirrors sortedtailmerge!'s offset bookkeeping)
    kept_offsets = PropagationBase._offsetsfromcounts(kept_counts)
    new_offsets = PropagationBase._offsetsfromcounts(new_counts)
    n_kept = kept_offsets[end] - 1
    n_new = new_offsets[end] - 1
    n_total = n_kept + n_new
    new_sortedprefix = sum(sorted_kept_counts)

    resize_factor = 1.5
    if PauliPropagation.capacity(prop_cache) < n_total
        resize!(prop_cache, round(Int, n_total * resize_factor))
        main_terms, main_coeffs, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(prop_cache)
    end

    # real pass: redo the same walk, now writing each task's output directly into its final
    # position -- kept head at kept_offsets[task], new tail right after it at n_kept+new_offsets[task]
    AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
        rng = task_partitioner[task_id]
        _fusedbranchwrite!(aux_terms, aux_coeffs, kept_offsets[task_id], aux_terms, aux_coeffs, n_kept + new_offsets[task_id],
            main_terms, main_coeffs, rng.start, rng.stop, gate_mask, kept_val, new_val, productfunc, truncfunc, old_sortedprefix, Val(GateType), Val(true))
    end

    PropagationBase._commitwrite!(prop_cache, n_total, new_sortedprefix)

    return prop_cache
end

# flagging condition for PauliRotation or ImaginaryPauliRotation
@inline _branchcondition(::Val{:PauliRotation}, does_commute) = !does_commute
@inline _branchcondition(::Val{:ImaginaryPauliRotation}, does_commute) = does_commute

# Walks terms[lo:hi], branching each term according to `_branchcondition(Val(GateType), ...)` and
# truncating inline. Writes survivors from kept_start/new_start when DoWrite; otherwise only counts
# (dry-run sizing pass). n_sorted_kept counts survivors that originated within the old sorted prefix,
# which stay contiguous at the front of the kept head -- see caller. Returns (n_kept, n_new,
# n_sorted_kept).
@inline function _fusedbranchwrite!(kept_out_terms, kept_out_coeffs, kept_start,
    new_out_terms, new_out_coeffs, new_start,
    terms, coeffs, lo, hi, gate_mask::TT, kept_val, new_val, productfunc::PF, truncfunc::F, old_sortedprefix::Int,
    ::Val{GateType}, ::Val{DoWrite}) where {TT,PF,F,GateType,DoWrite}

    kept_pos = kept_start
    n_sorted_kept = 0
    new_pos = new_start

    @inbounds for ii in lo:hi
        pstr = terms[ii]
        coeff = coeffs[ii]

        does_commute = PauliPropagation.commutes(gate_mask, pstr)
        branches = _branchcondition(Val(GateType), does_commute)

        if !branches
            kept_pos = PropagationBase._writeandadvance!(kept_out_terms, kept_out_coeffs, kept_pos, pstr, coeff, Val(DoWrite))
            ii <= old_sortedprefix && (n_sorted_kept += 1)
        else
            coeff1 = coeff * kept_val
            if !truncfunc(pstr, coeff1)
                kept_pos = PropagationBase._writeandadvance!(kept_out_terms, kept_out_coeffs, kept_pos, pstr, coeff1, Val(DoWrite))
                ii <= old_sortedprefix && (n_sorted_kept += 1)
            end

            new_pstr, sign = productfunc(gate_mask, pstr)
            coeff2 = coeff * new_val * sign
            if !truncfunc(new_pstr, coeff2)
                new_pos = PropagationBase._writeandadvance!(new_out_terms, new_out_coeffs, new_pos, new_pstr, coeff2, Val(DoWrite))
            end
        end
    end

    return (kept_pos - kept_start, new_pos - new_start, n_sorted_kept)
end

### Pauli Noise

"""
    applymergetruncate!(gate::PauliNoise, prop_cache::VectorPauliPropagationCache, lambda; fused::Bool=false, kwargs...)

Fused, task-partitioned overload of `applymergetruncate!` for `PauliNoise` -- see file header. Exact,
since there's nothing to merge. Only used when `fused=true`; otherwise falls through (via `invoke`) to
default behavior.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.PauliNoise, prop_cache::PauliPropagation.VectorPauliPropagationCache, lambda;
    fused::Bool=false,
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing,
    thread::Bool=true, kwargs...)

    if !fused
        return invoke(PauliPropagation.applymergetruncate!,
            Tuple{PauliPropagation.PauliNoise,PauliPropagation.AbstractPauliPropagationCache,typeof(lambda)},
            gate, prop_cache, lambda;
            min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
    end

    PauliPropagation._check_qind_range(PauliPropagation.nqubits(prop_cache), gate.qind)
    PauliPropagation._check_noise_strength(PauliPropagation.PauliNoise, lambda)

    if activesize(prop_cache) == 0
        return prop_cache
    end

    truncfunc(pstr, coeff) = _fusedtruncfunc(pstr, coeff; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc)

    _fusedapplytruncatenoise!(prop_cache, gate, lambda, truncfunc; thread)

    return prop_cache
end

function _fusedapplytruncatenoise!(prop_cache::PauliPropagation.VectorPauliPropagationCache, gate::PauliPropagation.PauliNoise, lambda, truncfunc;
    thread::Bool=true)

    n_old = activesize(prop_cache)

    # pstr identity is unchanged, so the old sorted prefix survives wherever its terms weren't truncated
    old_sortedprefix = sortedprefix(mainsum(prop_cache))

    qind = gate.qind

    task_partitioner, n_tasks = PropagationBase._preparetasks(n_old, thread)
    main_terms, main_coeffs, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(prop_cache)

    kept_counts = Vector{Int}(undef, n_tasks)
    sorted_kept_counts = Vector{Int}(undef, n_tasks)

    # dry run: each task counts its own surviving output size, without writing
    AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
        rng = task_partitioner[task_id]
        kept_counts[task_id], sorted_kept_counts[task_id] = _noisewrite!(aux_terms, aux_coeffs, 1,
            main_terms, main_coeffs, rng.start, rng.stop, gate, qind, lambda, truncfunc, old_sortedprefix, Val(false))
    end

    # small serial prefix sum over just the per-task counts (mirrors sortedtailmerge!'s offset bookkeeping)
    kept_offsets = PropagationBase._offsetsfromcounts(kept_counts)
    n_kept = kept_offsets[end] - 1
    new_sortedprefix = sum(sorted_kept_counts)

    # real pass: redo the same walk, now writing each task's output directly into its final position
    AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
        rng = task_partitioner[task_id]
        _noisewrite!(aux_terms, aux_coeffs, kept_offsets[task_id],
            main_terms, main_coeffs, rng.start, rng.stop, gate, qind, lambda, truncfunc, old_sortedprefix, Val(true))
    end

    PropagationBase._commitwrite!(prop_cache, n_kept, new_sortedprefix)

    return prop_cache
end


@inline function _noisewrite!(out_terms, out_coeffs, out_start, terms, coeffs, lo, hi,
    gate::PauliPropagation.PauliNoise, qind, lambda, truncfunc::F, old_sortedprefix::Int, ::Val{DoWrite}) where {F,DoWrite}

    pos = out_start
    n_sorted_kept = 0
    @inbounds for ii in lo:hi
        pstr = terms[ii]
        new_coeff = PauliPropagation.isdamped(gate, getpauli(pstr, qind)) ? coeffs[ii] * (1 - lambda) : coeffs[ii]
        if !truncfunc(pstr, new_coeff)
            pos = PropagationBase._writeandadvance!(out_terms, out_coeffs, pos, pstr, new_coeff, Val(DoWrite))
            ii <= old_sortedprefix && (n_sorted_kept += 1)
        end
    end

    return (pos - out_start, n_sorted_kept)
end
