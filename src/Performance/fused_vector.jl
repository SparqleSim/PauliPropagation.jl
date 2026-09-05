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
        return _invokedefault(gate, prop_cache, theta;
            min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
    end

    return _fusedrotation!(gate, prop_cache, cos(theta), sin(theta), Val(:PauliRotation);
        min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
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
        return _invokedefault(gate, prop_cache, tau;
            normalize_coeffs, min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
    end

    PauliPropagation._check_qind_range(PauliPropagation.nqubits(prop_cache), gate.qinds)

    # an empty sum has no identity coefficient to normalize by, so return ahead of that too
    if PauliPropagation.activesize(prop_cache) == 0
        return prop_cache
    end

    _fusedrotation!(gate, prop_cache, cosh(tau), sinh(tau), Val(:ImaginaryPauliRotation);
        min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)

    # This gate assumes we are working in the Schrödinger picture evolving states
    # we normalize by the coefficient of the identity Pauli string for numerical stability
    # getcoeff sums over any duplicates and binary-searches the sorted prefix, so it is both
    # correct and fast here without relying on merge!() having fully deduplicated everything
    # scaling the active range rather than the whole cache leaves the trash past it alone
    if normalize_coeffs
        PauliPropagation.activecoeffs(prop_cache) .*= 1 / PauliPropagation.getcoeff(PauliPropagation.activesum(prop_cache), 0)
    end

    return prop_cache
end

### Shared core for the two branching rotation gates

# Applies the gate and truncates in the merge that follows, saving the extra truncate!() pass.
# Products too heavy to ever be kept are dropped as they are made, so they are never written or
# sorted; everything the coefficient decides waits for the merge -- see `_coefftruncfunc`.
function _fusedrotation!(gate, prop_cache, kept_val, new_val, gatetype::Val;
    min_abs_coeff::Real, max_weight::Real, max_freq::Real, max_sins::Real, customtruncfunc,
    thread::Bool, kwargs...)

    PauliPropagation._check_qind_range(PauliPropagation.nqubits(prop_cache), gate.qinds)

    if PauliPropagation.activesize(prop_cache) == 0
        return prop_cache
    end

    gate_mask = _gatemask(PauliPropagation.symboltoint(PauliPropagation.paulitype(prop_cache), gate.symbols, gate.qinds),
        PauliPropagation.terms(mainsum(prop_cache)))

    truncfunc(pstr, coeff) = _coefftruncfunc(pstr, coeff; min_abs_coeff, max_freq, max_sins, customtruncfunc)

    # the tail sort in the merge below needs the new terms in parent order
    sorted_before = sortedprefix(mainsum(prop_cache)) == activesize(prop_cache)

    _fusedapplytruncaterotation!(prop_cache, gate_mask, kept_val, new_val, max_weight, gatetype; thread)

    PropagationBase.xorsortedtailmerge!(prop_cache, _plainmask(gate_mask), sorted_before; thread, truncfunc, kwargs...)

    return prop_cache
end

# Appends every product past the end of the active range, in parent order.
function _fusedapplytruncaterotation!(prop_cache::PauliPropagation.VectorPauliPropagationCache, gate_mask::TT, kept_val, new_val, max_weight, ::Val{GateType};
    thread::Bool=true) where {TT,GateType}

    n_old = activesize(prop_cache)
    task_partitioner, n_tasks = PropagationBase._preparetasks(n_old, thread)

    n_new = if n_tasks == 1
        _serialbranchwrite!(prop_cache, n_old, gate_mask, kept_val, new_val, max_weight, Val(GateType))
    else
        _taskedbranchwrite!(prop_cache, n_old, task_partitioner, n_tasks, gate_mask, kept_val, new_val, max_weight, Val(GateType))
    end

    # the terms already there kept their Pauli strings and their order, so the sorted prefix stands
    setactivesize!(prop_cache, n_old + n_new)

    return prop_cache
end

# One task writes each product as it makes it, so nothing has to be counted first.
# A term makes at most one product, so a walk never outruns the free slots it started with.
function _serialbranchwrite!(prop_cache, n_old::Int, gate_mask, kept_val, new_val, max_weight, gatetype::Val)
    pos = n_old + 1
    lo = 1

    while lo <= n_old
        _growto!(prop_cache, pos)
        terms, coeffs, _, _ = PropagationBase._mainauxarrays(prop_cache)

        hi = min(n_old, lo + PauliPropagation.capacity(prop_cache) - pos)
        pos += _fusedbranchwrite!(terms, coeffs, pos, terms, coeffs, lo, hi, gate_mask, kept_val, new_val, max_weight, gatetype, Val(true))
        lo = hi + 1
    end

    return pos - n_old - 1
end

# Several tasks have to agree on where each of them writes, so a pass counts their products first.
function _taskedbranchwrite!(prop_cache, n_old::Int, task_partitioner, n_tasks::Int, gate_mask, kept_val, new_val, max_weight, gatetype::Val)
    main_terms, main_coeffs, _, _ = PropagationBase._mainauxarrays(prop_cache)

    new_counts = Vector{Int}(undef, n_tasks)

    # dry run: each task counts the products it will append, without writing
    AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
        rng = task_partitioner[task_id]
        new_counts[task_id] = _fusedbranchwrite!(main_terms, main_coeffs, 1, main_terms, main_coeffs, rng.start, rng.stop,
            gate_mask, kept_val, new_val, max_weight, gatetype, Val(false))
    end

    new_offsets = PropagationBase._offsetsfromcounts(new_counts)
    n_new = new_offsets[end] - 1

    # Without a weight limit no product is dropped, so no products means nothing branched.
    # Nothing was scaled either, and the second walk can be skipped.
    if n_new == 0 && isinf(max_weight)
        return 0
    end

    if _growto!(prop_cache, n_old + n_new)
        main_terms, main_coeffs, _, _ = PropagationBase._mainauxarrays(prop_cache)
    end

    # real pass: scale the branching coefficients in place, append each task's products at its offset
    AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
        rng = task_partitioner[task_id]
        _fusedbranchwrite!(main_terms, main_coeffs, n_old + new_offsets[task_id], main_terms, main_coeffs, rng.start, rng.stop,
            gate_mask, kept_val, new_val, max_weight, gatetype, Val(true))
    end

    return n_new
end

# Room for at least `n` terms, in geometric steps so growing costs a copy only now and then.
# Returns whether the arrays moved.
function _growto!(prop_cache, n::Int)
    PauliPropagation.capacity(prop_cache) >= n && return false
    resize!(prop_cache, round(Int, n * 1.5))
    return true
end

# flagging condition for PauliRotation or ImaginaryPauliRotation
@inline _branchcondition(::Val{:PauliRotation}, does_commute) = !does_commute
@inline _branchcondition(::Val{:ImaginaryPauliRotation}, does_commute) = does_commute

# Walks terms[lo:hi], branching each term according to `_branchcondition(Val(GateType), ...)`. A
# branching term keeps its Pauli string and only has its coefficient scaled, so it stays where it is
# and needs no test -- its weight already passed. Its product is written to out_terms/out_coeffs from
# new_start, unless it is too heavy, which merging cannot undo. The output may be the very arrays
# being walked, which is how a product lands past the end of the active range. When DoWrite is false
# nothing is written and the products are only counted. Returns the number of products.
@inline function _fusedbranchwrite!(out_terms, out_coeffs, new_start, terms, coeffs, lo, hi,
    gate_mask::TT, kept_val, new_val, max_weight, ::Val{GateType}, ::Val{DoWrite}) where {TT,GateType,DoWrite}

    new_pos = new_start

    # the pointer a ByteMask reads through is valid only inside this block
    GC.@preserve terms begin
        bytes = _bytesof(terms, gate_mask)

        @inbounds for ii in lo:hi
            does_commute = _gatecommutesat(gate_mask, terms, bytes, ii)
            _branchcondition(Val(GateType), does_commute) || continue

            pstr = terms[ii]
            coeff = coeffs[ii]
            DoWrite && (coeffs[ii] = coeff * kept_val)

            new_pstr, sign = _gateproduct(gate_mask, pstr, bytes, ii)
            if !_truncateweight(new_pstr, max_weight)
                new_pos = PropagationBase._writeandadvance!(out_terms, out_coeffs, new_pos, new_pstr, coeff * new_val * sign, Val(DoWrite))
            end
        end
    end

    return new_pos - new_start
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
        return _invokedefault(gate, prop_cache, lambda;
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

    # One task can filter in place: the walk writes at or behind the term it just read, so source
    # and destination may be the same arrays, and nothing has to be counted first.
    if n_tasks == 1
        n_kept, new_sortedprefix = _noisewrite!(main_terms, main_coeffs, 1,
            main_terms, main_coeffs, 1, n_old, gate, qind, lambda, truncfunc, old_sortedprefix, Val(true))
        setactivesize!(prop_cache, n_kept)
        setsortedprefix!(mainsum(prop_cache), new_sortedprefix)
        return prop_cache
    end

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
