###
##
# Resampling: reduces the number of terms in a TermSum by randomly sampling from the coefficient distribution
##
###
# Every strategy lines the terms up on [0, total_weight), each on an interval as wide as its weight, lays teeth on that
# line (a comb or random draws), and gives each term the weight of the teeth in its interval; terms without teeth are
# dropped. `_weigh_terms` finds where the intervals start and `_set_coeffs_and_drop_zeros!` walks them.

## RE-SAMPLING
"""
    resample(tsum::AbstractTermSum, target_size::Integer; resample_func=nothing, squared=false, thread=true, kwargs...)

Resamples `tsum` down to about `target_size` terms, reweighting the survivors so that the result is an unbiased estimate of `tsum`.
With `squared=true`, terms are sampled by their squared coefficients, and the result is no longer unbiased.
`resample_func` defaults to `semideterministic_systematic_resample!`, or to `multinomial_resample!` when `squared=true`.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function resample(tsum::AbstractTermSum, target_size::Integer, resample_args...; kwargs...)
    resampled = deepcopy(tsum)
    resample!(resampled, target_size, resample_args...; kwargs...)
    return resampled
end

"""
    resample!(tsum::AbstractTermSum, target_size::Integer; thread=true, kwargs...)
    resample!(prop_cache::AbstractPropagationCache, target_size::Integer; thread=true, kwargs...)

In-place version of `resample`. See `resample` for details.
"""
function resample!(tsum::AbstractTermSum, target_size::Integer, resample_args...; kwargs...)
    prop_cache = PropagationCache(tsum)
    resample!(prop_cache, target_size, resample_args...; kwargs...)

    # the resampled terms go back into `tsum`
    extractsum!(prop_cache, tsum)
    return tsum
end

function resample!(prop_cache::AbstractPropagationCache, target_size, resample_args...; resample_func=nothing, squared=false, resample_kwargs...)
    @assert target_size > 0 "target_size must be positive"

    if isnothing(resample_func)
        if !squared
            # about as accurate as the calibrated systematic_resample!, and faster
            resample_func = semideterministic_systematic_resample!
        else
            # 2-norm sampling usually draws a few independent samples
            resample_func = multinomial_resample!
        end
    end

    _check_target_size(resample_func, length(prop_cache), target_size)

    resample_func(prop_cache, target_size, resample_args...; squared, resample_kwargs...)

    return prop_cache
end

# a resampler that keeps each term at most once cannot grow the sum; multinomial_resample! overloads this
function _check_target_size(resample_func, active_size, target_size)
    if target_size > active_size
        throw(ArgumentError("$resample_func cannot grow $active_size terms to target_size $target_size. Use multinomial_resample!."))
    end
    return
end

"""
    multinomial_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, thread=true)

Draws `target_size` times with replacement, each term with probability proportional to its absolute coefficient (squared with `squared=true`), and keeps every drawn term with the weight of its draws.
At most `target_size` terms survive, in their original order, so a sorted sum stays sorted.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function multinomial_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared::Bool=false, thread::Bool=true, kwargs...)
    weight_func = _get_weight_func(squared)
    total_weight, interval_starts = _weigh_terms(weight_func, prop_cache; thread)
    weight_per_draw = total_weight / target_size

    # sorted, so that the draws below a position are found by binary search
    sorted_draws = sort!(rand(typeof(total_weight), target_size) .* total_weight)
    draw_count_func(position) = searchsortedfirst(sorted_draws, position) - 1
    new_coeff_func(coeff, n_draws) = _compute_new_coeff(n_draws, weight_per_draw, coeff, squared)

    _set_coeffs_and_drop_zeros!(weight_func, draw_count_func, new_coeff_func, prop_cache, interval_starts; thread)
    return prop_cache
end

# independent draws can reach any target_size
_check_target_size(::typeof(multinomial_resample!), active_size, target_size) = nothing

"""
    systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, calibrate=true, rtol=0.01, atol=1, thread=true)

Low-variance resampling: an evenly spaced comb at a random offset is laid over the terms' weights, and every term it hits is kept once, with the weight of its teeth.
The number of survivors is usually close to, and rarely more than, `target_size`.
With `calibrate=true`, the comb spacing is narrowed until the expected number of survivors reaches `(1 - rtol) * target_size - atol`.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared::Bool=false, calibrate=true, rtol=0.01, atol=1, thread::Bool=true, kwargs...)
    weight_func = _get_weight_func(squared)
    total_weight, interval_starts = _weigh_terms(weight_func, prop_cache; thread)

    comb_spacing = if calibrate
        _calibrate_comb_spacing(weight_func, prop_cache, total_weight, target_size; rtol, atol, thread)
    else
        total_weight / target_size
    end
    teeth_count_func = _build_teeth_count_func(comb_spacing)
    new_coeff_func(coeff, n_teeth) = _compute_new_coeff(n_teeth, comb_spacing, coeff, squared)

    _set_coeffs_and_drop_zeros!(weight_func, teeth_count_func, new_coeff_func, prop_cache, interval_starts; thread)
    return prop_cache
end

# The comb spacing that keeps `target_size` unique terms in expectation, to within the tolerances. A comb keeps a term
# with probability min(1, weight / spacing), and Newton steps from total_weight / target_size approach the target from
# below.
function _calibrate_comb_spacing(weight_func::W, prop_cache::AbstractPropagationCache, total_weight, target_size; rtol::Real, atol::Real, thread::Bool) where {W}
    lowest_n_unique = (1 - rtol) * target_size - atol

    spacing = total_weight / target_size
    for _ in 1:5
        n_heavy, light_weight = _count_heavy_and_weigh_light(weight_func, spacing, prop_cache; thread)
        if n_heavy + light_weight / spacing >= lowest_n_unique || iszero(light_weight)
            return spacing
        end
        spacing = light_weight / (target_size - n_heavy)
    end
    return spacing
end

# the number of terms weighing at least `spacing` and the total weight of the others; a function of its own, so that
# its closures capture a `spacing` that does not change
function _count_heavy_and_weigh_light(weight_func::W, spacing, prop_cache::AbstractPropagationCache; thread::Bool) where {W}
    is_heavy(coeff) = weight_func(coeff) >= spacing
    function light_weight(coeff)
        if is_heavy(coeff)
            return zero(spacing)
        end
        return weight_func(coeff)
    end
    n_heavy, total_light_weight, _ = _count_and_weigh_terms(is_heavy, light_weight, prop_cache; thread)
    return n_heavy, total_light_weight
end

"""
    semideterministic_systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, thread=true)

Keeps every term whose absolute coefficient exceeds the sum of all absolute coefficients divided by `target_size` as it is, and fills the rest of `target_size` by systematic resampling of the other terms.
`squared=true` is not supported.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function semideterministic_systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, thread::Bool=true, kwargs...)
    if squared
        throw(ArgumentError("semideterministic_systematic_resample! does not support squared=true."))
    end

    @assert 0 < target_size <= length(prop_cache) "target_size must be between 1 and length(prop_cache)"

    # kept terms get empty intervals, so the comb runs over the other terms only
    keep_threshold = mapreducecoeffs(abs, +, prop_cache; thread) / target_size
    is_kept(coeff) = abs(coeff) > keep_threshold
    function comb_weight(coeff)
        if is_kept(coeff)
            return zero(keep_threshold)
        end
        return abs(coeff)
    end

    # one pass counts the kept terms and weighs the others
    n_kept, total_comb_weight, comb_interval_starts = _count_and_weigh_terms(is_kept, comb_weight, prop_cache; thread)
    # without teeth left for the comb, its spacing is zero
    n_comb_teeth = target_size - n_kept
    comb_spacing = if n_comb_teeth > 0
        total_comb_weight / n_comb_teeth
    else
        zero(total_comb_weight)
    end
    teeth_count_func = _build_teeth_count_func(comb_spacing)

    # computes both and selects one, since a branch on the scattered kept terms is often mispredicted
    function new_coeff_func(coeff, n_teeth)
        comb_coeff = _compute_new_coeff(n_teeth, comb_spacing, coeff, false)
        return ifelse(is_kept(coeff), coeff, comb_coeff)
    end

    _set_coeffs_and_drop_zeros!(comb_weight, teeth_count_func, new_coeff_func, prop_cache, comb_interval_starts; thread)
    return prop_cache
end

# a term's weight: the absolute value of its coefficient, or its absolute square when resampling squared
function _get_weight_func(squared::Bool)
    if squared
        return abs2
    end
    return abs
end

# a function counting the teeth below a position, for a comb of `comb_spacing` at a random offset; every call lays a
# new comb
function _build_teeth_count_func(comb_spacing)
    # a comb of spacing zero lays no teeth
    teeth_per_weight = if iszero(comb_spacing)
        zero(comb_spacing)
    else
        inv(comb_spacing)
    end
    offset = rand(typeof(comb_spacing))
    return position -> floor(position * teeth_per_weight - offset)
end

# the new coefficient for `n_teeth` teeth of `weight_per_tooth` each, with the original sign; when resampling squared,
# the weight is the absolute square of the coefficient
function _compute_new_coeff(n_teeth, weight_per_tooth, coeff, squared::Bool)
    if squared
        return sqrt(n_teeth * weight_per_tooth) * sign(coeff)^2
    end
    return _apply_sign(n_teeth * weight_per_tooth, coeff)
end

# `magnitude` with the sign or complex phase of `coeff`; copysign is cheaper than multiplying by sign(coeff)
_apply_sign(magnitude, coeff::Real) = copysign(magnitude, coeff)
_apply_sign(magnitude, coeff) = magnitude * sign(coeff)


## WEIGHING THE TERMS, CHUNK BY CHUNK
# A chunk is a stretch of terms that one thread walks: a task's range of a CPU array, a zone of a multi sum, or the
# whole sum otherwise. Weighing the chunks first tells each one where its intervals start, so that the chunks can then
# be walked in parallel.

# the total weight of the terms, and where the intervals of each chunk start
function _weigh_terms(weight_func::W, prop_cache::AbstractPropagationCache; thread::Bool=true) where {W}
    _, total_weight, interval_starts = _count_and_weigh_terms(_count_none, weight_func, prop_cache; thread)
    return total_weight, interval_starts
end

_count_none(coeff) = false

# as `_weigh_terms`, and how many terms `count_func` accepts, in the same pass
function _count_and_weigh_terms(count_func::C, weight_func::W, prop_cache::AbstractPropagationCache; thread::Bool=true) where {C,W}
    n_counted, total_weight, chunk_weights = _count_and_weigh_chunks(StorageType(prop_cache), count_func, weight_func, prop_cache; thread)
    interval_starts = similar(chunk_weights)
    interval_start = zero(eltype(chunk_weights))
    for chunk_id in eachindex(chunk_weights)
        interval_starts[chunk_id] = interval_start
        interval_start += chunk_weights[chunk_id]
    end
    return n_counted, total_weight, interval_starts
end

# Each storage returns the count, the total weight and the weight of every chunk, in walking order.

# a dictionary is one chunk
function _count_and_weigh_chunks(::StorageType, count_func::C, weight_func::W, prop_cache::AbstractPropagationCache; thread::Bool) where {C,W}
    n_counted = 0
    total_weight = zero(real(numcoefftype(prop_cache)))
    for coeff in coefficients(prop_cache)
        n_counted += count_func(coeff)
        total_weight += weight_func(coeff)
    end
    return n_counted, total_weight, [total_weight]
end

# an array is one chunk per task on the CPU, and one chunk elsewhere
function _count_and_weigh_chunks(::ArrayStorage, count_func::C, weight_func::W, prop_cache::AbstractPropagationCache; thread::Bool) where {C,W}
    coeffs = coefficients(prop_cache)
    if !_iscpuarray(coeffs)
        n_counted = mapreducecoeffs(count_func, +, prop_cache; init=0, thread)
        total_weight = mapreducecoeffs(weight_func, +, prop_cache; thread)
        return n_counted, total_weight, [total_weight]
    end

    task_partitioner, n_tasks = _preparetasks(length(coeffs), thread)
    chunk_counts = Vector{Int}(undef, n_tasks)
    chunk_weights = Vector{real(numcoefftype(prop_cache))}(undef, n_tasks)
    function count_and_weigh_chunk!(task_id)
        chunk_counts[task_id], chunk_weights[task_id] = _count_and_weigh_range(count_func, weight_func, coeffs, task_partitioner[task_id])
    end
    _eachtask(count_and_weigh_chunk!, n_tasks)
    return sum(chunk_counts), sum(chunk_weights), chunk_weights
end

# a multi sum is one chunk per zone, each on its own thread
function _count_and_weigh_chunks(::MultiSumStorage, count_func::C, weight_func::W, prop_cache::AbstractPropagationCache; thread::Bool) where {C,W}
    function count_and_weigh_zone(zonecache)
        n_counted, zone_weight, _ = _count_and_weigh_chunks(StorageType(zonecache), count_func, weight_func, zonecache; thread=false)
        return (n_counted, zone_weight)
    end
    zone_counts_and_weights = _zonevalues(count_and_weigh_zone, Tuple{Int,real(numcoefftype(prop_cache))}, prop_cache, thread)
    zone_weights = map(last, zone_counts_and_weights)
    return sum(first, zone_counts_and_weights), sum(zone_weights), zone_weights
end

# the count and the weight of coeffs[range], in two accumulators, which vectorize where a tuple would not
@inline function _count_and_weigh_range(count_func::C, weight_func::W, coeffs, range) where {C,W}
    n_counted = 0
    total_weight = zero(real(eltype(coeffs)))
    @inbounds @simd for ii in range
        n_counted += count_func(coeffs[ii])
        total_weight += weight_func(coeffs[ii])
    end
    return n_counted, total_weight
end


## SETTING THE COEFFICIENTS AND DROPPING ZEROS
# Gives every term `new_coeff_func(coeff, n_teeth)` for the teeth in its interval and drops the terms whose new
# coefficient is zero; the others keep their order. `interval_starts` must come from `_weigh_terms` with the same
# `weight_func`.
function _set_coeffs_and_drop_zeros!(weight_func::W, teeth_count_func::T, new_coeff_func::F, prop_cache::AbstractPropagationCache, interval_starts; thread::Bool) where {W,T,F}
    _set_coeffs_and_drop_zeros!(StorageType(prop_cache), weight_func, teeth_count_func, new_coeff_func, prop_cache, interval_starts; thread)
    return prop_cache
end

# a dictionary sets all coefficients in one walk, then drops the zeros
function _set_coeffs_and_drop_zeros!(::StorageType, weight_func::W, teeth_count_func::T, new_coeff_func::F, prop_cache::AbstractPropagationCache, chunk_interval_starts; thread::Bool) where {W,T,F}
    main_sum = mainsum(prop_cache)
    interval_end = only(chunk_interval_starts)
    teeth_below_end = teeth_count_func(interval_end)
    for (term, coeff) in main_sum
        teeth_below_start = teeth_below_end
        interval_end += weight_func(coeff)
        teeth_below_end = teeth_count_func(interval_end)
        set!(main_sum, term, new_coeff_func(coeff, teeth_below_end - teeth_below_start))
    end
    filtercoeffs!(!iszero, prop_cache; thread)
    return prop_cache
end

# On the CPU, every task compacts its chunk in place; with several tasks, the kept terms are then copied into the
# auxiliary arrays.
function _set_coeffs_and_drop_zeros!(::ArrayStorage, weight_func::W, teeth_count_func::T, new_coeff_func::F, prop_cache::AbstractPropagationCache, chunk_interval_starts; thread::Bool) where {W,T,F}
    if !_iscpuarray(prop_cache)
        _set_coeffs_and_drop_zeros_on_device!(weight_func, teeth_count_func, new_coeff_func, prop_cache, only(chunk_interval_starts); thread)
        return prop_cache
    end

    task_partitioner, n_tasks = _preparetasks(activesize(prop_cache), thread)
    if length(chunk_interval_starts) != n_tasks
        throw(ArgumentError("the terms were weighed in $(length(chunk_interval_starts)) chunks but are walked in $n_tasks"))
    end
    _set_coeffs_and_drop_zeros_in_tasks!(weight_func, teeth_count_func, new_coeff_func, prop_cache, chunk_interval_starts, task_partitioner, n_tasks)
    return prop_cache
end

function _set_coeffs_and_drop_zeros_in_tasks!(weight_func::W, teeth_count_func::T, new_coeff_func::F, prop_cache, chunk_interval_starts, task_partitioner, n_tasks::Int) where {W,T,F}
    main_terms, main_coefficients, aux_terms, aux_coefficients = _mainauxarrays(prop_cache)
    n_sorted = sortedprefix(mainsum(prop_cache))

    kept_counts = Vector{Int}(undef, n_tasks)
    sorted_kept_counts = Vector{Int}(undef, n_tasks)
    function walk_chunk!(task_id)
        chunk = task_partitioner[task_id]
        kept_counts[task_id], sorted_kept_counts[task_id] = _set_coeffs_and_compact!(weight_func, teeth_count_func, new_coeff_func,
            main_terms, main_coefficients, chunk.start, chunk.stop, chunk_interval_starts[task_id], n_sorted)
    end
    _eachtask(walk_chunk!, n_tasks)

    # a single chunk is already compacted in place
    if n_tasks == 1
        setactivesize!(prop_cache, only(kept_counts))
        setsortedprefix!(mainsum(prop_cache), only(sorted_kept_counts))
        return prop_cache
    end

    offsets = _offsetsfromcounts(kept_counts)
    function copy_kept!(task_id)
        chunk_start = task_partitioner[task_id].start
        copyto!(aux_terms, offsets[task_id], main_terms, chunk_start, kept_counts[task_id])
        copyto!(aux_coefficients, offsets[task_id], main_coefficients, chunk_start, kept_counts[task_id])
    end
    _eachtask(copy_kept!, n_tasks)

    _commitwrite!(prop_cache, offsets[end] - 1, sum(sorted_kept_counts))
    return prop_cache
end

# Sets the coefficients of terms[lo:hi] and moves the kept terms to the front of the range. Returns how many were kept
# and how many of those were among the first `n_sorted`. A dropped term is written too and overwritten by the next, so
# that random drops cost no mispredicted branch.
@inline function _set_coeffs_and_compact!(weight_func::W, teeth_count_func::T, new_coeff_func::F, terms, coefficients, lo, hi, interval_start, n_sorted) where {W,T,F}
    write_pos = lo
    n_sorted_kept = 0
    interval_end = interval_start
    teeth_below_end = teeth_count_func(interval_end)

    @inbounds for ii in lo:hi
        term = terms[ii]
        coeff = coefficients[ii]
        teeth_below_start = teeth_below_end
        interval_end += weight_func(coeff)
        teeth_below_end = teeth_count_func(interval_end)
        new_coeff = new_coeff_func(coeff, teeth_below_end - teeth_below_start)

        terms[write_pos] = term
        coefficients[write_pos] = new_coeff
        is_kept = !iszero(new_coeff)
        write_pos += is_kept
        n_sorted_kept += is_kept & (ii <= n_sorted)
    end

    return write_pos - lo, n_sorted_kept
end

# Off the CPU, a prefix sum of the weights gives every interval, each term is set independently, and then the zeros are
# dropped.
function _set_coeffs_and_drop_zeros_on_device!(weight_func::W, teeth_count_func::T, new_coeff_func::F, prop_cache, interval_start; thread::Bool) where {W,T,F}
    active_coeffs = activecoeffs(prop_cache)
    interval_ends = _get_real_buffer(coefficients(auxsum(prop_cache)), active_coeffs)
    AK.map!(weight_func, interval_ends, active_coeffs; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
    AK.accumulate!(+, interval_ends; init=zero(eltype(interval_ends)), max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)

    AK.foreachindex(active_coeffs; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do term_index
        coeff = active_coeffs[term_index]
        interval_end = interval_start + interval_ends[term_index]
        n_teeth = teeth_count_func(interval_end) - teeth_count_func(interval_end - weight_func(coeff))
        active_coeffs[term_index] = new_coeff_func(coeff, n_teeth)
    end

    filtercoeffs!(!iszero, prop_cache; thread)
    return prop_cache
end

# a real-valued buffer of length(coeffs), reusing `dst` when the coefficients are real
function _get_real_buffer(dst, coeffs)
    if eltype(coeffs) <: Real
        return view(dst, 1:length(coeffs))
    else
        return similar(coeffs, real(eltype(coeffs)))
    end
end

# each zone walks on its own thread, starting where the zones before it end
function _set_coeffs_and_drop_zeros!(::MultiSumStorage, weight_func::W, teeth_count_func::T, new_coeff_func::F, prop_cache::AbstractPropagationCache, zone_interval_starts; thread::Bool) where {W,T,F}
    if length(zone_interval_starts) != nzones(prop_cache)
        throw(ArgumentError("the terms were weighed in $(length(zone_interval_starts)) chunks but are walked in $(nzones(prop_cache)) zones"))
    end

    function walk_zone!(zone_id)
        zonecache = zonecaches(prop_cache)[zone_id]
        _set_coeffs_and_drop_zeros!(StorageType(zonecache), weight_func, teeth_count_func, new_coeff_func, zonecache, (zone_interval_starts[zone_id],); thread=false)
    end
    _eachzone(walk_zone!, prop_cache, thread)

    _syncsums!(prop_cache)
    return prop_cache
end
