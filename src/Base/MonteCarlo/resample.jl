###
##
# Resampling: reduces the number of terms in a TermSum by randomly sampling from the coefficient distribution
##
###
# Every strategy gives each term an interval of [0, total_weight) as wide as its weight, the intervals in the order of
# the terms, lays teeth over the intervals (a comb at a random offset, or random draws), and keeps each term with the
# weight of the teeth in its interval.
#
# A strategy weighs the terms with `_weigh_terms`, which returns their total weight and their intervals, chooses its
# teeth from that total, and hands the intervals and the teeth to `_walk_intervals!`, which gives every term its new
# coefficient and drops the terms without teeth.

## RE-SAMPLING
"""
    resample(tsum::AbstractTermSum, target_size::Integer; resample_func=nothing, squared=false, thread=true, kwargs...)

Resamples `tsum` down (close) to `target_size` terms.
Renormalizes the survivors so that the sum stays an unbiased estimator of incoming sum.
If `squared=true`, resampling is performed on the absolute square of the coefficients and
is not an unbiased estimator of the incoming sum.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function resample(tsum::AbstractTermSum, target_size::Integer, resample_args...; kwargs...)
    return resample!(deepcopy(tsum), target_size, resample_args...; kwargs...)
end

"""
    resample!(tsum::AbstractTermSum, target_size::Integer; thread=true, kwargs...)
    resample!(prop_cache::AbstractPropagationCache, target_size::Integer; thread=true, kwargs...)

In-place version of `resample`. See `resample` for details.
"""
function resample!(tsum::AbstractTermSum, target_size::Integer, resample_args...; kwargs...)
    prop_cache = resample!(PropagationCache(tsum), target_size, resample_args...; kwargs...)

    # extract the original tsum
    return extractsum!(prop_cache, tsum)
end

function resample!(prop_cache::AbstractPropagationCache, target_size, resample_args...; resample_func=nothing, squared=false, resample_kwargs...)
    @assert target_size > 0 "target_size must be positive"

    # resample_func is expected to take prop_cache and target_size as arguments
    # anything else needs to be wrapped into a closure

    if isnothing(resample_func)
        if !squared
            # faster and almost as accurate as calibrated systematic_resample!,
            # but not compatible with 2-norm sampling
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

# Most resamplers keep each incoming term at most once, so they can only ever shrink the sum.
# Resamplers that draw independently, like multinomial_resample!, overload this to accept any size.
function _check_target_size(resample_func, active_size, target_size)
    if target_size > active_size
        throw(ArgumentError("$resample_func cannot grow $active_size terms to target_size $target_size. Use multinomial_resample!."))
    end
    return
end

"""
    multinomial_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, thread=true)

Draws `target_size` terms with replacement, each with probability proportional to its weight, and keeps every term drawn with the weight of all of its draws.
The terms stay where they are, so at most `target_size` of them survive and a sorted sum stays sorted.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function multinomial_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared::Bool=false, thread::Bool=true, kwargs...)
    total_weight, intervals = _weigh_terms(_get_weight_func(squared), prop_cache; thread)
    weight_per_draw = total_weight / target_size

    # the draws are teeth at random positions, sorted so that the draws below a position are found by a search
    sorted_draws = sort!(rand(typeof(total_weight), target_size) .* total_weight)
    draw_count_func(position) = searchsortedfirst(sorted_draws, position) - 1
    new_coeff_func(coeff, n_draws) = _compute_new_coeff(n_draws, weight_per_draw, coeff, squared)

    return _walk_intervals!(intervals, draw_count_func, new_coeff_func, prop_cache; thread)
end

# draws are independent of the incoming terms, so any target_size is reachable
_check_target_size(::typeof(multinomial_resample!), active_size, target_size) = nothing

"""
    systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, calibrate=true, rtol=0.01, atol=0, thread=true)

Low variance resampling technique that returns unique terms.
The number of surviving terms is often close to, and generally at most, `target_size`.
See `calibrate`/`rtol`/`atol` for tuning how closely the comb spacing is chosen to hit `target_size` unique survivors.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared::Bool=false, calibrate=true, rtol=0.01, atol=1, thread::Bool=true, kwargs...)
    weight_func = _get_weight_func(squared)
    total_weight, intervals = _weigh_terms(weight_func, prop_cache; thread)

    comb_spacing = _get_systematic_spacing(weight_func, prop_cache, total_weight, target_size; calibrate, rtol, atol, thread)
    teeth_count_func = _build_teeth_count_func(comb_spacing)
    new_coeff_func(coeff, n_teeth) = _compute_new_coeff(n_teeth, comb_spacing, coeff, squared)

    return _walk_intervals!(intervals, teeth_count_func, new_coeff_func, prop_cache; thread)
end

# A comb spacing of total_weight / target_size generally keeps fewer than target_size unique terms, so a calibrated comb
# narrows the spacing until it keeps about that many.
function _get_systematic_spacing(weight_func::W, prop_cache, total_weight, target_size; calibrate::Bool, rtol::Real, atol::Real, thread::Bool) where {W}
    if calibrate
        return _calibrate_comb_spacing(weight_func, prop_cache, total_weight, target_size; rtol, atol, thread)
    end
    return total_weight / target_size
end

# The comb spacing that keeps `target_size` unique terms in expectation, to within the tolerances. A comb keeps every
# term that weighs at least its spacing and every lighter one with probability weight / spacing. That expected count is
# concave in the inverse spacing, so Newton iterations from `total_weight / target_size`, which keeps at most
# `target_size`, never overshoot.
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

# how many terms weigh at least `spacing`, and the weight of all the others; `spacing` is an argument since it changes
# in the loop above
function _count_heavy_and_weigh_light(weight_func::W, spacing, prop_cache::AbstractPropagationCache; thread::Bool) where {W}
    is_heavy(coeff) = weight_func(coeff) >= spacing
    function light_weight(coeff)
        if is_heavy(coeff)
            return zero(spacing)
        end
        return weight_func(coeff)
    end
    n_heavy, total_light_weight, _ = _count_and_weigh_chunks(is_heavy, light_weight, prop_cache; thread)
    return n_heavy, total_light_weight
end

"""
    semideterministic_systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, thread=true)

Terms whose weight exceeds the average weight per surviving term, `total_weight / target_size`, are always kept;
the rest of `target_size` is filled by systematic comb resampling over the other terms, and a term that several teeth
hit is kept once with their combined weight.
`squared=true` is disallowed.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function semideterministic_systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, thread::Bool=true, kwargs...)
    if squared
        throw(ArgumentError("semideterministic_systematic_resample! does not support squared=true."))
    end

    @assert 0 < target_size <= length(prop_cache) "target_size must be between 1 and length(prop_cache)"

    # a term above the average weight per surviving term is kept as it is, with an empty interval under the comb
    # that runs over the others
    keep_threshold = mapreducecoeffs(abs, +, prop_cache; thread) / target_size
    is_kept(coeff) = abs(coeff) > keep_threshold
    function comb_weight(coeff)
        if is_kept(coeff)
            return zero(keep_threshold)
        end
        return abs(coeff)
    end

    # one pass counts the kept terms and weighs the others, whose intervals share the teeth the kept terms leave
    n_kept, total_comb_weight, comb_intervals = _count_and_weigh_terms(is_kept, comb_weight, prop_cache; thread)
    comb_spacing = _get_comb_spacing(total_comb_weight, target_size - n_kept)
    teeth_count_func = _build_teeth_count_func(comb_spacing)

    # both coefficients are computed and one is selected, since a branch on the scattered kept terms is often mispredicted
    function new_coeff_func(coeff, n_teeth)
        comb_coeff = _compute_new_coeff(n_teeth, comb_spacing, coeff, false)
        return ifelse(is_kept(coeff), coeff, comb_coeff)
    end

    return _walk_intervals!(comb_intervals, teeth_count_func, new_coeff_func, prop_cache; thread)
end

# the weight of a term's interval: its absolute value, or its absolute square when resampling squared
function _get_weight_func(squared::Bool)
    if squared
        return abs2
    end
    return abs
end

# the spacing of a comb that lays `n_teeth` teeth over `total_weight`; without teeth, the spacing is zero
function _get_comb_spacing(total_weight, n_teeth)
    if n_teeth > 0
        return total_weight / n_teeth
    end
    return zero(total_weight)
end

# Builds the function that counts the teeth below a position in [0, total_weight), for a comb of `comb_spacing` laid at
# a uniformly random offset, the one random number of a systematic resampling; every call lays a new comb. An interval
# holds the teeth below its end less those below its start. The spacing is inverted once, so that every interval
# multiplies where it would divide.
function _build_teeth_count_func(comb_spacing)
    teeth_per_weight = _get_teeth_per_weight(comb_spacing)
    offset = rand(typeof(comb_spacing))
    return position -> floor(position * teeth_per_weight - offset)
end

# a comb of spacing zero lays no teeth
function _get_teeth_per_weight(comb_spacing)
    if iszero(comb_spacing)
        return zero(comb_spacing)
    end
    return inv(comb_spacing)
end

# the coefficient a term keeps for `n_teeth` teeth worth `weight_per_tooth` each, with the sign it had;
# when resampling squared, the weight is the absolute square of the coefficient
function _compute_new_coeff(n_teeth, weight_per_tooth, coeff, squared::Bool)
    if squared
        return sqrt(n_teeth * weight_per_tooth) * sign(coeff)^2
    end
    return _apply_sign(n_teeth * weight_per_tooth, coeff)
end

# `magnitude` with the sign of a real `coeff`, which copysign sets in fewer instructions than a multiplication by
# `sign(coeff)`, or with the phase of a complex one
_apply_sign(magnitude, coeff::Real) = copysign(magnitude, coeff)
_apply_sign(magnitude, coeff) = magnitude * sign(coeff)


## WEIGHING THE TERMS, CHUNK BY CHUNK
# Weighing the terms and walking their intervals are two passes over the terms, both chunk by chunk. A chunk is a
# stretch of terms that one thread walks: one per task of an array on the CPU, one per zone of a multi sum, and a
# single one for any other storage. The first pass weighs every chunk, and the weights of the chunks before a chunk
# tell it where its intervals start, so that in the second pass every chunk walks its intervals without waiting for the
# others.

# Where the interval of every term lies: each is as wide as `weight_func` of the term's coefficient and starts where the
# interval of the term before it ends, from where the intervals of its chunk start.
struct _Intervals{W,V}
    weight_func::W
    chunk_interval_starts::V
end

# the total weight of the terms, and their intervals
function _weigh_terms(weight_func::W, prop_cache::AbstractPropagationCache; thread::Bool=true) where {W}
    _, total_weight, intervals = _count_and_weigh_terms(_count_none, weight_func, prop_cache; thread)
    return total_weight, intervals
end

_count_none(coeff) = false

# how many terms `count_func` accepts, found in the same pass, the total weight of the terms, and their intervals
function _count_and_weigh_terms(count_func::C, weight_func::W, prop_cache::AbstractPropagationCache; thread::Bool=true) where {C,W}
    n_counted, total_weight, chunk_weights = _count_and_weigh_chunks(count_func, weight_func, prop_cache; thread)
    chunk_interval_starts = similar(chunk_weights)
    interval_start = zero(eltype(chunk_weights))
    for chunk_id in eachindex(chunk_weights)
        chunk_interval_starts[chunk_id] = interval_start
        interval_start += chunk_weights[chunk_id]
    end
    return n_counted, total_weight, _Intervals(weight_func, chunk_interval_starts)
end

# How many terms `count_func` accepts, the sum of `weight_func` over all terms, and that sum for every chunk, in the
# order the walk takes them.
_count_and_weigh_chunks(count_func::C, weight_func::W, prop_cache::AbstractPropagationCache; thread::Bool=true) where {C,W} =
    _count_and_weigh_chunks(StorageType(prop_cache), count_func, weight_func, prop_cache; thread)

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

# an array on the CPU is one chunk per task; an array elsewhere is one chunk, reduced twice
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

# a multi sum is one chunk per zone, every zone counted and weighed on its own thread
function _count_and_weigh_chunks(::MultiSumStorage, count_func::C, weight_func::W, prop_cache::AbstractPropagationCache; thread::Bool) where {C,W}
    function count_and_weigh_zone(zonecache)
        n_counted, zone_weight, _ = _count_and_weigh_chunks(count_func, weight_func, zonecache; thread=false)
        return (n_counted, zone_weight)
    end
    zone_counts_and_weights = _zonevalues(count_and_weigh_zone, Tuple{Int,real(numcoefftype(prop_cache))}, prop_cache, thread)
    zone_weights = map(last, zone_counts_and_weights)
    return sum(first, zone_counts_and_weights), sum(zone_weights), zone_weights
end

# How many of coeffs[range] `count_func` accepts, and the sum of `weight_func` over them,
# in two accumulators that vectorize where a pair would not.
@inline function _count_and_weigh_range(count_func::C, weight_func::W, coeffs, range) where {C,W}
    n_counted = 0
    total_weight = zero(real(eltype(coeffs)))
    @inbounds @simd for ii in range
        n_counted += count_func(coeffs[ii])
        total_weight += weight_func(coeffs[ii])
    end
    return n_counted, total_weight
end


## WALKING THE SLOTS
# Walks the intervals in the order of the terms, gives every term `new_coeff_func(coeff, n_teeth)` for the
# `n_teeth = teeth_count_func(interval_end) - teeth_count_func(interval_start)` in its interval, and drops the terms
# whose new coefficient is zero. The kept terms stay in the order they had. `intervals` must come from weighing the
# same terms.
function _walk_intervals!(intervals::_Intervals, teeth_count_func::T, new_coeff_func::F, prop_cache::AbstractPropagationCache; thread::Bool) where {T,F}
    return _walk_intervals!(StorageType(prop_cache), intervals.weight_func, teeth_count_func, new_coeff_func, prop_cache, intervals.chunk_interval_starts; thread)
end

_has_zero_coeff(term, coeff) = iszero(coeff)

# a dictionary gives its terms their new coefficients in one walk and then drops those without
function _walk_intervals!(::StorageType, weight_func::W, teeth_count_func::T, new_coeff_func::F, prop_cache::AbstractPropagationCache, chunk_interval_starts; thread::Bool) where {W,T,F}
    main_sum = mainsum(prop_cache)
    interval_end = only(chunk_interval_starts)
    teeth_below_end = teeth_count_func(interval_end)
    for (term, coeff) in main_sum
        teeth_below_start = teeth_below_end
        interval_end += weight_func(coeff)
        teeth_below_end = teeth_count_func(interval_end)
        set!(main_sum, term, new_coeff_func(coeff, teeth_below_end - teeth_below_start))
    end
    return _truncate!(_has_zero_coeff, prop_cache; thread)
end

# One task walks the intervals of an array on the CPU and compacts in place, since it writes at or behind the term it
# just read; several tasks do so each in their chunk and then copy what they kept one after the other into the
# auxiliary arrays.
function _walk_intervals!(::ArrayStorage, weight_func::W, teeth_count_func::T, new_coeff_func::F, prop_cache::AbstractPropagationCache, chunk_interval_starts; thread::Bool) where {W,T,F}
    if !_iscpuarray(prop_cache)
        return _walk_intervals_on_device!(weight_func, teeth_count_func, new_coeff_func, prop_cache, only(chunk_interval_starts); thread)
    end

    task_partitioner, n_tasks = _preparetasks(activesize(prop_cache), thread)
    if length(chunk_interval_starts) != n_tasks
        throw(ArgumentError("the terms were weighed in $(length(chunk_interval_starts)) chunks but are walked in $n_tasks"))
    end
    if n_tasks > 1
        return _walk_intervals_in_tasks!(weight_func, teeth_count_func, new_coeff_func, prop_cache, chunk_interval_starts, task_partitioner, n_tasks)
    end

    main_sum = mainsum(prop_cache)
    n_kept, n_sorted_kept = _walk_intervals_and_compact!(weight_func, teeth_count_func, new_coeff_func, terms(main_sum), coefficients(main_sum),
        1, activesize(prop_cache), only(chunk_interval_starts), sortedprefix(main_sum))
    setactivesize!(prop_cache, n_kept)
    setsortedprefix!(main_sum, n_sorted_kept)
    return prop_cache
end

function _walk_intervals_in_tasks!(weight_func::W, teeth_count_func::T, new_coeff_func::F, prop_cache, chunk_interval_starts, task_partitioner, n_tasks::Int) where {W,T,F}
    main_terms, main_coefficients, aux_terms, aux_coefficients = _mainauxarrays(prop_cache)
    n_sorted = sortedprefix(mainsum(prop_cache))

    kept_counts = Vector{Int}(undef, n_tasks)
    sorted_kept_counts = Vector{Int}(undef, n_tasks)
    function walk_chunk!(task_id)
        chunk = task_partitioner[task_id]
        kept_counts[task_id], sorted_kept_counts[task_id] = _walk_intervals_and_compact!(weight_func, teeth_count_func, new_coeff_func,
            main_terms, main_coefficients, chunk.start, chunk.stop, chunk_interval_starts[task_id], n_sorted)
    end
    _eachtask(walk_chunk!, n_tasks)

    offsets = _offsetsfromcounts(kept_counts)
    function copy_kept!(task_id)
        chunk_start = task_partitioner[task_id].start
        copyto!(aux_terms, offsets[task_id], main_terms, chunk_start, kept_counts[task_id])
        copyto!(aux_coefficients, offsets[task_id], main_coefficients, chunk_start, kept_counts[task_id])
    end
    _eachtask(copy_kept!, n_tasks)

    return _commitwrite!(prop_cache, offsets[end] - 1, sum(sorted_kept_counts))
end

# Walks terms[lo:hi] on intervals from `interval_start` on, gives every term the coefficient for the teeth in its
# interval, and moves the terms it keeps to the front of the range. The teeth below an interval's start are those below
# the end of the interval before, so each position is counted once. Returns the number of kept terms and how many of
# them came from the first `n_sorted`. A dropped term is written too and overwritten by the next, so that random drops
# cost no mispredicted branch.
@inline function _walk_intervals_and_compact!(weight_func::W, teeth_count_func::T, new_coeff_func::F, terms, coefficients, lo, hi, interval_start, n_sorted) where {W,T,F}
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

# An array off the CPU finds every interval from a scan of the weights, in the auxiliary coefficients when those are
# real, gives every term its coefficient independently, and then drops those without.
function _walk_intervals_on_device!(weight_func::W, teeth_count_func::T, new_coeff_func::F, prop_cache, interval_start; thread::Bool) where {W,T,F}
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

    return _truncate!(_has_zero_coeff, prop_cache; thread)
end

# a real-valued buffer of length(coeffs), in the memory of `dst` when the coefficients are real
function _get_real_buffer(dst, coeffs)
    if eltype(coeffs) <: Real
        return view(dst, 1:length(coeffs))
    else
        return similar(coeffs, real(eltype(coeffs)))
    end
end

# every zone walks on its own thread, from where the zones before it end
function _walk_intervals!(::MultiSumStorage, weight_func::W, teeth_count_func::T, new_coeff_func::F, prop_cache::AbstractPropagationCache, zone_interval_starts; thread::Bool) where {W,T,F}
    if length(zone_interval_starts) != nzones(prop_cache)
        throw(ArgumentError("the terms were weighed in $(length(zone_interval_starts)) chunks but are walked in $(nzones(prop_cache)) zones"))
    end

    function walk_zone!(zone_id)
        zonecache = zonecaches(prop_cache)[zone_id]
        _walk_intervals!(StorageType(zonecache), weight_func, teeth_count_func, new_coeff_func, zonecache, (zone_interval_starts[zone_id],); thread=false)
    end
    _eachzone(walk_zone!, prop_cache, thread)

    return _syncsums!(prop_cache)
end
