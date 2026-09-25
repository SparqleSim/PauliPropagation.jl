###
##
# Resampling: reduces the number of terms in a TermSum by randomly sampling from the coefficient distribution
##
###
# Every strategy lays teeth on the cumulative weight of the terms and keeps each term with the weight
# of the teeth that fall into its slot, so all it asks of the storage is `mapreducecoeffs` and `mapslots!`.

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

    _checktargetsize(resample_func, length(prop_cache), target_size)

    resample_func(prop_cache, target_size, resample_args...; squared, resample_kwargs...)

    return prop_cache
end

# Most resamplers fold each incoming term into its own slot, so they can only ever shrink the sum.
# Resamplers that draw independently, like multinomial_resample!, overload this to accept any size.
function _checktargetsize(resample_func, active_size, target_size)
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
    weight_func = squared ? abs2 : abs
    total_weight = mapreducecoeffs(weight_func, +, prop_cache; thread)
    weight_per_draw = total_weight / target_size

    # the draws are teeth at random positions, in ascending order so that a slot counts its draws by two searches
    sorted_draws = sort!(rand(typeof(total_weight), target_size) .* total_weight)
    new_coeff_func(coeff, slot_start, slot_end) = _compute_new_coeff(_count_draws(sorted_draws, slot_start, slot_end), weight_per_draw, coeff, squared)

    mapslots!(weight_func, new_coeff_func, prop_cache; thread)
    return truncate!(prop_cache; min_abs_coeff=eps(), thread)
end

# draws are independent of the incoming terms, so any target_size is reachable
_checktargetsize(::typeof(multinomial_resample!), active_size, target_size) = nothing

"""
    systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, calibrate=true, rtol=0.01, atol=0, thread=true)

Low variance resampling technique that returns unique terms.
The number of surviving terms is often close to, and generally at most, `target_size`.
See `calibrate`/`rtol`/`atol` for tuning how closely the comb step is chosen to hit `target_size` unique survivors.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared::Bool=false, calibrate=true, rtol=0.01, atol=0, thread::Bool=true, kwargs...)
    weight_func = squared ? abs2 : abs
    total_weight = mapreducecoeffs(weight_func, +, prop_cache; thread)

    # a step of total_weight / target_size generally keeps fewer than target_size unique terms, so the step is scaled toward that many
    comb_step = calibrate ? _calibrate_prob_step(weight_func, prop_cache, total_weight, target_size; rtol, atol, thread) : total_weight / target_size
    comb_offset = rand() * comb_step
    new_coeff_func(coeff, slot_start, slot_end) = _compute_new_coeff(_count_combteeth(comb_step, comb_offset, slot_start, slot_end), comb_step, coeff, squared)

    mapslots!(weight_func, new_coeff_func, prop_cache; thread)
    return truncate!(prop_cache; min_abs_coeff=eps(), thread)
end

# scales the comb step toward keeping `target_size` unique terms; when resampling we can overshoot if rtol is small
function _calibrate_prob_step(weight_func::W, prop_cache::AbstractPropagationCache, total_weight, target_size; rtol::Real, atol::Real, thread::Bool) where {W}
    # the weights and tolerances are always real, also for complex coefficients
    RT = typeof(total_weight)

    # a comb of step `1 / inv_step` keeps a term with probability `min(1, weight * inv_step)`, so these sum to the
    # mean number of unique samples (or something close to it); `inv_step` is an argument since it changes below
    expected_n_unique(inv_step) = mapreducecoeffs(coeff -> min(1.0, weight_func(coeff) * inv_step), +, prop_cache; thread)

    inv_step = target_size / total_weight

    tolsatisfied(r) = ((1.0 - rtol) * target_size - atol) / target_size <= r <= 1.0 - eps(RT)
    for i in 1:5
        ratio = expected_n_unique(inv_step) / target_size

        if tolsatisfied(ratio)
            return 1 / inv_step
        end

        inv_step /= ratio
    end
    return 1 / inv_step
end

"""
    semideterministic_systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, thread=true)

Terms whose weight exceeds the average per-slot weight `total_weight / target_size` are always kept;
the remaining slots are filled by systematic comb resampling over what is left, folded into each term's own slot.
`squared=true` is disallowed.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function semideterministic_systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, thread::Bool=true, kwargs...)
    if squared
        throw(ArgumentError("semideterministic_systematic_resample! does not support squared=true."))
    end

    @assert 0 < target_size <= length(prop_cache) "target_size must be between 1 and length(prop_cache)"

    # a term above the average weight per slot is kept as it is, and takes no slot on the comb that runs over the rest
    keep_threshold = mapreducecoeffs(abs, +, prop_cache; thread) / target_size
    is_kept(coeff) = abs(coeff) > keep_threshold
    weight_func(coeff) = is_kept(coeff) ? zero(keep_threshold) : abs(coeff)

    n_comb_slots = target_size - mapreducecoeffs(is_kept, +, prop_cache; init=0, thread)
    comb_step = n_comb_slots > 0 ? mapreducecoeffs(weight_func, +, prop_cache; thread) / n_comb_slots : zero(keep_threshold)
    comb_offset = rand() * comb_step
    new_coeff_func(coeff, slot_start, slot_end) = is_kept(coeff) ? coeff : _compute_new_coeff(_count_combteeth(comb_step, comb_offset, slot_start, slot_end), comb_step, coeff, squared)

    mapslots!(weight_func, new_coeff_func, prop_cache; thread)
    return truncate!(prop_cache; min_abs_coeff=eps(), thread)
end

# how many teeth of a comb of `comb_step`, shifted by `comb_offset`, fall into `[slot_start, slot_end)`; a step of zero lays no teeth
function _count_combteeth(comb_step, comb_offset, slot_start, slot_end)
    iszero(comb_step) && return zero(comb_step)
    return floor((slot_end - comb_offset) / comb_step) - floor((slot_start - comb_offset) / comb_step)
end

# how many of the ascending `sorted_draws` fall into `[slot_start, slot_end)`
_count_draws(sorted_draws, slot_start, slot_end) = searchsortedfirst(sorted_draws, slot_end) - searchsortedfirst(sorted_draws, slot_start)

# the coefficient a term keeps for `n_teeth` teeth worth `weight_per_tooth` each, with the sign it had;
# when resampling squared, the weight is the absolute square of the coefficient
_compute_new_coeff(n_teeth, weight_per_tooth, coeff, squared::Bool) =
    squared ? sqrt(n_teeth * weight_per_tooth) * sign(coeff)^2 : n_teeth * weight_per_tooth * sign(coeff)


## SLOTS ON THE CUMULATIVE WEIGHT
"""
    mapslots!(weight_func, new_coeff_func, prop_cache::AbstractPropagationCache; thread=true)

Every term takes a slot as wide as `weight_func(coeff)` on the cumulative weight, in the order of the terms,
and is given the coefficient `new_coeff_func(coeff, slot_start, slot_end)` in place.
`thread=false` runs on the calling thread alone.
"""
mapslots!(weight_func::W, new_coeff_func::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {W,F} =
    _mapslots!(StorageType(prop_cache), weight_func, new_coeff_func, prop_cache; thread)

function _mapslots!(::DictStorage, weight_func::W, new_coeff_func::F, prop_cache::AbstractPropagationCache; kwargs...) where {W,F}
    main_sum = mainsum(prop_cache)
    if _hasdictinternals(storage(main_sum))
        _mapslots_internals!(weight_func, new_coeff_func, storage(main_sum))
        return prop_cache
    end

    slot_end = zero(real(numcoefftype(prop_cache)))
    for (term, coeff) in main_sum
        slot_start = slot_end
        slot_end += weight_func(coeff)
        set!(main_sum, term, new_coeff_func(coeff, slot_start, slot_end))
    end

    return prop_cache
end

function _mapslots!(::ArrayStorage, weight_func::W, new_coeff_func::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {W,F}
    active_coeffs = activecoeffs(prop_cache)

    # the slot ends are the cumulative weights, in the auxiliary coefficients when those are real
    slot_ends = _realweightbuffer(coefficients(auxsum(prop_cache)), active_coeffs)
    AK.map!(weight_func, slot_ends, active_coeffs; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
    AK.accumulate!(+, slot_ends; init=zero(eltype(slot_ends)), max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)

    AK.foreachindex(active_coeffs; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do term_index
        coeff = active_coeffs[term_index]
        slot_end = slot_ends[term_index]
        active_coeffs[term_index] = new_coeff_func(coeff, slot_end - weight_func(coeff), slot_end)
    end

    return prop_cache
end


# The slots of a zone follow the slots of all earlier zones. Zone weights are first reduced with
# the map-reduce primitive, then each zone maps its local slots with the appropriate offset.
function _mapslots!(::MultiSumStorage, weight_func::W, new_coeff_func::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {W,F}
    total_zone_weight(zonecache) = mapreducecoeffs(weight_func, +, zonecache; thread=false)
    zone_weights = _zonevalues(total_zone_weight, real(numcoefftype(prop_cache)), prop_cache, thread)
    zone_slot_starts = pushfirst!(cumsum(zone_weights), zero(eltype(zone_weights)))

    map_zone_slots!(zone_id) = _map_shifted_slots!(weight_func, new_coeff_func, zonecaches(prop_cache)[zone_id], zone_slot_starts[zone_id])
    _eachzone(map_zone_slots!, prop_cache, thread)

    return prop_cache
end

# `mapslots!` on one zone, with its slots starting at `zone_slot_start` instead of at zero.
function _map_shifted_slots!(weight_func::W, new_coeff_func::F, zonecache, zone_slot_start) where {W,F}
    shifted_new_coeff_func(coeff, slot_start, slot_end) = new_coeff_func(coeff, zone_slot_start + slot_start, zone_slot_start + slot_end)
    return mapslots!(weight_func, shifted_new_coeff_func, zonecache; thread=false)
end

# a real-valued buffer of length(coeffs), in the memory of `dst` when the coefficients are real
function _realweightbuffer(dst, coeffs)
    if eltype(coeffs) <: Real
        return view(dst, 1:length(coeffs))
    else
        return similar(coeffs, real(eltype(coeffs)))
    end
end
