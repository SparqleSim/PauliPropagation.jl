###
##
# Resampling: reduces the number of terms in a TermSum by randomly sampling from the coefficient distribution
##
###
# Every strategy lays teeth on the cumulative weight of the terms and keeps each term with the weight
# of the teeth that fall into its slot, so all it asks of the storage is `mapreducecoeffs` and `mapslotsandtruncate!`.

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

    return mapslotsandtruncate!(weight_func, new_coeff_func, _truncatezero, prop_cache; thread)
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

    return mapslotsandtruncate!(weight_func, new_coeff_func, _truncatezero, prop_cache; thread)
end

# The comb step that keeps `target_size` unique terms in expectation, to within the tolerances. A comb keeps every term
# that weighs at least its step and every lighter one with probability weight / step. That expected count is concave in
# the inverse step, so Newton's steps from `total_weight / target_size`, which keeps at most `target_size`, never overshoot.
function _calibrate_prob_step(weight_func::W, prop_cache::AbstractPropagationCache, total_weight, target_size; rtol::Real, atol::Real, thread::Bool) where {W}
    lowest_n_unique = (1 - rtol) * target_size - atol

    step = total_weight / target_size
    for _ in 1:5
        n_heavy, light_weight = _heavyandlight(weight_func, step, prop_cache; thread)
        if n_heavy + light_weight / step >= lowest_n_unique || iszero(light_weight)
            return step
        end
        step = light_weight / (target_size - n_heavy)
    end
    return step
end

# how many terms weigh at least `step`, and the weight of all the others; `step` is an argument since it changes in the loop above
function _heavyandlight(weight_func::W, step, prop_cache::AbstractPropagationCache; thread::Bool) where {W}
    is_heavy(coeff) = weight_func(coeff) >= step
    light_weight(coeff) = is_heavy(coeff) ? zero(step) : weight_func(coeff)
    return _countandweigh(is_heavy, light_weight, prop_cache; thread)
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

    n_kept, comb_weight = _countandweigh(is_kept, weight_func, prop_cache; thread)
    n_comb_slots = target_size - n_kept
    comb_step = n_comb_slots > 0 ? comb_weight / n_comb_slots : zero(keep_threshold)
    comb_offset = rand() * comb_step
    # both are computed and one is selected, since a branch on the scattered kept terms is often mispredicted
    new_coeff_func(coeff, slot_start, slot_end) = ifelse(is_kept(coeff), coeff, _compute_new_coeff(_count_combteeth(comb_step, comb_offset, slot_start, slot_end), comb_step, coeff, squared))

    return mapslotsandtruncate!(weight_func, new_coeff_func, _truncatezero, prop_cache; thread)
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

# a term that no tooth or draw falls into is given a zero coefficient
_truncatezero(term, coeff) = truncatemincoeff(coeff, eps())


## SLOTS ON THE CUMULATIVE WEIGHT
"""
    mapslots!(weight_func, new_coeff_func, prop_cache::AbstractPropagationCache; thread=true)

Every term takes a slot as wide as `weight_func(coeff)` on the cumulative weight, in the order of the terms,
and is given the coefficient `new_coeff_func(coeff, slot_start, slot_end)` in place.
`thread=false` runs on the calling thread alone.
"""
mapslots!(weight_func::W, new_coeff_func::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {W,F} =
    mapslotsandtruncate!(weight_func, new_coeff_func, nothing, prop_cache; thread)

"""
    mapslotsandtruncate!(weight_func, new_coeff_func, truncfunc, prop_cache::AbstractPropagationCache; thread=true)

Like `mapslots!`, but drops every term for which `truncfunc(term, new_coeff)` returns `true`.
The kept terms stay in the order they had.
On a multithreaded CPU array, `weight_func` and `truncfunc` are called twice for every term, so they must return the same for the same arguments each time.
"""
mapslotsandtruncate!(weight_func::W, new_coeff_func::F, truncfunc::G, prop_cache::AbstractPropagationCache; thread::Bool=true) where {W,F,G} =
    _mapslotsandtruncate!(StorageType(prop_cache), weight_func, new_coeff_func, truncfunc, prop_cache; thread)

# A dictionary gives its terms their new coefficients and then truncates.
function _mapslotsandtruncate!(storage::DictStorage, weight_func::W, new_coeff_func::F, truncfunc::G, prop_cache::AbstractPropagationCache; thread::Bool=true) where {W,F,G}
    _mapslots!(storage, weight_func, new_coeff_func, prop_cache)
    return _truncate!(truncfunc, prop_cache; thread)
end

# An array on the CPU truncates as it walks the slots; an array elsewhere walks them and then truncates.
function _mapslotsandtruncate!(storage::ArrayStorage, weight_func::W, new_coeff_func::F, truncfunc::G, prop_cache::AbstractPropagationCache; thread::Bool=true) where {W,F,G}
    if _iscpuarray(prop_cache)
        return _mapslotsandtruncatecpu!(weight_func, new_coeff_func, truncfunc, prop_cache; thread)
    end

    _mapslots!(storage, weight_func, new_coeff_func, prop_cache; thread)
    return _truncate!(truncfunc, prop_cache; thread)
end

# the walks of the storages that truncate after them
function _mapslots!(::DictStorage, weight_func::W, new_coeff_func::F, prop_cache::AbstractPropagationCache; kwargs...) where {W,F}
    main_sum = mainsum(prop_cache)

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

# One task walks the slots from zero and compacts in place, since it writes at or behind the term it just read;
# without a `truncfunc`, every term keeps its place.
function _mapslotsandtruncatecpu!(weight_func::W, new_coeff_func::F, truncfunc::G, prop_cache; thread::Bool=true) where {W,F,G}
    n = activesize(prop_cache)
    task_partitioner, n_tasks = _preparetasks(n, thread)

    if n_tasks == 1
        main_terms, main_coefficients = terms(mainsum(prop_cache)), coefficients(mainsum(prop_cache))
        n_kept, n_sorted_kept = _mapslotsinplace!(weight_func, new_coeff_func, truncfunc, main_terms, main_coefficients, 1, n,
            zero(real(numcoefftype(prop_cache))), sortedprefix(mainsum(prop_cache)), Val(truncfunc !== nothing))
        setactivesize!(prop_cache, n_kept)
        setsortedprefix!(mainsum(prop_cache), n_sorted_kept)
        return prop_cache
    end

    return _mapslotsintasks!(weight_func, new_coeff_func, truncfunc, prop_cache, task_partitioner, n_tasks)
end

# Several tasks first sum the weights of their parts, so that each knows where its slots start,
# then give their terms the new coefficients in place and count what they keep, then write that into the auxiliary arrays.
# Without a `truncfunc`, every term keeps its place and nothing is written.
function _mapslotsintasks!(weight_func::W, new_coeff_func::F, truncfunc::G, prop_cache, task_partitioner, n_tasks::Int) where {W,F,G}
    n_sorted = sortedprefix(mainsum(prop_cache))
    main_terms, main_coefficients, aux_terms, aux_coefficients = _mainauxarrays(prop_cache)
    no_weight = zero(real(numcoefftype(prop_cache)))

    chunk_weights = Vector{typeof(no_weight)}(undef, n_tasks)
    function sum_chunk_weight!(task_id)
        chunk = task_partitioner[task_id]
        chunk_weights[task_id] = sum(weight_func, view(main_coefficients, chunk))
    end
    _eachtask(sum_chunk_weight!, n_tasks)
    chunk_slot_starts = pushfirst!(cumsum(chunk_weights), no_weight)

    kept_counts = Vector{Int}(undef, n_tasks)
    sorted_kept_counts = Vector{Int}(undef, n_tasks)
    function map_and_count!(task_id)
        chunk = task_partitioner[task_id]
        kept_counts[task_id], sorted_kept_counts[task_id] = _mapslotsinplace!(weight_func, new_coeff_func, truncfunc, main_terms, main_coefficients,
            chunk.start, chunk.stop, chunk_slot_starts[task_id], n_sorted, Val(false))
    end
    _eachtask(map_and_count!, n_tasks)

    if truncfunc === nothing
        return prop_cache
    end

    offsets = _offsetsfromcounts(kept_counts)

    written = Vector{Int}(undef, n_tasks)
    function write_kept!(task_id)
        chunk = task_partitioner[task_id]
        written[task_id] = _compactwrite!(truncfunc, aux_terms, aux_coefficients, offsets[task_id], offsets[task_id+1] - 1,
            main_terms, main_coefficients, chunk.start, chunk.stop)
    end
    _eachtask(write_kept!, n_tasks)

    if written != kept_counts
        _throwreplaymismatch()
    end
    return _commitwrite!(prop_cache, offsets[end] - 1, sum(sorted_kept_counts))
end

# Walks terms[lo:hi] on slots from `slot_start` on and gives every term its new coefficient in place.
# With `Compact`, the kept terms move to the front of the range; otherwise every term stays where it is.
# Returns the number of kept terms and how many of them came from the first `n_sorted`.
# A dropped term is written too and overwritten by the next, so that random drops cost no mispredicted branch.
@inline function _mapslotsinplace!(weight_func::W, new_coeff_func::F, truncfunc::G, terms, coefficients, lo, hi,
    slot_start, n_sorted, ::Val{Compact}) where {W,F,G,Compact}

    write_pos = lo
    n_sorted_kept = 0
    slot_end = slot_start

    @inbounds for ii in lo:hi
        term = terms[ii]
        coeff = coefficients[ii]
        slot_start = slot_end
        slot_end += @inline weight_func(coeff)
        new_coeff = @inline new_coeff_func(coeff, slot_start, slot_end)
        is_kept = truncfunc === nothing || !(@inline truncfunc(term, new_coeff))

        if Compact
            terms[write_pos] = term
            coefficients[write_pos] = new_coeff
        else
            coefficients[ii] = new_coeff
        end
        write_pos += is_kept
        n_sorted_kept += is_kept & (ii <= n_sorted)
    end

    return write_pos - lo, n_sorted_kept
end

# Writes the terms of terms[lo:hi] that `truncfunc` keeps from `write_start` on and returns how many it kept.
# As in `_mapslotsinplace!`, a dropped term is written and then overwritten, but nothing is written past `write_stop`, where the next task's part begins.
@inline function _compactwrite!(truncfunc::G, output_terms, output_coefficients, write_start, write_stop, terms, coefficients, lo, hi) where {G}
    write_pos = write_start

    @inbounds for ii in lo:hi
        term = terms[ii]
        coeff = coefficients[ii]
        is_kept = !(@inline truncfunc(term, coeff))

        if write_pos <= write_stop
            output_terms[write_pos] = term
            output_coefficients[write_pos] = coeff
        end
        write_pos += is_kept
    end

    return write_pos - write_start
end


# The slots of a zone follow the slots of all earlier zones. Zone weights are first reduced with
# the map-reduce primitive, then each zone maps its local slots with the appropriate offset.
function _mapslotsandtruncate!(::MultiSumStorage, weight_func::W, new_coeff_func::F, truncfunc::G, prop_cache::AbstractPropagationCache; thread::Bool=true) where {W,F,G}
    total_zone_weight(zonecache) = mapreducecoeffs(weight_func, +, zonecache; thread=false)
    zone_weights = _zonevalues(total_zone_weight, real(numcoefftype(prop_cache)), prop_cache, thread)
    zone_slot_starts = pushfirst!(cumsum(zone_weights), zero(eltype(zone_weights)))

    map_zone_slots!(zone_id) = _map_shifted_slots!(weight_func, new_coeff_func, truncfunc, zonecaches(prop_cache)[zone_id], zone_slot_starts[zone_id])
    _eachzone(map_zone_slots!, prop_cache, thread)

    return _syncsums!(prop_cache)
end

# `mapslotsandtruncate!` on one zone, with its slots starting at `zone_slot_start` instead of at zero.
function _map_shifted_slots!(weight_func::W, new_coeff_func::F, truncfunc::G, zonecache, zone_slot_start) where {W,F,G}
    shifted_new_coeff_func(coeff, slot_start, slot_end) = @inline new_coeff_func(coeff, zone_slot_start + slot_start, zone_slot_start + slot_end)
    return mapslotsandtruncate!(weight_func, shifted_new_coeff_func, truncfunc, zonecache; thread=false)
end

# a real-valued buffer of length(coeffs), in the memory of `dst` when the coefficients are real
function _realweightbuffer(dst, coeffs)
    if eltype(coeffs) <: Real
        return view(dst, 1:length(coeffs))
    else
        return similar(coeffs, real(eltype(coeffs)))
    end
end


## COUNT AND WEIGHT IN ONE PASS
# How many coefficients `count_func` accepts, and the sum of `weight_func` over all of them.
_countandweigh(count_func::C, weight_func::W, prop_cache::AbstractPropagationCache; thread::Bool=true) where {C,W} =
    _countandweigh(StorageType(prop_cache), count_func, weight_func, prop_cache; thread)

# a dictionary, or any other storage walked in order, sums both in the same walk
function _countandweigh(::StorageType, count_func::C, weight_func::W, prop_cache::AbstractPropagationCache; thread::Bool=true) where {C,W}
    no_weight = zero(real(numcoefftype(prop_cache)))
    count_and_weight(coeff) = (count_func(coeff), weight_func(coeff))
    return mapreducecoeffs(count_and_weight, _addpairs, prop_cache; init=(0, no_weight), neutral=(0, no_weight), thread)
end

# An array on the CPU is counted and weighed by every task with two accumulators, which vectorize where a pair does not.
# An array elsewhere is reduced twice.
function _countandweigh(::ArrayStorage, count_func::C, weight_func::W, prop_cache::AbstractPropagationCache; thread::Bool=true) where {C,W}
    if !_iscpuarray(prop_cache)
        return mapreducecoeffs(count_func, +, prop_cache; init=0, thread), mapreducecoeffs(weight_func, +, prop_cache; thread)
    end

    active_coeffs = coefficients(prop_cache)
    task_partitioner, n_tasks = _preparetasks(length(active_coeffs), thread)
    counts = zeros(Int, n_tasks)
    weights = zeros(real(numcoefftype(prop_cache)), n_tasks)
    function count_and_weigh_part!(task_id)
        count = 0
        weight = zero(eltype(weights))
        @inbounds @simd for ii in task_partitioner[task_id]
            count += count_func(active_coeffs[ii])
            weight += weight_func(active_coeffs[ii])
        end
        counts[task_id] = count
        weights[task_id] = weight
    end
    _eachtask(count_and_weigh_part!, n_tasks)

    return sum(counts), sum(weights)
end

# every zone counts and weighs on its own thread
function _countandweigh(::MultiSumStorage, count_func::C, weight_func::W, prop_cache::AbstractPropagationCache; thread::Bool=true) where {C,W}
    zone_count_and_weight(zonecache) = _countandweigh(count_func, weight_func, zonecache; thread=false)
    zone_values = _zonevalues(zone_count_and_weight, Tuple{Int,real(numcoefftype(prop_cache))}, prop_cache, thread)
    return reduce(_addpairs, zone_values)
end

_addpairs(pair1, pair2) = (pair1[1] + pair2[1], pair1[2] + pair2[2])
