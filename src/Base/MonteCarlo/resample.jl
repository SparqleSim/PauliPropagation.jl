###
##
# Resampling: reduces the number of terms in a TermSum by randomly sampling from the coefficient distribution
##
###
# TODO: Resampling currently only works for array-based TermSums
using StatsBase

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

    if target_size > activesize(prop_cache)
        throw(ArgumentError("target_size must be less than the current active size of the prop_cache."))
    end
    # resample_func is expected to take prop_cache and target_size as arguments
    # anything else needs to be wrapped into a closure

    if isnothing(resample_func)
        if !squared
            # faster and almost as accurate as calibrated systematic_resample!, 
            # but not compatible with 2-norm sampling
            resample_func = semideterministic_systematic_resample!
        else
            # likely we just draw one sample
            # and probably we don't want coefficients to be pre-merged
            resample_func = multinomial_resample!
        end
    end

    resample_func(prop_cache, target_size, resample_args...; squared, resample_kwargs...)

    return prop_cache
end

# the auxsum's raw arrays (write destination) and the cache's active terms/coeffs (read source)
_resamplearrays(prop_cache::AbstractPropagationCache) = (terms(auxsum(prop_cache)), coefficients(auxsum(prop_cache)),
    activeterms(prop_cache), activecoeffs(prop_cache))

# a real-valued scratch buffer of length(coeffs) for cumulative weights
# always real, even for complex coeffs. 
# Reuses `dst`'s memory if possible, else allocates a new array
function _realweightbuffer(dst, coeffs)
    if eltype(coeffs) <: Real
        return view(dst, 1:length(coeffs))
    else
        return similar(coeffs, real(eltype(coeffs)))
    end
end

## This the naive resampling where one draw's a random number per sample.
function multinomial_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, thread::Bool=true, kwargs...)

    dst_terms, dst_coeffs, src_terms, src_coeffs = _resamplearrays(prop_cache)

    _multinomial_resample!(dst_terms, dst_coeffs, src_terms, src_coeffs, target_size; squared, thread)

    swapsums!(prop_cache)
    setactivesize!(prop_cache, target_size)
    # draws are i.i.d. samples from the source distribution, so the result carries no sorted order
    setsortedprefix!(mainsum(prop_cache), 0)

    return prop_cache
end

function _multinomial_resample!(dst_terms, dst_coeffs, terms, coeffs, target_size; squared=false, thread::Bool=true)
    power = squared ? 2 : 1

    # Compute the cumulative distribution
    # TODO: make non-allocating
    cum_probs = coeffcumsum(coeffs, power; thread)
    total_weight = cum_probs[end]

    dst_terms_view = view(dst_terms, 1:target_size)
    AK.foreachindex(dst_terms_view; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do i
        # Sample a random number in [0, total_weight)
        r = rand() * total_weight

        # we don't parallelize this because we are already threading over the outer loop
        idx = searchsortedfirst(cum_probs, r)

        dst_terms[i] = terms[idx]
        # for power=2 (squared=true) this is the sqrt of the weight share, so that |dst_coeff|^2
        # (not |dst_coeff|) equals the assigned share and the squared 2-norm is what's conserved
        dst_coeffs[i] = (total_weight / target_size)^(1 / power) * sign(coeffs[idx])^power
    end

    return nothing
end

"""
    systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, calibrate=true, rtol=0.01, atol=0, thread=true)

Low variance resampling technique that returns unique terms.
The number of surviving terms is often close to, and generally at most, `target_size`.
See `calibrate`/`rtol`/`atol` for tuning how closely the comb step is chosen to hit `target_size` unique survivors.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared::Bool=false, calibrate=true, rtol=0.01, atol=0, thread::Bool=true, kwargs...)
    dst_terms, dst_coeffs, src_terms, src_coeffs = _resamplearrays(prop_cache)

    _systematic_resample!(dst_terms, dst_coeffs, src_terms, src_coeffs, target_size; squared, calibrate, rtol, atol, thread)

    swapsums!(prop_cache)
    # active size does not need to be changed.

    # dst_terms[i] = src_terms[i] for every i (only coefficients change)
    # old sorted prefix is still the new prefix
    setsortedprefix!(mainsum(prop_cache), sortedprefix(auxsum(prop_cache)))

    # now filter out the exactly 0.0 and set active size
    truncate!(prop_cache; min_abs_coeff=eps(), thread)

    return prop_cache
end

function _systematic_resample!(dst_terms, dst_coeffs, terms, coeffs, target_size; squared::Bool=false, calibrate=true, rtol=0.02, atol=1, thread::Bool=true)
    power = squared ? 2 : 1

    # cumulative weights are always real; reuses dst_coeffs's memory when possible (see _realweightbuffer)
    cum_probs = _realweightbuffer(dst_coeffs, coeffs)
    AK.map!(c -> abs(c)^power, cum_probs, coeffs; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
    AK.accumulate!(+, cum_probs; init=zero(eltype(cum_probs)), max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
    total_weight = cum_probs[end]

    # Normally: step = total_weight / target_size
    # but this will generally produce less unique terms
    # scale the step to go toward target_size many unique terms
    if calibrate
        step = _calibrate_prob_step(coeffs, total_weight, target_size; power, rtol, atol, thread)
    else
        step = total_weight / target_size
    end

    # the source of randomness shifting the comb
    offset = rand() * step


    # We iterate over INPUT terms. Each term is processed once.
    # The "let" block is necessary because otherwise "step" get boxed (wow again)
    let step=step, offset=offset, power=power
        AK.foreachindex(terms; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do i

            # c_start is technically stoch_cum_probs[i-1],
            # but we thread, so need to recompute from coeff
            c_end = cum_probs[i]
            c_start = c_end - abs(coeffs[i])^power

            # Calculate how many "comb teeth" fall into [c_start, c_end)
            lower_idx = floor((c_start - offset) / step)
            upper_idx = floor((c_end - offset) / step)
            n_copies = upper_idx - lower_idx

            dst_terms[i] = terms[i]
            # write one entry with the combined weight
            # n_copies can be 0, and it will be filtered later
            dst_coeffs[i] = (n_copies * step)^(1 / power) * sign(coeffs[i])^power
        end
    end

    return nothing
end


function _calibrate_prob_step(coeffs, total_weight, target_size; power=1, rtol::Real=0.02, atol::Real=0, thread::Bool=true)
    # coeffs may be complex, but the sums/tolerances computed here are always real
    RT = real(eltype(coeffs))

    # this function estimates the mean number of unique samples (or something close to it)
    # when resampling we can overshoot if rtol is small
    getcurrentsum(cs, inv_s) = AK.mapreduce(c -> min(1.0, abs(c)^power * inv_s), +, cs; init=zero(RT), neutral=zero(RT), max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)

    inv_step = target_size / total_weight

    tolsatisfied(r) = ((1.0 - rtol) * target_size - atol) / target_size <= r <= 1.0 - eps(RT)
    for i in 1:5
        current_sum = getcurrentsum(coeffs, inv_step)

        ratio = current_sum / target_size

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

    @assert 0 < target_size <= activesize(prop_cache) "target_size must be between 1 and activesize(prop_cache)"

    dst_terms, dst_coeffs, src_terms, src_coeffs = _resamplearrays(prop_cache)
    # src_coeffs may be complex; weights/counts computed below are always real
    RT = real(eltype(src_coeffs))

    # compute the average weight capacity of the resampled terms
    # if a term is above this threshold, it is always kept, otherwise it is resampled
    total_weight = AK.mapreduce(abs, +, src_coeffs; init=zero(RT), neutral=zero(RT), max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
    threshold = total_weight / target_size

    # how many terms are taken deterministically
    n_det = AK.mapreduce(c -> abs(c) > threshold, +, src_coeffs; init=0, neutral=0, max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)

    # how many slots are left for the stochastic part
    n_stoch = target_size - n_det

    # cumulative weights are always real; reuses dst_coeffs's memory when possible (see _realweightbuffer)
    stoch_cum_probs = _realweightbuffer(dst_coeffs, src_coeffs)
    AK.map!(c -> abs(c) > threshold ? zero(eltype(stoch_cum_probs)) : abs(c), stoch_cum_probs, src_coeffs; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
    AK.accumulate!(+, stoch_cum_probs; init=zero(eltype(stoch_cum_probs)), max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
    total_stoch_weight = stoch_cum_probs[end]

    # step and offset of the comb for systematic resampling; step is only meaningful if there
    # are stochastic slots left to fill
    step = n_stoch > 0 ? total_stoch_weight / n_stoch : zero(total_stoch_weight)
    offset = rand() * step

    # single pass: each term is either kept deterministically or assigned its comb share
    let step = step, offset = offset, threshold = threshold, n_stoch = n_stoch
        AK.foreachindex(src_terms; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do i
            coeff = src_coeffs[i]
            abs_coeff = abs(coeff)
            dst_terms[i] = src_terms[i]

            if abs_coeff > threshold
                dst_coeffs[i] = coeff
            elseif n_stoch == 0
                # no stochastic slots left, so every sub-threshold term is dropped
                dst_coeffs[i] = zero(eltype(src_coeffs))
            else
                # c_start is technically stoch_cum_probs[i-1],
                # but we thread, so need to recompute from coeff
                c_end = stoch_cum_probs[i]
                c_start = c_end - abs_coeff

                lower_idx = floor((c_start - offset) / step)
                upper_idx = floor((c_end - offset) / step)
                n_copies = upper_idx - lower_idx

                dst_coeffs[i] = n_copies * step * sign(coeff)
            end
        end
    end

    swapsums!(prop_cache)
    # active size does not need to be changed.

    # dst_terms[i] = src_terms[i] for every i (only coefficients change)
    # old sorted prefix is still the new prefix
    setsortedprefix!(mainsum(prop_cache), sortedprefix(auxsum(prop_cache)))

    # now filter out the exactly 0.0
    truncate!(prop_cache; min_abs_coeff=eps(), thread)


    return prop_cache
end
