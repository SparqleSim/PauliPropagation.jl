###
##
# Resampling: reduces the number of terms in a TermSum by randomly sampling from the coefficient distribution
##
###
# TODO: Resampling currently only works for array-based TermSums
using StatsBase

## RE-SAMPLING
coeffcumsum!(coeffs) = AK.accumulate((x1, x2) -> x1 + abs(x2), coeffs; init=zero(eltype(coeffs)), neutral=zero(eltype(coeffs)))
coeffcumsum(coeffs) = coeffcumsum!(copy(coeffs))
function coeffcumsum(coeffs, power::Real)
    mapped_coeffs = AK.map(c -> abs(c)^power, coeffs)
    mapped_coeffs = coeffcumsum!(mapped_coeffs)
    return mapped_coeffs
end

"""
    resample(tsum::AbstractTermSum, target_size::Integer; resample_func=nothing, squared=false, kwargs...)

Resamples `tsum` down (close) to `target_size` terms.
Renormalizes the survivors so that the sum stays an unbiased estimator of incoming sum.
If `squared=true`, resampling is performed on the absolute square of the coefficients and
is not an unbiased estimator of the incoming sum.
"""
function resample(tsum::AbstractTermSum, target_size::Integer, resample_args...; kwargs...)
    return resample!(deepcopy(tsum), target_size, resample_args...; kwargs...)
end

"""
    resample!(tsum::AbstractTermSum, target_size::Integer; kwargs...)
    resample!(prop_cache::AbstractPropagationCache, target_size::Integer; resample_func=nothing, kwargs...)

In-place version of `resample`. See `resample` for details.
"""
function resample!(tsum::AbstractTermSum, target_size::Integer, resample_args...; resample_func=nothing, resample_kwargs...)
    prop_cache = resample!(PropagationCache(tsum), target_size, resample_args...; resample_func=resample_func, resample_kwargs...)

    # extract the original tsum
    return extractsum!(prop_cache, tsum)
end

function resample!(prop_cache::AbstractPropagationCache, target_size, resample_args...; resample_func=nothing, resample_kwargs...)
    @assert target_size > 0 "target_size must be positive"

    if target_size > activesize(prop_cache)
        throw(ArgumentError("target_size must be less than the current active size of the prop_cache."))
    end
    # resample_func is expected to take prop_cache and target_size as arguments
    # anything else needs to be wrapped into a closure

    # default to our current best technique
    if isnothing(resample_func)
        resample_func = systematic_resample!
    end

    resample_func(prop_cache, target_size, resample_args...; resample_kwargs...)

    return prop_cache
end

## This the naive resampling where one draw's a random number per sample.
function multinomial_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, kwargs...)

    dst_terms = terms(auxsum(prop_cache))
    dst_coeffs = coefficients(auxsum(prop_cache))
    src_terms = activeterms(prop_cache)
    src_coeffs = activecoeffs(prop_cache)

    _multinomial_resample!(dst_terms, dst_coeffs, src_terms, src_coeffs, target_size; squared)

    swapsums!(prop_cache)
    setactivesize!(prop_cache, target_size)

    return prop_cache
end

function _multinomial_resample!(dst_terms, dst_coeffs, terms, coeffs, target_size; squared=false)
    power = squared ? 2 : 1

    # Compute the cumulative distribution
    # TODO: make non-allocating
    cum_probs = coeffcumsum(coeffs, power)
    total_weight = cum_probs[end]

    dst_terms_view = view(dst_terms, 1:target_size)
    AK.foreachindex(dst_terms_view) do i
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
    systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false)

Classic (low-variance) systematic resampling: a single evenly-spaced "comb" of `target_size` teeth
is laid down over the cumulative coefficient distribution (with one shared random offset), and each
tooth draws whichever term interval it falls into. Always returns exactly `target_size` terms;
a term with large weight may be drawn into several destination slots.
See `systematic_resample_merged!` for a variant that instead folds repeat draws of the same term
into a single slot.
"""
function systematic_resample!(prop_cache::AbstractPropagationCache, target_size::Integer; squared::Bool=false, kwargs...)
    dst_terms = terms(auxsum(prop_cache))
    dst_coeffs = coefficients(auxsum(prop_cache))
    src_terms = activeterms(prop_cache)
    src_coeffs = activecoeffs(prop_cache)

    _systematic_resample!(dst_terms, dst_coeffs, src_terms, src_coeffs, target_size; squared)

    swapsums!(prop_cache)
    setactivesize!(prop_cache, target_size)

    return prop_cache
end

function _systematic_resample!(dst_terms, dst_coeffs, terms, coeffs, target_size; squared::Bool=false)
    power = squared ? 2 : 1

    cum_probs = coeffcumsum(coeffs, power)
    total_weight = cum_probs[end]

    step = total_weight / target_size
    # the single source of randomness shifting the comb
    offset = rand() * step

    dst_terms_view = view(dst_terms, 1:target_size)
    let step = step, offset = offset, power = power
        AK.foreachindex(dst_terms_view) do i
            u = offset + (i - 1) * step

            idx = searchsortedfirst(cum_probs, u)

            dst_terms[i] = terms[idx]
            dst_coeffs[i] = (total_weight / target_size)^(1 / power) * sign(coeffs[idx])^power
        end
    end

    return nothing
end


"""
    systematic_resample_merged!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false, calibrate=true, rtol=0.01, atol=0)

Like `systematic_resample!`, but repeat draws of the same term are folded into that term's own slot
instead of being copied into separate destination slots. The number of surviving (unique) terms is
therefore only close to, and generally at most, `target_size`; see `calibrate`/`rtol`/`atol` for tuning
how closely the comb step is chosen to hit `target_size` unique survivors.
"""
function systematic_resample_merged!(prop_cache::AbstractPropagationCache, target_size::Integer; squared::Bool=false, calibrate=true, rtol=0.01, atol=0, kwargs...)
    dst_terms = terms(auxsum(prop_cache))
    dst_coeffs = coefficients(auxsum(prop_cache))
    src_terms = activeterms(prop_cache)
    src_coeffs = activecoeffs(prop_cache)

    _systematic_resample_merged!(dst_terms, dst_coeffs, src_terms, src_coeffs, target_size; squared, calibrate, rtol, atol)

    swapsums!(prop_cache)
    # active size does not need to be changed.

    # now filter out the exactly 0.0
    truncate!(prop_cache; min_abs_coeff=eps())


    return prop_cache
end

function _systematic_resample_merged!(dst_terms, dst_coeffs, terms, coeffs, target_size; squared::Bool=false, calibrate=true, rtol=0.02, atol=1)
    power = squared ? 2 : 1

    cum_probs = coeffcumsum(coeffs, power)
    total_weight = cum_probs[end]

    # Normally: step = total_weight / target_size
    # but this will generally produce less unique terms
    # scale the step to go toward target_size many unique terms
    if calibrate
        step = _calibrate_prob_step(coeffs, total_weight, target_size; power, rtol, atol)
    else
        step = total_weight / target_size
    end

    # the source of randomness shifting the comb
    offset = rand() * step


    # We iterate over INPUT terms. Each term is processed once.
    # The "let" block is necessary because otherwise "step" get boxed (wow again)
    let step=step, offset=offset, power=power
        AK.foreachindex(terms) do i

            # Calculate the interval this term occupies in the cumulative probability distribution
            c_end = cum_probs[i]
            c_start = (i == 1) ? zero(c_end) : cum_probs[i-1]

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


function _calibrate_prob_step(coeffs, total_weight, target_size; power=1, rtol::Real=0.02, atol::Real=0)

    # this function estimates the mean number of unique samples (or something close to it)
    # when resampling we can overshoot if rtol is small
    getcurrentsum(cs, inv_s) = AK.mapreduce(c -> min(1.0, abs(c)^power * inv_s), +, cs; init=zero(eltype(cs)))

    # println("Total raw: ", total_weight)

    inv_step = target_size / total_weight
    # println("Starting with: ", inv_step)

    tolsatisfied(r) = ((1.0 - rtol) * target_size - atol) / target_size <= r <= 1.0 - eps(eltype(coeffs))
    for i in 1:10
        # println(i)
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
    detfraction_systematic_resample_merged!(prop_cache::AbstractPropagationCache, target_size::Integer, deterministic_fraction::Real=0.0; squared=false)

Generalizes `systematic_resample_merged!` by always keeping terms whose weight exceeds
`total_weight / (deterministic_fraction * target_size)`, and filling the remaining slots by
systematic comb resampling (folded into each term's own slot, as in `systematic_resample_merged!`)
over what is left. `deterministic_fraction=0` recovers pure stochastic resampling (nothing is kept
deterministically); `deterministic_fraction=1` recovers `semideterministic_systematic_resample_merged!`.
`squared=true` is not supported since a deterministically-kept term does not preserve the squared
2-norm the way a resampled one does.
"""
function detfraction_systematic_resample_merged!(prop_cache::AbstractPropagationCache, target_size::Integer, deterministic_fraction::Real=0.0; squared::Bool=false, kwargs...)
    if squared
        throw(ArgumentError("squared resampling is not supported for detfraction resampling."))
    end
    @assert 0 <= deterministic_fraction <= 1 "deterministic_fraction must be between 0 and 1"

    dst_terms = terms(auxsum(prop_cache))
    dst_coeffs = coefficients(auxsum(prop_cache))
    src_terms = activeterms(prop_cache)
    src_coeffs = activecoeffs(prop_cache)

    @assert 0 < target_size <= activesize(prop_cache) "target_size must be between 1 and activesize(prop_cache)"

    # a term is always kept deterministically if its weight exceeds this threshold, otherwise it is resampled
    total_weight = AK.mapreduce(abs, +, src_coeffs; init=zero(eltype(src_coeffs)))
    threshold = total_weight / (deterministic_fraction * target_size)

    # how many terms are taken deterministically
    n_det = AK.mapreduce(c -> abs(c) > threshold, +, src_coeffs; init=0)

    # How many slots are left for the stochastic part
    n_stoch = target_size - n_det

    # the weights of terms that are resampled. Deterministic coeffs are set to zero.
    stoch_weights = AK.map(c -> abs(c) > threshold ? zero(eltype(src_coeffs)) : abs(c), src_coeffs)
    # get the probability distribution for the stochastic part
    stoch_cum_probs = AK.accumulate!(+, stoch_weights; init=0.0)
    total_stoch_weight = stoch_cum_probs[end]

    # step and offset of the comb for systematic resampling
    step = n_stoch > 0 ? total_stoch_weight / n_stoch : 0.0
    offset = rand() * step

    AK.foreachindex(src_terms) do i
        coeff = src_coeffs[i]
        abs_coeff = abs(coeff)

        # the term is always copied
        dst_terms[i] = src_terms[i]

        if abs_coeff > threshold
            dst_coeffs[i] = coeff
        else
            c_end = stoch_cum_probs[i]
            c_start = (i == 1) ? zero(c_end) : stoch_cum_probs[i-1]

            lower_idx = floor((c_start - offset) / step)
            upper_idx = floor((c_end - offset) / step)
            n_copies = upper_idx - lower_idx

            dst_coeffs[i] = n_copies * step * sign(coeff)
        end
    end

    swapsums!(prop_cache)
    # active size does not need to be changed.

    # now filter out the exactly 0.0
    truncate!(prop_cache; min_abs_coeff=eps())


    return prop_cache
end


"""
    semideterministic_systematic_resample_merged!(prop_cache::AbstractPropagationCache, target_size::Integer; squared=false)

Terms whose weight exceeds the average per-slot weight `total_weight / target_size` are always kept;
the remaining slots are filled by systematic comb resampling over what is left, folded into each
term's own slot as in `systematic_resample_merged!`. Equivalent to
`detfraction_systematic_resample_merged!(prop_cache, target_size, 1.0; kwargs...)`.
"""
function semideterministic_systematic_resample_merged!(prop_cache::AbstractPropagationCache, target_size::Integer; kwargs...)
    return detfraction_systematic_resample_merged!(prop_cache, target_size, 1.0; kwargs...)
end
