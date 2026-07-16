# TODO: simplify by sticking to a couple of subroutines
# TODO: less code duplication and more extensible


## RE-SAMPLING
coeffcumsum!(coeffs) = AK.accumulate((x1, x2) -> x1 + abs(x2), coeffs; init=zero(eltype(coeffs)), neutral=zero(eltype(coeffs)))
coeffcumsum(coeffs) = coeffcumsum!(copy(coeffs))
function coeffcumsum(coeffs, power::Real)
    mapped_coeffs = AK.map(c -> abs(c)^power, coeffs)
    mapped_coeffs = coeffcumsum!(mapped_coeffs)
    return mapped_coeffs
end

function resample(psum::VectorPauliSum, target_size::Integer, resample_args...; resample_func=nothing, resample_kwargs...)
    return resample!(deepcopy(psum), target_size, resample_args...; resample_func=resample_func, resample_kwargs...)
end

function resample!(psum::VectorPauliSum, target_size::Integer, resample_args...; resample_func=nothing, resample_kwargs...)
    prop_cache = resample!(PropagationCache(psum), target_size, resample_args...; resample_func=resample_func, resample_kwargs...)

    # extract the original psum
    return extractsum!(prop_cache, psum)
end

function resample!(prop_cache::VectorPauliPropagationCache, target_size, resample_args...; resample_func=nothing, resample_kwargs...)
    @assert target_size > 0 "target_size must be positive"

    if target_size > activesize(prop_cache)
        # return prop_cache
        throw(ArgumentError("target_size must be less than the current active size of the prop_cache."))
    end
    # resample_func is expected to take prop_cache and target_size as arguments
    # anything else needs to be wrapped into a closure

    # default to our current best technique
    if isnothing(resample_func)
        # resample_func = (prop_cache, target_size) -> semideterministic_systematic_resample_merged!(prop_cache, target_size, 0.4)
        resample_func = systematic_resample_merged! # semideterministic_systematic_resample_merged! # 
    end

    resample_func(prop_cache, target_size, resample_args...; resample_kwargs...)

    return prop_cache
end

## This the naive resampling where one draw's a random number per sample.
function multinomial_resample!(prop_cache::VectorPauliPropagationCache, target_size::Integer; power=1)
    @assert power ∈ (1, 2) "currently only power=1 and power=2 supported for multinomial resampling."

    dst_terms = paulis(auxsum(prop_cache))
    dst_coeffs = coefficients(auxsum(prop_cache))
    terms = activeterms(prop_cache)
    coeffs = activecoeffs(prop_cache)

    _multinomial_resample!(dst_terms, dst_coeffs, terms, coeffs, target_size, power)

    swapsums!(prop_cache)
    setactivesize!(prop_cache, target_size)

    return prop_cache
end

function _multinomial_resample!(dst_terms, dst_coeffs, terms, coeffs, target_size, power)
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
        dst_coeffs[i] = total_weight * sign(coeffs[idx])^power / target_size
    end

    return nothing
end

function systematic_resample!(prop_cache::VectorPauliPropagationCache, target_size::Integer; power=1)
    dst_terms = paulis(auxsum(prop_cache))
    dst_coeffs = coefficients(auxsum(prop_cache))
    terms = activeterms(prop_cache)
    coeffs = activecoeffs(prop_cache)

    _systematic_resample!(dst_terms, dst_coeffs, terms, coeffs, target_size, power)

    swapsums!(prop_cache)
    setactivesize!(prop_cache, target_size)

    return prop_cache
end

function _systematic_resample!(dst_terms, dst_coeffs, terms, coeffs, target_size, power)
    cum_probs = coeffcumsum(coeffs, power)
    total_weight = cum_probs[end]

    # 'step' is the distance between the teeth of the comb
    # it is also the mass ratio of each sampled term
    step = total_weight / target_size
    offset = rand() * step

    dst_terms_view = view(dst_terms, 1:target_size)
    AK.foreachindex(dst_terms_view) do i

        # Calculate the fixed position for this specific walker
        pointer = offset + (i - 1) * step

        # we don't parallelize this because we are already threading over the outer loop
        parent_idx = searchsortedfirst(cum_probs, pointer)

        # avoid numerical overflow
        parent_idx = min(parent_idx, length(terms))

        dst_terms[i] = terms[parent_idx]
        dst_coeffs[i] = step * sign(coeffs[parent_idx])^power
    end

    return nothing
end

function systematic_resample_merged!(prop_cache::VectorPauliPropagationCache, target_size::Integer; power::Integer=1, calibrate=true, rtol=0.01, atol=0)
    dst_terms = paulis(auxsum(prop_cache))
    dst_coeffs = coefficients(auxsum(prop_cache))
    terms = activeterms(prop_cache)
    coeffs = activecoeffs(prop_cache)

    _systematic_resample_merged!(dst_terms, dst_coeffs, terms, coeffs, target_size; power, calibrate, rtol, atol)

    swapsums!(prop_cache)
    # active size does not need to be changed.

    # now filter out the exactly 0.0
    truncate!(prop_cache; min_abs_coeff=eps())


    return prop_cache
end

function _systematic_resample_merged!(dst_terms, dst_coeffs, terms, coeffs, target_size; power=1,calibrate=true, rtol=0.02, atol=1)
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
            dst_coeffs[i] = n_copies * step * sign(coeffs[i])^power
        end
    end

    return nothing
end


function _calibrate_prob_step(coeffs, total_weight, target_size; power=1, rtol::Real=0.02, atol::Real=0)

    # this function estimates the mean number of unique samples (or something close to it)
    # when resampling we can overshoot if rtol is small
    getcurrentsum(coeffs, inv_s) = AK.mapreduce(c -> min(1.0, abs(c)^power * inv_s), +, coeffs; init=zero(eltype(coeffs)))

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


function semideterministic_systematic_resample_merged!(prop_cache::VectorPauliPropagationCache, target_size::Integer)
    dst_terms = paulis(auxsum(prop_cache))
    dst_coeffs = coefficients(auxsum(prop_cache))
    terms = activeterms(prop_cache)
    coeffs = activecoeffs(prop_cache)

    @assert 0 < target_size <= activesize(prop_cache) "target_size must be between 1 and activesize(prop_cache)"

    # compute the average weight capacity of the resampled terms
    # if a term is above this threshold, it is always kept, otherwise it is resampled
    total_weight = AK.mapreduce(abs, +, coeffs; init=zero(eltype(coeffs)))
    threshold = total_weight / target_size

    # how many terms are taken deterministically
    n_det = AK.mapreduce(c -> abs(c) > threshold, +, coeffs; init=0)

    # How many slots are left for the stochatic part
    # this will never be zero unless the distribtion is completely flat and target_size==activesize
    n_stoch = target_size - n_det

    # the weights of terms that are resampled. Deterministic coeffs are set to zero.
    stoch_weights = AK.map(c -> abs(c) > threshold ? zero(eltype(coeffs)) : abs(c), coeffs)
    # get the probability distribution for the stochastic part
    stoch_cum_probs = AK.accumulate!(+, stoch_weights; init=0.0)
    total_stoch_weight = stoch_cum_probs[end]

    # step and offset of the comb for systematic resampling
    step = n_stoch > 0 ? total_stoch_weight / n_stoch : 0.0
    offset = rand() * step

    # We iterate over INPUT terms
    AK.foreachindex(terms) do i
        coeff = coeffs[i]
        abs_coeff = abs(coeff)

        # the term is always copied
        dst_terms[i] = terms[i]

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




function detfraction_systematic_resample_merged!(prop_cache::VectorPauliPropagationCache, target_size::Integer, deterministic_fraction)
    dst_terms = paulis(auxsum(prop_cache))
    dst_coeffs = coefficients(auxsum(prop_cache))
    terms = activeterms(prop_cache)
    coeffs = activecoeffs(prop_cache)

    # "deterministic_fraction" is ∈ [0, 1]
    # it implies that "deterministic_fraction" of the new buffer is filled deterministically
    # NOT that that fraction of the old buffer is moved over deterministically
    # first adapt the percentile so that it reflects this
    n_det_desired = floor(Int, deterministic_fraction * target_size)
    det_ratio_input = clamp(n_det_desired / activesize(prop_cache), 0.0, 1.0)
    threshold = _estimate_deterministic_threshold(coeffs, 100 * (1.0 - det_ratio_input))  # adapt for percentile() call

    # this extra pass could be avoided if we take "desired_det_count" directly
    # and accept a potential small error that could make us go beyond "target_size" by a bit
    n_det_actual = AK.mapreduce(c -> abs(c) > threshold, +, coeffs; init=0)

    # How many slots are left for the stochatic part
    # set at least to remain unbiased
    target_stoch_slots = max(1, target_size - n_det_actual)

    # the weights of terms that are resampled. Deterministic coeffs are set to zero.
    stoch_weights = AK.map(c -> abs(c) > threshold ? zero(eltype(coeffs)) : abs(c), coeffs)
    # get the probability distribution for the stochastic part
    stoch_cum_probs = AK.accumulate!(+, stoch_weights; init=0.0)
    total_stoch_weight = stoch_cum_probs[end]

    # check if we have stochastic slots and valid weight
    valid_stoch = (target_stoch_slots > 0) && (total_stoch_weight > eps(total_stoch_weight))

    step = valid_stoch ? total_stoch_weight / target_stoch_slots : 0.0
    offset = valid_stoch ? rand() * step : 0.0

    # We iterate over INPUT terms. Each term is processed once.
    AK.foreachindex(terms) do i
        coeff = coeffs[i]
        abs_coeff = abs(coeff)

        # the term is always copied
        dst_terms[i] = terms[i]

        if abs_coeff > threshold
            dst_coeffs[i] = coeff
        elseif valid_stoch
            c_end = stoch_cum_probs[i]
            c_start = (i == 1) ? zero(c_end) : stoch_cum_probs[i-1]

            lower_idx = floor((c_start - offset) / step)
            upper_idx = floor((c_end - offset) / step)
            n_copies = upper_idx - lower_idx

            dst_coeffs[i] = n_copies * step * sign(coeff)
        else
            dst_coeffs[i] = zero(coeff)
        end
    end

    swapsums!(prop_cache)
    # active size does not need to be changed.

    # now filter out the exactly 0.0
    truncate!(prop_cache; min_abs_coeff=eps())


    return prop_cache
end

using Statistics

function _estimate_deterministic_threshold(coeffs, target_percentile; sample_size=2048)
    N = length(coeffs)

    random_selection = view(coeffs, rand(1:N, sample_size))

    # percentile function expects values between 0 to 100, not 0 to 1
    return percentile(abs.(random_selection), target_percentile)
end


