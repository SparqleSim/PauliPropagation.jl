using StatsBase

"""
    estimatemse(circ, pstr::PauliString, n_mcsamples::Integer, thetas=π; stateoverlapfunc=overlapwithzero, heisenberg=true, customtruncfunc=nothing, max_batch_size=1_000_000)

Function to estimate the mean square error of a truncated circuit simulation using Monte Carlo sampling.
Returns the mean squared error of the truncated Pauli propagation simulation averaged over the `thetas`∈ [theta, theta] of the angle `theta` of each `PauliRotation`.
Currently, the function only supports circuits with `PauliRotation` and `CliffordGate` gates.

The length the `thetas` vector should be equal to the number of parametrized gates in the circuit.
Alternatively, `thetas` can be a single real number applicable for all parametrized gates.
The default `thetas=π` or any other non-array values assume that the circuit consists only of `PauliRotation` -`CliffordGate`.
For `PauliRotation`, the value should be the integration range of the parameters around zero.

An initial state overlap function `stateoverlapfunc` can be provided to calculate the overlap of the backpropagated Pauli strings with the initial state.
Importantly, the `kwargs` can be used to set the truncation parameters of the Pauli propagation. Currently supported are `max_weight`, `max_freq`, and `max_sins`.
Note that `min_abs_coeff` is not supported here, as we consider errors integrated over the angles. `max_freq` effectively truncates small coefficients below (1/2)^`max_freq` on average over `thetas ∈ [-π, π]`.
A custom truncation function can be passed as `customtruncfunc` with the signature `customtruncfunc(pstr::PauliStringType, coefficient)::Bool`.

`n_mcsamples` may be too large to hold as a single `error_array` in memory. `max_batch_size` bounds the size of that array,
splitting the `n_mcsamples` samples into batches of at most `max_batch_size` and averaging the batched results
(the final, possibly smaller, batch is weighted accordingly).
"""
function estimatemse(circ, pstr::PauliString, n_mcsamples::Integer, thetas=π; max_batch_size=1_000_000, kwargs...)
    # this function is only valid for ParametrizedGates and non-splitting non-parametrized gates (here only CliffordGates).
    # this will not not error for non-parametrized splitting gates, e.g. T-gates.
    # TODO: Enable this for general parametrized gates. At least for PauliNoise.
    if !all(g -> isa(g, PauliRotation) || isa(g, CliffordGate), circ)
        throw("`circ` must contain only `PauliRotation`s and `CliffordGate`s.")
    end

    error_array = zeros(min(max_batch_size, n_mcsamples))

    total_error = 0.0
    offset = 0
    while offset < n_mcsamples
        chunk_len = min(max_batch_size, n_mcsamples - offset)
        chunk = @view error_array[1:chunk_len]

        estimatemse!(circ, pstr, chunk, thetas; kwargs...)
        total_error += sum(chunk)

        offset += chunk_len
    end

    return total_error / n_mcsamples
end

"""
    estimatemse!(
    circ, pstr::PauliString, error_array::AbstractVector, thetas;
    stateoverlapfunc=overlapwithzero, max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, heisenberg=true
    )

In-place version of `estimatemse`. This function takes an array `error_array` of length `n_mcsamples` as an argument and modifies it in-place,
building a single `VectorPauliSum` of that length internally. Use `estimatemse`'s `max_batch_size` to bound how large `error_array` gets.
A custom truncation function can be passed as `customtruncfunc` with the signature `customtruncfunc(pstr::PauliStringType, coefficient)::Bool`.
"""
function estimatemse!(
    circ, pstr::PauliString, error_array::AbstractVector, thetas;
    stateoverlapfunc=overlapwithzero,
    max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, heisenberg=true,
)

    # if max_freq or max_sins is not Inf, then the Pauli string must be a PathProperties type
    # check whether the coefficient of the Pauli string is properly wrapped and convert it if not
    if (max_freq != Inf || max_sins != Inf) && !(pstr.coeff isa PathProperties)
        pstr = wrapcoefficients(pstr, PauliFreqTracker)
    end
    unit_coeff = pstr.coeff isa PathProperties ? typeof(pstr.coeff)(1.0) : 1.0
    
    coeff0_sq = tonumber(pstr.coeff)^2
    
    # the same theta (radius) can be used everywhere. Padd to a vector.
    nparams = countparameters(circ)
    thetas = isa(thetas, AbstractVector) ? thetas : fill(thetas, nparams)

    # reverses the circuit for the common Heisenberg case
    circ, thetas = _preparecircuit(circ, thetas, heisenberg)

    # convert each split probability into a pseudo-angle so that mcapplytoall!(...; squared=true)
    # branches with exactly the desired probability while leaving coefficients numerically invariant
    pseudo_thetas = _pseudothetas(circ, thetas)

    n_mcsamples = length(error_array)

    psum = VectorPauliSum(pstr.nqubits, fill(pstr.term, n_mcsamples), fill(unit_coeff, n_mcsamples))
    
    # execute the path sampling
    PropagationBase._propagate!(
        mcapplytoalltruncateinplace!, circ, psum, pseudo_thetas; 
        squared=true, max_weight=max_weight, max_freq=max_freq, 
        max_sins=max_sins, customtruncfunc=customtruncfunc
    )

    terms, coeffs = paulis(psum), coefficients(psum)

    AK.foreachindex(error_array) do i
        term, coeff = terms[i], coeffs[i]
        # `coeff` is exactly the untouched unit coefficient (1.0) for a path that survived
        # the whole circuit and exactly 0 for one that got zeroed out by truncation somewhere
        # along the way, so `1 - tonumber(coeff)` is the truncated/not-truncated indicator.
        error_array[i] = coeff0_sq * (1 - tonumber(coeff)) * stateoverlapfunc(term)^2
    end

    return mean(error_array)
end

function mcapplytoalltruncateinplace!(gate, psum::VectorPauliSum, args...; squared=true, max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing)
    mcapplytoall!(gate, psum, args...; squared=squared)

    terms, coeffs = paulis(psum), coefficients(psum)

    function istruncated(pstr, coeff)
        is_truncated = false
        if truncateweight(pstr, max_weight)
            is_truncated = true
        elseif truncatefrequency(coeff, max_freq)
            is_truncated = true
        elseif truncatesins(coeff, max_sins)
            is_truncated = true
        elseif !isnothing(customtruncfunc) && customtruncfunc(pstr, coeff)
            is_truncated = true
        end
        return is_truncated
    end

    AK.foreachindex(terms) do i
        term, coeff = terms[i], coeffs[i]
        if istruncated(term, coeff)
            coeffs[i] = zero(eltype(coeffs))
        end
    end

    return psum
end

## Utilities for `estimatemse()`

# Convert the per-gate split probabilities (integrated over the θ-range in `thetas`) into pseudo-angles
# for which sin(θ_eff)^2 exactly equals the desired split probability. Feeding these into
# `mcapplytoall!(gate::PauliRotation, psum, θ_eff; squared=true)` reproduces the exact branch probability
# and leaves coefficients numerically untouched, matching the self-normalizing importance-sampling scheme.
function _pseudothetas(circ, thetas)
    nparams = countparameters(circ)

    if length(thetas) != nparams
        throw("Vector `thetas` of length $(length(thetas)) must have same length the number of parametrized gates $(nparams) in `circ`.")
    end

    pseudo_thetas = similar(thetas, Float64)
    theta_idx = 1
    for gate in circ
        if isa(gate, ParametrizedGate)
            split_prob = _calculatesplitprobabilities(gate, thetas[theta_idx])
            pseudo_thetas[theta_idx] = asin(sqrt(split_prob))
            theta_idx += 1
        end
    end
    return pseudo_thetas
end

# Splitting probability of a `PauliRotation`, averaged over `theta` uniformly drawn from `[-r, r]`.
_calculatesplitprobabilities(gate::PauliRotation, r::Number) = 0.5 * (1 - sin(2r) / (2r))
