using StatsBase

"""
    estimatemse(circ, pstr::PauliString, n_mcsamples::Integer, thetas=π; stateoverlapfunc=overlapwithzero, circuit_is_reversed=false, customtruncfunc=nothing)

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
"""
function estimatemse(circ, pstr::PauliString, n_mcsamples::Integer, thetas=π; kwargs...)
    # this function is only valid for ParametrizedGates and non-splitting non-parametrized gates (here only CliffordGates).
    # this will not not error for non-parametrized splitting gates, e.g. T-gates.
    # TODO: Enable this for general parametrized gates. At least for PauliNoise.
    if !all(g -> isa(g, PauliRotation) || isa(g, CliffordGate), circ)
        throw("`circ` must contain only `PauliRotation`s and `CliffordGate`s.")
    end

    error_array = zeros(n_mcsamples)

    return estimatemse!(circ, pstr, error_array, thetas; kwargs...)
end

"""
    estimatemse!(
    circ, pstr::PauliString, error_array::AbstractVector, thetas;
    stateoverlapfunc=overlapwithzero, circuit_is_reversed=false, max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, max_batch_size=length(error_array)
    )

In-place version of `estimatemse`. This function takes an array `error_array` of length `n_mcsamples` as an argument and modifies it in-place.
A custom truncation function can be passed as `customtruncfunc` with the signature `customtruncfunc(pstr::PauliStringType, coefficient)::Bool`.
`max_batch_size` bounds the size of the `VectorPauliSum` used internally for a single Monte Carlo sampling batch, trading off memory for fewer batches.
It has no effect on the statistics of the returned estimate.
"""
function estimatemse!(
    circ, pstr::PauliString, error_array::AbstractVector, thetas;
    stateoverlapfunc=overlapwithzero, circuit_is_reversed=false,
    max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing,
    max_batch_size=length(error_array)
)
    nparams = countparameters(circ)
    thetas = isa(thetas, AbstractVector) ? thetas : fill(thetas, nparams)
    if length(thetas) != nparams
        throw("Vector `thetas` of length $(length(thetas)) must have same length the number of parametrized gates $(nparams) in `circ`.")
    end

    # if max_freq or max_sins is not Inf, then the Pauli string must be a PathProperties type
    # check whether the coefficient of the Pauli string is properly wrapped and convert it if not
    if (max_freq != Inf || max_sins != Inf) && !(pstr.coeff isa PathProperties)
        pstr = wrapcoefficients(pstr, PauliFreqTracker)
    end
    unit_coeff = pstr.coeff isa PathProperties ? typeof(pstr.coeff)(1.0) : 1.0
    coeff0_sq = tonumber(pstr.coeff)^2

    # convert each split probability into a pseudo-angle so that mcapplytoall!(...; squared=true)
    # branches with exactly the desired probability while leaving coefficients numerically invariant
    pseudo_thetas = _pseudothetas(circ, thetas)
    circ_final, thetas_final = circuit_is_reversed ? (circ, pseudo_thetas) : toheisenberg(circ, pseudo_thetas)

    n_mcsamples = length(error_array)
    offset = 0
    while offset < n_mcsamples
        chunk_len = min(max_batch_size, n_mcsamples - offset)

        psum = VectorPauliSum(pstr.nqubits, fill(pstr.term, chunk_len), fill(unit_coeff, chunk_len))
        PropagationBase._propagate!(mcapplytoall!, circ_final, psum, thetas_final; squared=true)

        terms, coeffs = paulis(psum), coefficients(psum)
        is_truncated = truncateweight.(terms, max_weight) .| truncatefrequency.(coeffs, max_freq) .| truncatesins.(coeffs, max_sins)
        if !isnothing(customtruncfunc)
            is_truncated .|= customtruncfunc.(terms, coeffs)
        end

        error_array[offset+1:offset+chunk_len] .= coeff0_sq .* stateoverlapfunc.(terms) .^ 2 .* is_truncated

        offset += chunk_len
    end

    return mean(error_array)
end

## Utilities for `estimatemse()`

# Convert the per-gate split probabilities (integrated over the θ-range in `thetas`) into pseudo-angles
# for which sin(θ_eff)^2 exactly equals the desired split probability. Feeding these into
# `mcapplytoall!(gate::PauliRotation, psum, θ_eff; squared=true)` reproduces the exact branch probability
# and leaves coefficients numerically untouched, matching the self-normalizing importance-sampling scheme.
function _pseudothetas(circ, thetas)
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
