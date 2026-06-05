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
function estimatemse(circ, pstr::PauliString, n_mcsamples::Integer, thetas=π; stateoverlapfunc=overlapwithzero, circuit_is_reversed=false, kwargs...)
    # this function is only valid for ParametrizedGates and non-splitting non-parametrized gates (here only CliffordGates).
    # this will not not error for non-parametrized splitting gates, e.g. T-gates. 
    # TODO: Enable this for general parametrized gates. At least for PauliNoise.
    if !all(g -> isa(g, PauliRotation) || isa(g, CliffordGate), circ)
        throw("`circ` must contain only `PauliRotation`s and `CliffordGate`s.")
    end

    split_probabilities = _calculatesplitprobabilities(circ, thetas)

    error_array = zeros(n_mcsamples)

    return estimatemse!(circ, pstr, error_array, thetas, split_probabilities; stateoverlapfunc, circuit_is_reversed, kwargs...)

end

"""
    estimatemse!(
    circ, pstr::PauliString, error_array::AbstractVector, thetas, split_probabilities; 
    stateoverlapfunc=overlapwithzero, circuit_is_reversed=false, max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing
    )

In-place version of `estimatemse`. This function takes an array `error_array` of length `n_mcsamples` as an argument and modifies it in-place. 
It further assumes that the `thetas` and `split_probabilities` are already correctly calculated and provided as arguments. 
In general they will be vectors, but they can also be real numbers.
A custom truncation function can be passed as `customtruncfunc` with the signature `customtruncfunc(pstr::PauliStringType, coefficient)::Bool`.
"""
function estimatemse!(
    circ, pstr::PauliString, error_array::AbstractVector, thetas, split_probabilities;
    stateoverlapfunc=overlapwithzero, circuit_is_reversed=false, max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing
)
    # This function takes an error_array as an argument and modifies it in-place.

    # length(thetas) should be equal to the number of parametrized gates in the circuit
    if isa(thetas, AbstractVector)
        if length(thetas) != countparameters(circ)
            throw("Vector `thetas` of length $(length(thetas)) must have same length the number of parametrized gates $(countparameters(circ)) in `circ`.")
        end
    end
    # length(split_probabilities) should be equal to the number of total gates in the circuit
    if isa(split_probabilities, AbstractArray)
        if length(split_probabilities) != length(circ)
            throw("Vector `split_probabilities` of length $(length(split_probabilities)) must have same length as `circ` of length $(length(circ)).")
        end
    end

    # if max_freq or max_sins is not Inf, then the Pauli string must be a PathProperties type
    # check whether the coefficient of the Pauli string is properly wrapped and convert it if not
    if max_freq != Inf || max_sins != Inf
        if !(pstr.coeff isa PathProperties)
            pstr = wrapcoefficients(pstr, PauliFreqTracker)
        end
    end

    # reverse the circuit once 
    if !circuit_is_reversed
        circ = reverse(circ)
    end

    # turn the PauliRotation gates into MaskedPauliRotation gates
    circ = _tomaskedpaulirotation(circ, pstr.nqubits)
    circ = _checkgatetypesandconcretize(circ, pstr.nqubits)

    n_mcsamples = length(error_array)
    @threads for ii in 1:n_mcsamples
        final_pstr, is_truncated = montecarlopropagation(circ, pstr, thetas, split_probabilities; max_weight, max_freq, max_sins, customtruncfunc)

        # multiply the coefficient of the backpropagated Pauli with the overlap with the initial state
        # and then multply with `is_truncated` to get the final error.
        # if truncated, then the error is coeff * initial_state_func(final_pstr), if not truncated, then the error is 0
        error_array[ii] = tonumber(final_pstr.coeff)^2 * stateoverlapfunc(final_pstr.term)^2 * is_truncated
    end

    return mean(error_array)

end

# TODO: provide an easy way to just Monte Carlo sample paths. This will happen in a refactor.

# Perform a single Monte Carlo propagation of a Pauli string through an already reversed circuit. Returns the final Pauli string and a boolean indicating whether the path was truncated.
# It further assumes that the `thetas` and `split_probabilities` are already correctly calculated and provided as arguments. 
# In general they will be vectors, but they can also be real numbers.
function montecarlopropagation(circ, pstr::PauliString, thetas, split_probabilities; max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing)

    param_idx = length(thetas)
    prob_idx = length(split_probabilities)

    is_truncated = false

    for gate in circ
        # apply the gate to the Pauli string
        # if the gate splits, the Pauli string is split with a probability 1 - split_prob
        pstr = mcapply(gate, pstr, _getelmt(thetas, param_idx), _getelmt(split_probabilities, prob_idx))

        # check if one would truncate the Pauli string
        is_truncated = _checktruncation(pstr, max_weight, max_freq, max_sins; customtruncfunc)

        # decrement the parameter index if gate is parametrized
        if isa(gate, ParametrizedGate) && param_idx > 0
            param_idx -= 1
        end
        # always decrement the probability index
        # will only be zero at the end of the circuit
        prob_idx -= 1
    end
    return pstr, is_truncated
end


function _checktruncation(pstr::PauliString, max_weight, max_freq, max_sins; customtruncfunc=nothing)
    # check truncations
    if truncateweight(pstr.term, max_weight)
        is_truncated = true
    elseif truncatefrequency(pstr.coeff, max_freq)
        is_truncated = true
    elseif truncatesins(pstr.coeff, max_sins)
        is_truncated = true
    elseif !isnothing(customtruncfunc)
        is_truncated = customtruncfunc(pstr.term, pstr.coeff)
    else
        is_truncated = false
    end
    return is_truncated
end


## Monte Carlo apply functions

# `mcapply()` function for a `CliffordGate` is just the `apply()` function because it does not split.
mcapply(gate::CliffordGate, pstr::PauliString, theta, split_probability) = PauliString(pstr.nqubits, only(apply(gate, pstr.term, pstr.coeff, clifford_map[gate.symbol]))...)


# MC apply function for a `MaskedPauliRotation`.
# This will error if a `PauliRotation` is not converted to a `MaskedPauliRotation` before calling this function.
# If the gate commutes with the pauli string, the pauli string is left unchanged. 
# Else the pauli string is split off with a probability 1 - `split_prob`.
function mcapply(gate::MaskedPauliRotation, pstr::PauliString, theta, split_prob)

    if commutes(gate, pstr.term)
        return pstr
    end
    coeff = pstr.coeff

    # if the gate does not commute with the pauli string, remain with probability `split_prob` and split off with probability 1 - `split_prob`.
    if rand() < split_prob
        # branch into the new Pauli string, ifnore ignore the sign
        new_term, sign = paulirotationproduct(gate, pstr.term)
        # for PathProperties: increment sin and frequency count
        coeff = _incrementsinandfreq(coeff)
    else
        new_term = pstr.term
        # Pauli doesn't get changed
        # for PathProperties: increment cos and frequency count
        coeff = _incrementcosandfreq(coeff)
    end

    # the sign has abs(sign) = 1, so updating the coefficient with abs(sign) doesn't do anything
    return PauliString(pstr.nqubits, new_term, coeff)
end

function mcapply(gate::GT, args...; kwargs...) where {GT<:Gate}
    # TODO
    throw("Numerical certificates for gates of type $(GT) are not yet supported.")
end

## These functions exist for PathProperties that track nsins, ncos and/or freq
function _incrementcosandfreq(pth::PProp) where {PProp<:PathProperties}
    fields = fieldnames(PProp)
    @inline function updateval(val, field)
        if field == :ncos
            return val + 1
        elseif field == :freq
            return val + 1
        else
            return val
        end
    end
    return PProp((updateval(getfield(pth, field), field) for field in fields)...)
end

function _incrementsinandfreq(pth::PProp) where {PProp<:PathProperties}
    fields = fieldnames(PProp)
    @inline function updateval(val, field)
        if field == :nsins
            return val + 1
        elseif field == :freq
            return val + 1
        else
            return val
        end
    end
    return PProp((updateval(getfield(pth, field), field) for field in fields)...)
end

_incrementcosandfreq(val::Number) = val
_incrementsinandfreq(val::Number) = val

## Utilities for `estimatemse()`

# Function that automatically calculates the vector of splitting probabilities of the gates in the circuit based on a vector of thetas.
# For Pauli gates, the theta value is interpreted as the limits of the integration [-theta, theta].
# For AmplitudeDampingNoise, the splitting probability is the damping rate.
function _calculatesplitprobabilities(circ::AbstractArray, thetas::AbstractArray)
    if length(thetas) != countparameters(circ)
        throw("Vector `thetas` must have same length the number of parametrized gates in `circ`.")
    end

    split_probabilities = zeros(length(circ))

    theta_idx = 1
    for (ii, gate) in enumerate(circ)
        if isa(gate, ParametrizedGate)
            split_probabilities[ii] = _calculatesplitprobabilities(gate, thetas[theta_idx])
            theta_idx += 1
        end
    end
    return split_probabilities
end


# Function that automatically calculates the splitting probability of the gates in the circuit based on a one number theta.
# This assumes that the circuit consists only of `PauliRotation` -`CliffordGate`.
function _calculatesplitprobabilities(circ::AbstractArray, r::Number)
    return 0.5 * (1 - sin(2 * r) / (2 * r))
end

_calculatesplitprobabilities(gate::PauliRotationUnion, theta::Number) = 0.5 * (1 - sin(2theta) / (2theta))

_calculatesplitprobabilities(gate::AmplitudeDampingNoise, theta::Number) = theta

_calculatesplitprobabilities(gate, theta::Number) = 0.0

function _checkgatetypesandconcretize(circ, nqubits)
    npaulirots = count(gate -> isa(gate, MaskedPauliRotation), circ)
    ncliffords = count(gate -> isa(gate, CliffordGate), circ)
    if npaulirots + ncliffords != length(circ)
        throw("The circuit must contain only `MaskedPauliRotation`s and `CliffordGate`s.")
    end

    # because we are working with hot loops, we need the circuit to be as concretely typed as possible
    if ncliffords == 0
        circ = convert(Vector{MaskedPauliRotation{getinttype(nqubits)}}, circ)
    elseif npaulirots == 0
        circ = convert(Vector{CliffordGate}, circ)
    else
        circ = convert(Vector{Union{MaskedPauliRotation{getinttype(nqubits)},CliffordGate}}, circ)
    end
    return circ
end

### Utilities for `montecarlopropagation()`
_getelmt(arr::AbstractArray, idx::Integer) = arr[idx]
_getelmt(num::Number, idx::Integer) = num
