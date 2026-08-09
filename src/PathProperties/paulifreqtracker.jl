### paulifreqtracker.jl
##
# PauliFreqTracker type and methods.
# It records the behavior at PauliRotation gates, i.e., the number of times it received a sin or cos factor, and the total number of branchings/splits.
# These path properties can be used for truncations.
# By default, we support `max_freq` and `max_nsins` truncations if the coefficients are of type `PauliFreqTracker`.
##
###

"""
    PauliFreqTracker(coeff::Number, nsins::Int, ncos::Int, freq::Int)

Wrapper type for numerical coefficients in Pauli propagation that records 
the number of sin and cos factors applied via a `PauliRotation` gate, and the so-called frequency, which is their sum.
It appears redundant but these three properties need to be tracked separately because of how merging affects them.
"""
struct PauliFreqTracker{T<:Number} <: PathProperties
    coeff::T
    nsins::Int
    ncos::Int
    freq::Int
end

"""
    PauliFreqTracker(coeff::Number)

Constructor for `PauliFreqTracker` from only a coefficient.
Initializes `nsins`, `ncos`, and `freq` to zero.
"""
PauliFreqTracker(coeff::Number) = PauliFreqTracker(float(coeff), 0, 0, 0)
PauliFreqTracker{T}(coeff::Number) where {T<:Number} = PauliFreqTracker{T}(convert(T, coeff), 0, 0, 0)

PropagationBase.numcoefftype(::Type{PauliFreqTracker{T}}) where {T<:Number} = T

### Specializations for PauliRotations that incremet the nsins, ncos, and freq

# Overload of `applytoall!` for `PauliRotation` gates acting onto Pauli sums with `PathProperties` coefficients. 
function PropagationBase.applytoall!(gate::PauliRotation, prop_cache::PauliPropagationCache{PauliSum{TT,PProp}}, theta; kwargs...) where {TT,PProp<:PathProperties}

    psum = mainsum(prop_cache)
    aux_psum = auxsum(prop_cache)

    # compute the bitmask of the gate generator for faster operations
    gate_mask = symboltoint(paulitype(prop_cache), gate.symbols, gate.qinds)

    # loop over all Pauli strings and their coefficients in the Pauli sum
    for (pstr, coeff) in psum

        if commutes(gate_mask, pstr)
            # if the gate commutes with the pauli string, do nothing
            continue
        end

        # else we know the gate will split th Pauli string into two
        pstr, coeff1, new_pstr, coeff2 = splitapply(gate_mask, pstr, coeff, theta; kwargs...)

        # set the coefficient of the original Pauli string
        set!(psum, pstr, coeff1)

        # set the coefficient of the new Pauli string in the aux_psum
        # we can set the coefficient because PauliRotations create non-overlapping new Pauli strings
        set!(aux_psum, new_pstr, coeff2)
    end

    return prop_cache
end

## Specializations for PauliRotations that increment the nsins, ncos, and freq
# can be used by all PathProperties types that have the necessary fields `ncos`, `nsins`, and `freq`
function splitapply(gate_mask::Integer, pstr::PauliStringType, coeff::PProp, theta; kwargs...) where {PProp<:PathProperties}
    # increments ncos and freq field if applicable
    coeff1 = _applycos(coeff, theta; kwargs...)
    new_pstr, sign = paulirotationproduct(gate_mask, pstr)
    # increments nsins and freq field if applicable
    coeff2 = _applysin(coeff, theta, sign; kwargs...)

    return pstr, coeff1, new_pstr, coeff2
end

# These also work for other PathProperties types that have a `coeff` field defined
# Multiply sin(theta) * sign to the `coeff` field of a `PathProperties` object.
# Increments the `nsins` and `freq` fields by 1 if applicable.
function _applysin(pth::PProp, theta, sign=1; kwargs...) where {PProp<:PathProperties}
    fields = fieldnames(PProp)

    if :coeff ∉ fields
        throw(
            "The $(PProp) object does not have a field `coeff` to use the `_applysin` operation. " *
            "Consider defining _applysin(pth::$(PProp), theta, sign; kwargs...)"
        )
    end

    function updateval(val, field)
        if field == :coeff
            # apply sin to the `coeff` field
            return val * sin(theta) * sign
        elseif field == :nsins
            # increment the `nsins` field
            return val + 1
        elseif field == :freq
            # increment the `freq` field
            return val + 1
        else
            return val
        end
    end

    return PProp((updateval(getfield(pth, field), field) for field in fields)...)
end


# Multiply cos(theta) * sign to the `coeff` field of a `PathProperties` object.
# Increments the `ncos` and `freq` fields by 1 if applicable.
function _applycos(pth::PProp, theta, sign=1; kwargs...) where {PProp<:PathProperties}
    fields = fieldnames(PProp)

    if :coeff ∉ fields
        throw(
            "The $(PProp) object does not have a field `coeff` to use the `_applysin` operation. " *
            "Consider defining _applycos(pth::$(PProp), theta, sign; kwargs...)"
        )
    end

    function updateval(val, field)
        if field == :coeff
            # apply cos to the `coeff` field
            return val * cos(theta) * sign
        elseif field == :ncos
            # increment the `ncos` field
            return val + 1
        elseif field == :freq
            # increment the `freq` field
            return val + 1
        else
            return val
        end
    end

    return PProp((updateval(getfield(pth, field), field) for field in fields)...)
end


## Specialization for Monte Carlo path sampling with PathProperties coefficients
# can be used by all PathProperties types that have the necessary fields `ncos`, `nsins`, and `freq`

# Increment the `ncos` and `freq` fields of a `PathProperties` object by 1, leaving `coeff` untouched.
function _incrementcosandfreq(pth::PProp) where {PProp<:PathProperties}
    fields = fieldnames(PProp)

    function updateval(val, field)
        if field == :ncos || field == :freq
            return val + 1
        else
            return val
        end
    end

    return PProp((updateval(getfield(pth, field), field) for field in fields)...)
end

# Increment the `nsins` and `freq` fields of a `PathProperties` object by 1, leaving `coeff` untouched.
function _incrementsinandfreq(pth::PProp) where {PProp<:PathProperties}
    fields = fieldnames(PProp)

    function updateval(val, field)
        if field == :nsins || field == :freq
            return val + 1
        else
            return val
        end
    end

    return PProp((updateval(getfield(pth, field), field) for field in fields)...)
end

# Overload of `mcapplytoall!` for `CliffordGate`s acting onto `VectorPauliSum`s with `PathProperties` coefficients.
# The generic method in `vectormontecarlo.jl` isolates the raw ±1 sign via `apply(gate, term, one(coeff), ...)`,
# but `one(coeff)` for a `PathProperties` coefficient returns another `PathProperties` object, not a plain
# number, so the subsequent `sign^power` fails. Isolating the sign via `one(tonumber(coeff))` instead sidesteps
# this: `apply` then returns a plain-number sign, which the generic `*` operator multiplies onto `coeff`,
# leaving `nsins`/`ncos`/`freq` untouched, exactly as for Clifford gates in the deterministic pipeline.
function PropagationBase.mcapplytoall!(gate::CliffordGate, psum::VectorPauliSum{TV,CV}; squared=false, kwargs...) where {TV,CT<:PathProperties,CV<:AbstractVector{CT}}
    lookup_map = clifford_map[gate.symbol]

    power = squared ? 2 : 1

    term_vec = paulis(psum)
    coeff_vec = coefficients(psum)
    AK.foreachindex(term_vec) do ii
        term = term_vec[ii]
        coeff = coeff_vec[ii]

        new_term, sign = only(apply(gate, term, one(tonumber(coeff)), lookup_map))

        new_coeff = coeff * sign^power

        term_vec[ii] = new_term
        coeff_vec[ii] = new_coeff
    end

    return psum
end


# Overload of `mcapplytoall!` for `PauliRotation` gates acting onto `VectorPauliSum`s with `PathProperties`
# coefficients. Identical branch-sampling logic to the generic method in `vectormontecarlo.jl`, but additionally
# tracks the `nsins`/`ncos`/`freq` counters (without touching `coeff` itself) so that `max_freq`/`max_sins`
# truncations remain usable under Monte Carlo path sampling.
function PropagationBase.mcapplytoall!(gate::PauliRotation, psum::VectorPauliSum{TV,CV}, theta; squared=false, kwargs...) where {TV,CT<:PathProperties,CV<:AbstractVector{CT}}
    power = squared ? 2 : 1

    sin_val = sin(theta)
    abs_sin = abs(sin_val)^power
    sin_sign = sign(sin_val)^power

    cos_val = cos(theta)
    abs_cos = abs(cos_val)^power
    cos_sign = sign(cos_val)^power

    # >= 1 for power=1 and = 1 for power=2
    normalization = abs_sin + abs_cos

    # probability of branching off
    p = abs_sin / normalization

    gate_mask = symboltoint(paulitype(psum), gate.symbols, gate.qinds)

    term_vec = paulis(psum)
    coeff_vec = coefficients(psum)
    AK.foreachindex(term_vec) do ii
        term = term_vec[ii]
        coeff = coeff_vec[ii]

        if !commutes(gate_mask, term)
            if rand() < p
                # Apply sine branch
                new_term, prod_sign = paulirotationproduct(gate_mask, term)
                new_coeff = _incrementsinandfreq(coeff * normalization * sin_sign * prod_sign^power)
                # Update in place
                term_vec[ii] = new_term
                coeff_vec[ii] = new_coeff
            else
                # Apply cos branch
                new_coeff = _incrementcosandfreq(coeff * normalization * cos_sign)
                # Update in place
                coeff_vec[ii] = new_coeff
            end
        end
    end

    return psum
end