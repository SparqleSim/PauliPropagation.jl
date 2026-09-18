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
    gate_mask = symboltoint(paulitype(prop_cache), gate.symbols, gate.qinds)

    function rotate(pstr, coeff)
        if commutes(gate_mask, pstr)
            return Unchanged()
        else
            _, kept_coeff, _, new_coeff = splitapply(gate_mask, pstr, coeff, theta; kwargs...)
            return Branch(kept_coeff, new_coeff)
        end
    end

    return xorbranch!(rotate, prop_cache, gate_mask; thread=get(kwargs, :thread, true))
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

# Under Monte Carlo path sampling the branch a term keeps is counted like the branches of a split.
_cosbranch(pth::PathProperties, weight) = _incrementcosandfreq(pth * weight)
_sinbranch(pth::PathProperties, weight) = _incrementsinandfreq(pth * weight)
