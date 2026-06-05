# Custom truncation function

# Define the custom truncation functions by dissipation-assisted damping
"""
    truncatedampingcoeff(
        pstr::Integer, 
        coeff::Real, 
        gamma::Number, 
        min_abs_coeff::Float64
    )

Custom truncation function with dissipation-assisted damping of coefficients.
It returns `true` (indicating to truncate) if the coefficient exponentially damped by the Pauli weight drops below `min_abs_coeff`.

The function evaluates the condition:
`abs(coeff) * 10^(-gamma * countweight(pstr)) < min_abs_coeff`

`gamma` is the damping factor controlling the rate of exponential decay with Pauli weight.

To turn this function into a custom truncation function for `propagate()`, you need to define a _closure_ 
that only takes the `pstr` and `coeff` as arguments, but has `gamma` and `min_abs_coeff` fixed.

Example:
```julia
customtruncfunc = (pstr, coeff) -> truncatedampingcoeff(pstr, coeff, 0.5, 1e-10) 
```
"""
function truncatedampingcoeff(
    pstr::PauliStringType, coeff, gamma::Number, min_abs_coeff::Real
)

    return abs(tonumber(coeff)) * 10.0^(-gamma * countweight(pstr)) < min_abs_coeff
end


## TODO: Make actual use of this file or remove.
## from here on, the are just for the backend

## Truncations on the Pauli string:

# Return `true` if an integer Pauli string should be truncated because its weight (i.e., number of non-identity Paulis) is larger than `max_weight`. 
function truncateweight(pstr::Integer, max_weight::Real)
    return countweight(pstr) > max_weight
end

function truncateweight(pstr, max_weight::Real)
    return false
end


## Truncations on the coefficient:

# Truncations on unsuitable coefficient types defaults to false.
function truncatemincoeff(coeff, min_abs_coeff::Real)
    return false
end


# Return `true` if `abs(coeff) < min_abs_coeff`. 
function truncatemincoeff(coeff::Number, min_abs_coeff::Real)
    return abs(coeff) < min_abs_coeff
end


# Return `true` if `abs(path_property.coeff) < min_abs_coeff`. 
function truncatemincoeff(path_property::PProp, min_abs_coeff::Real) where {PProp<:PathProperties}
    if hasfield(PProp, :coeff)
        return abs(path_property.coeff) < min_abs_coeff
    else
        return false
    end
end


# Return `true` if  `PathProperties.freq > max_freq`. Truncations on coefficients should default to false if it is not applicable for a type.
function truncatefrequency(coeff, max_freq::Real)
    return false
end


# Return `true` if  `path_properties.freq > max_freq`.
function truncatefrequency(path_properties::PProp, max_freq::Real) where {PProp<:PathProperties}
    if hasfield(PProp, :freq)
        return path_properties.freq > max_freq
    else
        return false
    end
end

# Return `true` if  `PathProperties.nsins > max_sins`. Truncations on coefficients should default to false if it is not applicable for a type.
function truncatesins(coeff, max_sins::Real)
    return false
end


# Return `true` if  `path_properties.nsins > max_sins`.
function truncatesins(path_properties::PProp, max_sins::Real) where {PProp<:PathProperties}
    if hasfield(PProp, :nsins)
        return path_properties.nsins > max_sins
    else
        return false
    end
end

