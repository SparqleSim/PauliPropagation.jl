### PathProperties.jl
##
# PathProperties is a type that you can use to keep track of certain properties during propagation
# Currently provide only one concrete type PauliFreqTracker that keeps track of the number of sin and cos factors applied via a PauliRotation gate.
##
###

include("abstracttype.jl")
include("paulifreqtracker.jl")


## Propagation utilities here:

# wrap the coefficients in `PauliFreqTracker` if max_freq or max_sins are used
function _check_wrapping_into_paulifreqtracker(psum::PauliSum, max_freq, max_sins)

    # if max_freq or max_sins are used, and the coefficients are not PathProperties (could be custom)
    # then we wrap the coefficients in `PauliFreqTracker`
    if ((max_freq != Inf) | (max_sins != Inf)) & !(coefftype(psum) <: PathProperties)
        psum = wrapcoefficients(psum, PauliFreqTracker)
        return psum
    end

    # otherwise just return the original psum
    return psum

end

# if the psum is not of type `PauliSum` then we don't touch it 
function _check_wrapping_into_paulifreqtracker(psum, max_freq, max_sins)
    return psum
end

# given a coefficient type, make sure that the PauliSum is has the same coefficient type
# this is only to check whether we had wrapped the coefficients in `PauliFreqTracker`
function _check_unwrap_from_paulifreqtracker(::Type{CT}, psum::PauliSum{TT,CT}) where {TT,CT}
    # in this function the original coefficient type and the current coefficient type are the same
    return psum
end

function _check_unwrap_from_paulifreqtracker(::Type{CT}, psum::PauliSum{TT,PFT}) where {TT,CT,PFT<:PauliFreqTracker}
    # in this function is for when the original coefficient type was not `PauliFreqTracker` but that is what we have
    # we need to unwrap the coefficients

    # if the original coefficient type (CT) is not PauliFreqTracker (PFT), then unwrap
    if CT != PFT
        psum = unwrapcoefficients(psum)
    end
    return psum
end

# anything else is just directly returned
# don't know what do do with it, and we didn't automatically convert it before
function _check_unwrap_from_paulifreqtracker(T::Type, obj)
    return obj
end

# check that max_freq and max_sins are only used a PathProperties type tracking them
function _checkfreqandsinfields(psum, max_freq, max_sins)

    CT = coefftype(psum)

    if !(CT <: PathProperties) & ((max_freq != Inf) | (max_sins != Inf))
        throw(ArgumentError(
            "The `max_freq` and `max_sins` truncations can only be used with coefficients wrapped in `PathProperties` types.\n" *
            "Consider using `wrapcoefficients() with the `PauliFreqTracker` type" *
            " or use the out-of-place `propagate()` function for automatic conversion.")
        )
    end

    if (max_freq != Inf) & (!hasfield(CT, :freq))
        throw(ArgumentError(
            "The `max_freq` truncation is used, but the PathProperties type $CT does not have a `freq` field.")
        )
    end

    if (max_sins != Inf) & (!hasfield(CT, :nsins))
        throw(ArgumentError(
            "The `max_sins` truncation is used, but the PathProperties type $CT does not have a `nsins` field.")
        )
    end

    return
end