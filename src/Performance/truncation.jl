###
##
# Truncation and dispatch shared by the fused gate applications.
##
###

# Hands the gate back to the library's own `applymergetruncate!`, the method one step less specific
# than every overload in this module. Used by each of them when it cannot fuse.
@inline _invokedefault(gate, prop_cache, param; kwargs...) =
    invoke(PauliPropagation.applymergetruncate!,
        Tuple{typeof(gate),PauliPropagation.AbstractPauliPropagationCache,typeof(param)},
        gate, prop_cache, param; kwargs...)

# the fused path truncates in one pass, so an option it does not implement must not be silently dropped
function _checkunusedkwargs(kwargs)
    if !isempty(kwargs)
        throw(ArgumentError("keyword arguments $(join(keys(kwargs), ", ")) are not supported by Performance.propagate"))
    end
    return
end

@inline function _fusedtruncfunc(pstr, coeff; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc)
    PauliPropagation.truncateweight(pstr, max_weight) && return true
    return _coefftruncfunc(pstr, coeff; min_abs_coeff, max_freq, max_sins, customtruncfunc)
end

"""
    _coefftruncfunc(pstr, coeff; min_abs_coeff, max_freq, max_sins, customtruncfunc)

The truncations that read the coefficient, for a gate that truncates after merging rather than as
it produces terms.

A product below the threshold on its own still shifts a term it collides with, so dropping it as it
is produced leaves that term artificially large, and more terms then clear the threshold than
should. Weight is not tested here, but in `_fusedtruncfunc`, which reads the Pauli string alone.
"""
@inline function _coefftruncfunc(pstr, coeff; min_abs_coeff, max_freq, max_sins, customtruncfunc)
    PauliPropagation.truncatemincoeff(coeff, min_abs_coeff) && return true
    PauliPropagation.truncatefrequency(coeff, max_freq) && return true
    PauliPropagation.truncatesins(coeff, max_sins) && return true
    !isnothing(customtruncfunc) && customtruncfunc(pstr, coeff) && return true
    return false
end
