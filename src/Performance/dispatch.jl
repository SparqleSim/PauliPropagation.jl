###
##
# Dispatch shared by the fused gate applications.
##
###

# Hands the gate back to the library's own `applymergetruncate!`, the method one step less specific
# than every overload in this module. Used by each of them when it cannot fuse. The overloads are
# inlined, so that a call without `fused`, as every stock propagation makes, compiles this path alone.
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
