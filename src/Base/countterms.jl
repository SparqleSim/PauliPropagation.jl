###
##
# Counting how many terms a propagation carries, as `@countpaulis` and `@peakpaulis` do.
# `_propagate!` calls `_recordsize!` after every gate, which does nothing unless a counter is installed.
##
###

"""
    pushcounter!(counts::Vector{Int})
    popcounter!(counts::Vector{Int})

Start and stop appending the number of terms after every gate of every propagation to `counts`.
Counters stack, so an outer one also sees the gates seen by an inner one.
"""
function pushcounter!(counts::Vector{Int})
    lock(_COUNTERLOCK) do
        push!(_COUNTERS, counts)
    end
    return counts
end

function popcounter!(counts::Vector{Int})
    lock(_COUNTERLOCK) do
        idx = findlast(c -> c === counts, _COUNTERS)
        isnothing(idx) || deleteat!(_COUNTERS, idx)
    end
    return counts
end

# Global because propagations are reached through arbitrarily deep call stacks,
# locked because several of them may run concurrently.
const _COUNTERS = Vector{Int}[]
const _COUNTERLOCK = ReentrantLock()

@inline function _recordsize!(target)
    # the common case is no counter at all, which must stay free
    isempty(_COUNTERS) && return
    _pushsize!(target)
    return
end

@noinline function _pushsize!(target)
    n = _termcount(target)
    isnothing(n) && return

    lock(_COUNTERLOCK) do
        for counts in _COUNTERS
            push!(counts, n)
        end
    end

    return
end

"""
    _termcount(target)

The number of terms currently held by whatever `_propagate!` carries between gates,
or `nothing` if that is not something with a term count.
Overload this for custom propagation states.
"""
_termcount(target::Union{AbstractPropagationCache,AbstractTermSum}) = length(target)
_termcount(target) = nothing
