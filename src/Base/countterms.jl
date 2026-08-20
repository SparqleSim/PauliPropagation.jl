###
##
# Counting how many terms a propagation carries, as `@countpaulis` and `@peakpaulis` do.
# `_propagate!` calls `_recordsize!` after every gate, which does nothing unless a counter is installed.
##
###

# Global because propagations are reached through arbitrarily deep call stacks.
# The stack of installed counters is an immutable snapshot behind an atomic field,
# so the common no-counter check is a race-free single load; the lock serializes everything else.
mutable struct _CounterStack
    @atomic stack::Tuple{Vararg{Vector{Int}}}
    const lock::ReentrantLock
end

const _COUNTERS = _CounterStack((), ReentrantLock())

"""
    pushcounter!(counts::Vector{Int})
    popcounter!(counts::Vector{Int})

Start and stop appending the number of terms after every gate of every propagation to `counts`.
Counters stack, so an outer one also sees the gates seen by an inner one.
"""
function pushcounter!(counts::Vector{Int})
    lock(_COUNTERS.lock) do
        @atomic _COUNTERS.stack = ((@atomic _COUNTERS.stack)..., counts)
    end
    return counts
end

function popcounter!(counts::Vector{Int})
    lock(_COUNTERS.lock) do
        stack = @atomic _COUNTERS.stack
        idx = findlast(c -> c === counts, stack)
        isnothing(idx) || @atomic _COUNTERS.stack = (stack[1:idx-1]..., stack[idx+1:end]...)
    end
    return counts
end

@inline function _recordsize!(target)
    # the common case is no counter at all, which must stay free
    isempty(@atomic _COUNTERS.stack) && return
    _pushsize!(target)
    return
end

@noinline function _pushsize!(target)
    n = _termcount(target)
    isnothing(n) && return

    lock(_COUNTERS.lock) do
        for counts in (@atomic _COUNTERS.stack)
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

_peak(counts) = maximum(counts; init=0)

# Shared expansion for `@countpaulis`-style macros; downstream packages build theirs from this.
# Interpolating the functions as values keeps the expansion independent of the caller's module.
# A leading assignment is peeled off and performed outside the `try` block, because `try` opens a
# soft local scope in which assignments to new variables would not reach the calling scope.
function _countingexpr(expr, reducefunc)
    is_assignment = isa(expr, Expr) && expr.head === :(=)
    lhs = is_assignment ? expr.args[1] : nothing
    rhs = is_assignment ? expr.args[2] : expr

    counts = gensym(:counts)
    value = gensym(:value)
    assignment = is_assignment ? Expr(:(=), esc(lhs), value) : nothing

    return quote
        local $counts = $pushcounter!(Int[])
        local $value = try
            $(esc(rhs))
        finally
            $popcounter!($counts)
        end
        $assignment
        $reducefunc($counts)
    end
end
