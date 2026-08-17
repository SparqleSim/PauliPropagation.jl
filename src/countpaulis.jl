"""
    @countpaulis expr
    @countpaulis lhs = expr

Return the number of Pauli strings after every gate applied inside `expr`, as a `Vector{Int}`.

Counts are taken after merging and truncating, and are ordered as the gates are applied,
which for the default Heisenberg picture is the reverse of the circuit as written.
Every propagation inside `expr` contributes, including `mcpropagate`, `mcsample` and `rewindgradient`.
An assignment is instrumented by prefixing it, and still assigns.

```julia
counts = @countpaulis propagate(circuit, psum, thetas)
counts = @countpaulis psum = propagate(circuit, psum, thetas)
```
"""
macro countpaulis(expr)
    return _countingexpr(expr, :identity)
end

"""
    @peakpaulis expr
    @peakpaulis lhs = expr

Return the largest number of Pauli strings held after any gate applied inside `expr`, or `0` if nothing was propagated.

`expr` is typically a call to a function that propagates internally, possibly several times,
in which case the peak is taken over all of those propagations.
See `@countpaulis` for what is counted and when.

```julia
peak = @peakpaulis myexpectationvalue(circuit, thetas)
peak = @peakpaulis psum, base = headandtails(circuit, obs, thetas, nlayers)
```
"""
macro peakpaulis(expr)
    return _countingexpr(expr, :_peak)
end

_peak(counts) = maximum(counts; init=0)

# Evaluate `expr` with a counter installed and reduce what it counted.
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
        local $counts = PropagationBase.pushcounter!(Int[])
        local $value = try
            $(esc(rhs))
        finally
            PropagationBase.popcounter!($counts)
        end
        $assignment
        $reducefunc($counts)
    end
end
