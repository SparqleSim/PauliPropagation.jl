
function propagate(circuit, term_sum::AbstractTermSum, parameters=nothing; kwargs...)
    return propagate!(circuit, deepcopy(term_sum), parameters; kwargs...)
end


function propagate!(circuit, term_sum::AbstractTermSum, params=nothing; kwargs...)

    prop_cache = propagate!(circuit, PropagationCache(term_sum), params; kwargs...)

    # extracts the original input term sum
    return extractsum!(prop_cache, term_sum)
end

"""
    propagate!(circuit, prop_cache::AbstractPropagationCache, params=nothing; kwargs...)

Propagate a term sum through under the action of gates in `circuit` using a propagation cache `prop_cache`. 
This is in-place and modifies `prop_cache` along with the term sums it carries.
Parameters for the parametrized gates in `circuit` are given by `params`, and need to be passed as if the circuit was applied as written in the Schrödinger picture.
If params are not passed, the circuit must contain only non-parametrized `StaticGates`.
`kwargs` are passed to the lower-level functions `applymergetruncate!`, `applytoall!`, and `apply`,
as well as `merge!` and `truncate!`.
Default truncation kwargs are `min_abs_coeff` and `customtruncfunc`.
"""
function propagate!(circuit, prop_cache::AbstractPropagationCache, params=nothing; kwargs...)
    # directly call the internal function
    # the higher-level function can be overloaded if needed
    return _propagate!(circuit, prop_cache, params; kwargs...)
end

function _propagate!(circuit, prop_cache::AbstractPropagationCache, params=nothing; kwargs...)
    return _propagate!(applymergetruncate!, circuit, prop_cache, params; kwargs...)
end

"""
    _propagate!(stepfunc, circuit, target, params=nothing; kwargs...)

Generic gate-iteration loop shared by `propagate!` and Monte Carlo variants like `mcpropagate!`/`mcsample!`.
Promotes/validates `circuit`/`params`, then calls `stepfunc(gate, target, [param]; kwargs...)` for each gate in order.
`target` is typically a propagation cache, but can be any object `stepfunc` knows how to mutate in place
(e.g. a bare `VectorPauliSum` for `mcsample!`).
"""
function _propagate!(stepfunc::F, circuit, target, params=nothing; kwargs...) where {F}
    # if circuit is actually a single gate, promote it to a list [gate]
    # similarly the params if it is a single number
    circuit, params = _promotecircandparams(circuit, params)

    # if params is nothing, the circuit must contain only StaticGates
    # also check if the length of params equals the number of parametrized gates
    _checknumberofparams(circuit, params)

    # A useful iteration tool
    parameter_iterator = Iterators.Stateful(params)

    _withworkers(target, get(kwargs, :thread, true)) do
        for gate in circuit
            if isa(gate, ParametrizedGate)
                param = popfirst!(parameter_iterator)
                stepfunc(gate, target, param; kwargs...)
            else
                stepfunc(gate, target; kwargs...)
            end

            # free unless `@countpaulis` or `@peakpaulis` installed a counter
            _recordsize!(target)
        end
    end

    return target
end

# A cache held in arrays or split over zones keeps a worker on every thread for the whole loop
# (see `threading_utils.jl`); any other target runs the loop as it is.
_withworkers(f::F, target, thread::Bool) where {F} = f()
_withworkers(f::F, target::AbstractPropagationCache, thread::Bool) where {F} =
    thread ? _withworkers(StorageType(target), f) : f()
_withworkers(::StorageType, f::F) where {F} = f()
_withworkers(::ArrayStorage, f::F) where {F} = withworkers(f)

# A propagation over a multi sum keeps its workers up from the first gate to the last.
_withworkers(::MultiSumStorage, f::F) where {F} = withworkers(f)

"""
    applymergetruncate!(gate, prop_cache::AbstractPropagationCache; kwargs...)
    applymergetruncate!(gate, prop_cache::AbstractPropagationCache, parameter; kwargs...)

1st-level function below `propagate!` that applies one gate to all terms in the main term sum `mainsum(prop_cache)`, 
potentially using an auxiliary term sum `auxsum(prop_cache)` in the process. 
All terms are then merged and deduplicated into the main term sum.
Truncations are performed after merging.
This function can be overwritten for a custom gate if the lower-level functions `applytoall!`, and `apply` are not sufficient.
"""
function applymergetruncate!(gate, prop_cache::AbstractPropagationCache, args...; kwargs...)
    apply_merge_truncate!() = _applymergetruncate!(gate, prop_cache, args...; kwargs...)
    return _with_threads_freed_for(apply_merge_truncate!, StorageType(prop_cache))
end

# the array kernels of AcceleratedKernels start tasks of their own, which the workers make room for
_with_threads_freed_for(f::F, ::StorageType) where {F} = f()
_with_threads_freed_for(f::F, ::ArrayStorage) where {F} = _with_threads_freed_for(f)

function _applymergetruncate!(gate, prop_cache::AbstractPropagationCache, args...; kwargs...)
    # args is usually expected to be empty or contain a parameter for the gate
    # prop_cache is modified in place
    applytoall!(gate, prop_cache, args...; kwargs...)

    # usually this merges from some auxillary term sum into the main term sum
    # for vector-based caches, it deduplicates within the main term sum
    if requiresmerging(gate, prop_cache)
        merge!(prop_cache; kwargs...)
    end

    truncate!(prop_cache; kwargs...)

    return
end

"""
    requiresmerging(gate, prop_cache::AbstractPropagationCache)::Bool

Helper function that indicates whether merging is required after applying `gate` to `prop_cache`.
Defaults to `true`.
Overload it to return `false` for a gate and cache whose `applytoall!` never creates duplicate terms.
Such an `applytoall!` must then leave all terms in `mainsum(prop_cache)` and `auxsum(prop_cache)` empty, because nothing is moved back afterwards.
"""
requiresmerging(gate, prop_cache::AbstractPropagationCache) = requiresmerging(gate)

# one-argument fallback so gate-only overloads keep working
requiresmerging(gate) = true

"""
    applytoall!(gate, prop_cache::AbstractPropagationCache; kwargs...)
    applytoall!(gate, prop_cache::AbstractPropagationCache, parameter; kwargs...)

1st-level function below `propagate!` that applies one gate to all terms in the main term sum `term_sum = mainsum(prop_cache)`, 
potentially using an auxiliary term sum `aux_term_sum = auxsum(prop_cache)` in the process. 
After this function, all terms remaining in `term_sum` and `aux_term_sum` are merged, unless `requiresmerging(gate, prop_cache)` is `false`,
in which case all terms must be left in `term_sum` and `aux_term_sum` must be empty.
By default, `apply(gate, term, coeff, args...; kwargs...)` is mapped over every term with `flatmap!`, which every storage implements,
so a custom gate only needs `apply`.
The default implementation consumes `thread` to control that mapping; it does not forward `thread` to per-term `apply` calls.
This function can be overwritten for a custom gate if the lower-level function `apply()` is not sufficient.
In particular, this function can be used to manipulate both `term_sum` and `aux_term_sum` at the same time to reduce memory movement.
Note that manipulating `term_sum` on anything other than the current term will likely lead to errors.
"""
function applytoall!(gate, prop_cache::AbstractPropagationCache, args...; thread::Bool=true, kwargs...)
    apply_gate(term, coeff) = apply(gate, term, coeff, args...; kwargs...)
    return flatmap!(apply_gate, prop_cache; thread)
end


"""
    apply(gate::StaticGate, term, coeff; kwargs...)
    apply(gate::ParametrizedGate, term, coeff; kwargs...)

Lowest-level function that applies one gate to one term and its coefficient.
Is expected to return a tuple of (new_term, new_coeff) pairs.
This function must be overloaded for each custom gate type.
Common mistakes are to return a single pair instead of a tuple of pairs, 
such as `(new_term, new_coeff)`, instead of `((new_term, new_coeff),)`.
On an array sum, several tasks call `apply` once to count the pairs and once to write them, possibly at the same time on different terms,
so it must return the same pairs for the same term every time and must not return a one-shot iterator.

Example:
```julia
function apply(gate::NewStaticGate, term, coeff)
    # how some gate acts in the term and updates coeff
    ...
    return ((new_term1, new_coeff1), (new_term2, new_coeff2), ...)
end
function apply(gate::NewParametrizedGate, term, coeff, param)
    # how some gate acts in the term and updates coeff
    # a parameter `param` will be automatically passed if NewParametrizedGate <: ParametrizedGate
    ...
    return ((new_term1, new_coeff1), (new_term2, new_coeff2), ...)
end
```
"""
@inline apply(gate, args...; kwargs...) = _thrownotimplemented(gate, :apply)




function _promotecircandparams(circ, params)
    # if users pass a gate, we assume that thetas also requires a `[]` around it
    if circ isa Gate
        circ = [circ]

        if !isnothing(params)
            params = [params]
        end
    end

    if isnothing(params)
        params = []
    end

    return circ, params
end

function _checknumberofparams(circ, params)
    nparams = countparameters(circ)

    if nparams != length(params)
        throw(ArgumentError(
            "The number of parameters must match the number of parametrized gates in the circuit. " *
            "countparameters(circ)=$nparams, length(params)=$(length(params)).")
        )
    end

    return
end
