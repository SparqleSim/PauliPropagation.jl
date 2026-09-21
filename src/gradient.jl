### gradient.jl
##
# Computes gradients of a Pauli propagation expectation value via backpropagation.
##
###


"""
    rewindgradient(circuit, psum::AbstractPauliSum, params, overlapfunc; thread=true, kwargs...)

Compute `overlapfunc(propagate(circuit, psum, params; kwargs...))` together with its gradient with
respect to `params`, in one paired forward and backward sweep costing O(length(circuit)) gate applications.
The only parametrized gates can be `PauliRotation`s. 
All other gates must not be parametrized or frozen via `freeze(gate, param)` before inputting to `rewindgradient`.
Noise channels are not supported, frozen or not, because the backward sweep cannot undo them.
`overlapfunc` must be linear in the coefficients of the Pauli sum it is handed, e.g. any of
`overlapwithzero`, `overlapwithplus`, `overlapwithcomputational`, `overlapwithmaxmixed`, or
`overlapwithpaulisum`.
Both sweeps run with the gate implementations of the type of `psum`, so a `MultiPauliSum` runs them zone by zone.
`kwargs` are passed on to `applymergetruncate!` in the forward sweep and the operator side of the
backward sweep.
Returns `(expec, grad)`.

This design is adapted from the publication ``Backpropagating Pauli Propagation'' by Lin et al. (arXiv:2607.15184).
"""
function rewindgradient(circuit, psum::AbstractPauliSum, params, overlapfunc; kwargs...)
    return rewindgradient!(circuit, deepcopy(psum), params, overlapfunc; kwargs...)
end

"""
    rewindgradient!(circuit, psum::AbstractPauliSum, params, overlapfunc; kwargs...)
    rewindgradient!(circuit, prop_cache::AbstractPauliPropagationCache, params, overlapfunc; kwargs...)

In-place version of `rewindgradient`, which leaves the propagated operator in `psum` or `prop_cache`.
"""
function rewindgradient!(circuit, psum::AbstractPauliSum, params, overlapfunc; kwargs...)
    return rewindgradient!(circuit, PropagationCache(psum), params, overlapfunc; kwargs...)
end

function rewindgradient!(circuit, forward_cache::AbstractPauliPropagationCache, params, overlapfunc; thread::Bool=true, kwargs...)
    # check that the only parameterized gates are PauliRotations
    @assert all(gate -> isa(gate, StaticGate) || gate isa PauliRotation, circuit) "All parameterized gates must be PauliRotations."
    @assert all(_isrewindable, circuit) "Noise channels are not supported because they cannot be undone."

    # forward sweep: ordinary Heisenberg propagation, exactly as in `propagate`.
    propagate!(circuit, forward_cache, params; thread, kwargs...)
    expec = overlapfunc(activesum(forward_cache))

    # the dual sum starts from the final operator, with overlapfunc applied to each of its Pauli strings individually
    dual_cache = _dualcache(forward_cache, overlapfunc; thread)

    # backward sweep: undo gates in the circuit's own original order
    undo_circuit, undo_params = _preparecircuit(circuit, params, false)
    state = _BackwardSweepState(forward_cache, dual_cache, zeros(length(params)), 0)
    PropagationBase._propagate!(_undostep!, undo_circuit, state, undo_params; thread, kwargs...)

    return expec, state.grad
end


# The backward sweep undoes a gate by re-applying its Schrödinger-picture form,
# which only recovers the operator if the gate is invertible.
_isrewindable(gate) = true
_isrewindable(gate::ParametrizedNoiseChannel) = false
_isrewindable(gate::FrozenGate) = _isrewindable(gate.gate)


# State propagated through the backward sweep
# carries everything it needs to compute the gradient on the fly
mutable struct _BackwardSweepState{OC,DC}
    op_cache::OC
    dual_cache::DC
    grad::Vector{Float64}
    k::Int
end

# the backward sweep is carried by this state instead of a cache, so the operator sum is what is
# counted and what keeps its zone workers up
PropagationBase._termcount(state::_BackwardSweepState) = length(state.op_cache)
PropagationBase._withworkers(f::F, state::_BackwardSweepState, thread::Bool) where {F} =
    PropagationBase._withworkers(f, state.op_cache, thread)

# Records the gradient component for PauliRotation
function _undostep!(gate::PauliRotation, state::_BackwardSweepState, theta; thread::Bool=true, kwargs...)
    gate_mask = symboltoint(paulitype(state.op_cache), gate.symbols, gate.qinds)

    grad_contribution = _generatorcommutatordot(gate_mask, state.op_cache, state.dual_cache; thread)
    state.k += 1
    state.grad[state.k] = real(0.5im * grad_contribution)

    # the operator sum truncates normally; the dual sum never truncates on its own -- instead, right
    # after, its support is capped to whatever the (just-truncated) operator sum still has
    applymergetruncate!(gate, state.op_cache, theta; thread, kwargs...)
    applytoall!(gate, state.dual_cache, theta; thread)
    xormerge!(state.dual_cache, gate_mask; thread)
    _intersectfilter!(state.dual_cache, state.op_cache; thread)

    return state
end

# just undoes the application of a StaticGate. No gradient recorded.
function _undostep!(gate::StaticGate, state::_BackwardSweepState; thread::Bool=true, kwargs...)
    applymergetruncate!(gate, state.op_cache; thread, kwargs...)
    merge!(state.op_cache; thread)
    applytoall!(gate, state.dual_cache; thread)
    merge!(state.dual_cache; thread)
    _intersectfilter!(state.dual_cache, state.op_cache; thread)
    return state
end


# The dual of `prop_cache`: the same Pauli strings, each with the overlap it has on its own.
function _dualcache(prop_cache::AbstractPropagationCache, overlapfunc; thread::Bool=true)
    nq = nqubits(prop_cache)
    singletonoverlap(term, _) = overlapfunc(_singletonvectorpaulisum(nq, term))

    dual_cache = PropagationCache(convertcoefftype(ComplexF64, activesum(prop_cache)))
    return mapcoeffsbypair!(singletonoverlap, dual_cache; thread)
end

# A length-1 VectorPauliSum for a single Pauli string, for feeding into `overlapfunc`.
function _singletonvectorpaulisum(nq::Int, term, coeff=1.0)
    return VectorPauliSum(nq, [term], [ComplexF64(coeff)])
end


# Gradient contribution for one gate: real((i/2) * dual_sum(commutator(generator, op_sum))).
# Every operator term that anticommutes with the generator commutes to exactly one Pauli string,
# so this is a single pass over the operator's terms, each looking its commutator up in the dual sum.
function _generatorcommutatordot(gate_mask, op_cache, dual_cache; thread::Bool=true)
    dual_sum = activesum(dual_cache)

    function commutatoroverlap(term, coeff)
        commutes(term, gate_mask) && return zero(ComplexF64)
        new_term, comm_coeff = commutator(gate_mask, term)
        return comm_coeff * coeff * getmergedcoeff(dual_sum, new_term)
    end

    return mapreduce(commutatoroverlap, +, op_cache; init=zero(ComplexF64), thread)
end


# Caps dual_cache's support down to op_cache's (already truncated) support. This is written once per
# storage, like the term sum interface in `PropagationBase`, and a multi sum hands it to its zones.
#
# TODO: This could become a primitive in `PropagationBase` that walks two term sums together and does
# something with every term that appears in both. Keeping only those terms, as done here, is one use;
# multiplying the two coefficients of each shared term and adding the products up, which is what
# `scalarproduct` does with one lookup per term, is another. The reason it deserves to be a primitive
# of its own, rather than a filter with a lookup in the other sum, is the array case: when both sums
# are sorted, walking them side by side visits each term once, where looking every term up separately
# costs a binary search each and is many times slower.
function _intersectfilter!(dual_cache, op_cache; thread::Bool=true)
    _intersectfilter!(StorageType(dual_cache), dual_cache, op_cache; thread)
    return dual_cache
end

function _intersectfilter!(::PropagationBase.DictStorage, dual_cache, op_cache; thread::Bool=true)
    op_terms = storage(mainsum(op_cache))
    filterterms!(term -> haskey(op_terms, term), dual_cache; thread)
    return
end

# Both sides are sorted and duplicate-free at this point, so this is a merge-join of the two term
# arrays, sliced across tasks the same way `_mergesortedhead!` slices its own two-pointer merge
function _intersectfilter!(::PropagationBase.ArrayStorage, dual_cache, op_cache; thread::Bool=true)
    dual_terms_sorted = activeterms(dual_cache)
    op_terms_sorted = activeterms(op_cache)
    flags = activeflags(dual_cache)

    task_partitioner, n_tasks = PropagationBase._preparetasks(length(dual_terms_sorted), thread)

    AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
        dual_range = task_partitioner[task_id]
        _flagintersection!(flags, dual_terms_sorted, op_terms_sorted, dual_range.start, dual_range.stop)
    end

    filterviaflags!(dual_cache; thread)

    return
end

function _intersectfilter!(::MultiSumStorage, dual_cache, op_cache; thread::Bool=true)
    PropagationBase._eachzone(dual_cache, thread) do zone
        _intersectfilter!(zonecaches(dual_cache)[zone], zonecaches(op_cache)[zone]; thread=false)
    end
    PropagationBase._syncsums!(dual_cache)
    return
end

# flags the dual terms in [lo, hi] that also occur in op_terms_sorted
function _flagintersection!(flags, dual_terms_sorted, op_terms_sorted, lo::Int, hi::Int)
    lo > hi && return
    checkbounds(flags, lo:hi)
    checkbounds(dual_terms_sorted, lo:hi)

    n_op = length(op_terms_sorted)
    i = lo
    j = searchsortedfirst(op_terms_sorted, dual_terms_sorted[lo])

    @inbounds while i <= hi && j <= n_op
        dual_term = dual_terms_sorted[i]
        op_term = op_terms_sorted[j]
        if dual_term == op_term
            flags[i] = true
            i += 1
            j += 1
        elseif dual_term < op_term
            flags[i] = false
            i += 1
        else
            j += 1
        end
    end
    @inbounds while i <= hi
        flags[i] = false
        i += 1
    end

    return
end
