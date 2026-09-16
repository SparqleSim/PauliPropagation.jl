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
    dual_cache = PropagationCache(_dualsum(forward_cache, overlapfunc; thread))

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
    merge!(state.dual_cache; thread)
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


# The three operations below that are not gate applications are written once per storage, like
# the term sum interface in `PropagationBase`. A multi sum hands each of them to its zones.

# The dual sum of `prop_cache`: the same Pauli strings, each with the overlap it has on its own.
_dualsum(prop_cache::AbstractPropagationCache, overlapfunc; thread::Bool=true) =
    _dualsum(StorageType(prop_cache), prop_cache, overlapfunc; thread)

function _dualsum(::PropagationBase.DictStorage, prop_cache, overlapfunc; thread::Bool=true)
    nq = nqubits(prop_cache)
    dual_sum = PauliSum(nq, Dict{termtype(prop_cache),ComplexF64}())
    for (term, _) in mainsum(prop_cache)
        set!(dual_sum, term, overlapfunc(_singletonvectorpaulisum(nq, term)))
    end
    return dual_sum
end

function _dualsum(::PropagationBase.ArrayStorage, prop_cache, overlapfunc; thread::Bool=true)
    nq = nqubits(prop_cache)
    dual_terms = copy(activeterms(prop_cache))
    dual_coeffs = Vector{ComplexF64}(undef, length(dual_terms))
    AK.foreachindex(dual_terms; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do ii
        dual_coeffs[ii] = overlapfunc(_singletonvectorpaulisum(nq, dual_terms[ii]))
    end
    return VectorPauliSum(nq, dual_terms, dual_coeffs, length(dual_terms))
end

# the dual sum shares the zone assignment, so a Pauli string sits in the same zone of both sums
function _dualsum(::MultiSumStorage, prop_cache, overlapfunc; thread::Bool=true)
    dual_zones = map(zonecache -> _dualsum(zonecache, overlapfunc; thread), zonecaches(prop_cache))
    return Base.typename(typeof(mainsum(prop_cache))).wrapper(nsites(prop_cache), dual_zones, zonemap(prop_cache))
end

# A length-1 VectorPauliSum for a single Pauli string, for feeding into `overlapfunc`.
function _singletonvectorpaulisum(nq::Int, term, coeff=1.0)
    return VectorPauliSum(nq, [term], [ComplexF64(coeff)])
end


# Gradient contribution for one gate: real((i/2) * dual_sum(commutator(generator, op_sum))).
# Every operator term that anticommutes with the generator commutes to exactly one Pauli string,
# so this is a single pass over the operator's terms, each looking its commutator up in the dual sum.
_generatorcommutatordot(gate_mask, op_cache, dual_cache; thread::Bool=true) =
    _generatorcommutatordot(StorageType(op_cache), gate_mask, op_cache, dual_cache; thread)

function _generatorcommutatordot(::PropagationBase.DictStorage, gate_mask, op_cache, dual_cache; thread::Bool=true)
    dual_sum = mainsum(dual_cache)
    total = zero(ComplexF64)
    for (term, coeff) in mainsum(op_cache)
        commutes(term, gate_mask) && continue
        new_term, comm_coeff = commutator(gate_mask, term)
        total += comm_coeff * coeff * getcoeff(dual_sum, new_term)
    end
    return total
end

# the dual sum is sorted here, so the lookup is a binary search
function _generatorcommutatordot(::PropagationBase.ArrayStorage, gate_mask, op_cache, dual_cache; thread::Bool=true)
    op_terms, op_coeffs = activeterms(op_cache), activecoeffs(op_cache)
    dual_terms_sorted, dual_coeffs_sorted = activeterms(dual_cache), activecoeffs(dual_cache)
    n_dual = length(dual_terms_sorted)

    task_partitioner, n_tasks = PropagationBase._preparetasks(length(op_terms), thread)
    partials = zeros(ComplexF64, n_tasks)

    AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
        total = zero(ComplexF64)
        @inbounds for ii in task_partitioner[task_id]
            term = op_terms[ii]
            commutes(term, gate_mask) && continue
            new_term, comm_coeff = commutator(gate_mask, term)
            idx = searchsortedfirst(dual_terms_sorted, new_term)
            (idx <= n_dual && dual_terms_sorted[idx] == new_term) || continue
            total += comm_coeff * op_coeffs[ii] * dual_coeffs_sorted[idx]
        end
        partials[task_id] = total
    end

    return sum(partials)
end

# the commutators of an operator zone all sit in the one dual zone the generator's mask sends it to
function _generatorcommutatordot(::MultiSumStorage, gate_mask, op_cache, dual_cache; thread::Bool=true)
    partials = zeros(ComplexF64, nzones(op_cache))
    PropagationBase._eachzone(op_cache, thread) do zone
        dual_zone = PropagationBase._xortarget(zonemap(op_cache), zone, gate_mask)
        partials[zone] = _generatorcommutatordot(gate_mask, zonecaches(op_cache)[zone], zonecaches(dual_cache)[dual_zone]; thread=false)
    end
    return sum(partials)
end


# Caps dual_cache's support down to op_cache's (already truncated) support.
function _intersectfilter!(dual_cache, op_cache; thread::Bool=true)
    _intersectfilter!(StorageType(dual_cache), dual_cache, op_cache; thread)
    return dual_cache
end

function _intersectfilter!(::PropagationBase.DictStorage, dual_cache, op_cache; thread::Bool=true)
    op_terms = storage(mainsum(op_cache))
    filter!(term_and_coeff -> haskey(op_terms, first(term_and_coeff)), storage(mainsum(dual_cache)))
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
