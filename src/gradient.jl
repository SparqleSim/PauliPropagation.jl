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
`overlapfunc` must be linear in the coefficients of the `VectorPauliSum` it is handed, e.g. any of
`overlapwithzero`, `overlapwithplus`, `overlapwithcomputational`, `overlapwithmaxmixed`, or
`overlapwithpaulisum`.
`kwargs` are passed on to `applymergetruncate!` in the forward sweep and the operator side of the
backward sweep.
Returns `(expec, grad)`.
"""
function rewindgradient(circuit, psum::AbstractPauliSum, params, overlapfunc; kwargs...)
    return rewindgradient!(circuit, deepcopy(VectorPauliSum(psum)), params, overlapfunc; kwargs...)
end

"""
    rewindgradient!(circuit, psum::VectorPauliSum, params, overlapfunc; kwargs...)

In-place version of `rewindgradient`. Only allows the Pauli sum to be a `VectorPauliSum`.
"""
function rewindgradient!(circuit, psum::VectorPauliSum, params, overlapfunc; kwargs...)
    return rewindgradient!(circuit, PropagationCache(psum), params, overlapfunc; kwargs...)
end

function rewindgradient!(circuit, forward_cache::AbstractPauliPropagationCache, params, overlapfunc; thread::Bool=true, kwargs...)
    # check that the only parameterized gates are PauliRotations
    @assert all(gate -> !isparametrized(gate) || gate isa PauliRotation, circuit) "All parameterized gates must be PauliRotations."

    nq = nqubits(forward_cache)

    # forward sweep: ordinary Heisenberg propagation, exactly as in `propagate`.
    propagate!(circuit, forward_cache, params; thread, kwargs...)
    expec = overlapfunc(activesum(forward_cache))

    # seed the operator sum directly from the final operator
    # and the dual sum from overlapfunc applied to each of its Pauli strings individually.
    dual_terms = copy(activeterms(forward_cache))
    dual_coeffs = Vector{ComplexF64}(undef, length(forward_cache))
    # fill in the coefficients
    AK.foreachindex(dual_terms; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do ii
        dual_coeffs[ii] = overlapfunc(_singletonvectorpaulisum(nq, dual_terms[ii]))
    end
    dual_sum = VectorPauliSum(nq, dual_terms, dual_coeffs, length(dual_terms))

    reverse_cache = forward_cache
    dual_cache = PropagationCache(dual_sum)

    # backward sweep: undo gates in the circuit's own original order
    undo_circuit, undo_params = _preparecircuit(circuit, params, false)

    commutator_buffer = Vector{ComplexF64}(undef, length(forward_cache))
    state = _BackwardSweepState(reverse_cache, dual_cache, nq, zeros(length(params)), 0, commutator_buffer)
    PropagationBase._propagate!(_undostep!, undo_circuit, state, undo_params; thread, kwargs...)

    return expec, state.grad
end


# A length-1 VectorPauliSum for a single Pauli string, for feeding into `overlapfunc`.
function _singletonvectorpaulisum(nq::Int, term, coeff=1.0)
    return VectorPauliSum(nq, [term], [ComplexF64(coeff)])
end


# State propagated through the backward sweep
# carries everything it needs to compute the gradient on the fly
mutable struct _BackwardSweepState{OC,DC}
    op_cache::OC
    dual_cache::DC
    nq::Int
    grad::Vector{Float64}
    k::Int
    commutator_buffer::Vector{ComplexF64}
end

# Records the gradient component for PauliRotation
function _undostep!(gate::PauliRotation, state::_BackwardSweepState, theta; thread::Bool=true, kwargs...)
    gate_mask = symboltoint(state.nq, gate.symbols, gate.qinds)

    grad_contribution = _generatorcommutatordot(
        gate_mask, activeterms(state.op_cache), activecoeffs(state.op_cache),
        activeterms(state.dual_cache), activecoeffs(state.dual_cache), state.commutator_buffer; thread,
    )
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


# Gradient contribution for one gate: real((i/2) * dual_sum(commutator(generator, op_sum))). 
# For every operator term that anticommutes with the generator, the commutator sends it to exactly one new Pauli string,
# so this is a single parallel pass over the operator's terms, each doing one bit-level commutator
# and one binary search into the (sorted) dual sum's active terms
function _generatorcommutatordot(gate_mask::TT, op_terms, op_coeffs, dual_terms_sorted, dual_coeffs_sorted, buffer; thread::Bool=true) where TT
    n = length(op_terms)
    if n == 0
        return zero(ComplexF64)
    end
    n_dual = length(dual_terms_sorted)
    # the operator's active size does not alwaysshrink monotonically going backward 
    # merges cancancel terms out of order along the trajectory
    # so we need to check for resizes
    if n > length(buffer)
        resize!(buffer, n)
    end
    partial = @view buffer[1:n]

    AK.foreachindex(op_terms; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do ii
        term = op_terms[ii]
        if commutes(term, gate_mask)
            partial[ii] = zero(ComplexF64)
        else
            new_term, comm_coeff = commutator(gate_mask, term)
            idx = searchsortedfirst(dual_terms_sorted, new_term)
            if idx <= n_dual && dual_terms_sorted[idx] == new_term
                partial[ii] = comm_coeff * op_coeffs[ii] * dual_coeffs_sorted[idx]
            else
                partial[ii] = zero(ComplexF64)
            end
        end
    end

    return AK.mapreduce(identity, +, partial; init=zero(ComplexF64), max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
end

# Caps dual_cache's support down to op_cache's (already truncated) support. Both sides are sorted and
# duplicate-free at this point (dual_cache was just `merge!`d, op_cache just `applymergetruncate!`d), 
#so this is a single sequential merge-join of the two term arrays -- O(n+m) with only sequential memory
# versus a per-term binary search (O(n log m) random access, plus thread-dispatch overhead per
# call). Keeps the dual sum from ever growing past the operator sum's own size, without ever building it
# as an explicit, possibly exponential-size, operator.
function _intersectfilter!(dual_cache, op_cache; thread::Bool=true)
    dual_terms_sorted = activeterms(dual_cache)
    op_terms_sorted = activeterms(op_cache)
    flags = activeflags(dual_cache)

    n_dual = length(dual_terms_sorted)
    n_op = length(op_terms_sorted)

    i = 1
    j = 1
    @inbounds while i <= n_dual && j <= n_op
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
    @inbounds while i <= n_dual
        flags[i] = false
        i += 1
    end

    filterviaflags!(dual_cache; thread)

    return dual_cache
end
