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

    # the dual sum starts from the final operator, with overlapfunc applied to each of its Pauli strings individually,
    # and is carried beside the coefficients of the operator through the backward sweep
    op_cache = _withdual(forward_cache, overlapfunc; thread)

    # backward sweep: undo gates in the circuit's own original order
    undo_circuit, undo_params = _preparecircuit(circuit, params, false)
    state = _BackwardSweepState(op_cache, zeros(length(params)), 0)
    PropagationBase._propagate!(_undostep!, undo_circuit, state, undo_params; thread, _backwardkwargs(; kwargs...)...)

    return expec, state.grad
end


# The backward sweep undoes a gate by re-applying its Schrödinger-picture form,
# which only recovers the operator if the gate is invertible.
_isrewindable(gate) = true
_isrewindable(gate::ParametrizedNoiseChannel) = false
_isrewindable(gate::FrozenGate) = _isrewindable(gate.gate)


# The keywords of the backward sweep, in which a custom truncation reads the coefficient of the operator, as in the
# forward sweep.
function _backwardkwargs(; customtruncfunc=nothing, kwargs...)
    if isnothing(customtruncfunc)
        return (; kwargs...)
    end
    opcustomtruncfunc(pstr, coeff) = customtruncfunc(pstr, coeff.op)
    return (; customtruncfunc=opcustomtruncfunc, kwargs...)
end


# State propagated through the backward sweep
# carries everything it needs to compute the gradient on the fly
mutable struct _BackwardSweepState{OC}
    op_cache::OC
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

    state.k += 1
    state.grad[state.k] = _generatorcommutatordot(gate_mask, state.op_cache; thread)

    # the operator sum truncates normally, and the dual sum loses every Pauli string that it loses
    applymergetruncate!(gate, state.op_cache, theta; thread, kwargs...)

    return state
end

# just undoes the application of a StaticGate. No gradient recorded.
function _undostep!(gate::StaticGate, state::_BackwardSweepState; thread::Bool=true, kwargs...)
    applymergetruncate!(gate, state.op_cache; thread, kwargs...)
    return state
end


# The operator of `prop_cache` with its dual: every Pauli string keeps its coefficient and gets the overlap it has on its
# own as its coefficient in the dual sum. The dual coefficients are whatever `overlapfunc` returns for a single Pauli
# string -- real for every overlap the library ships. Nothing else in the backward sweep needs them complex: the dual is
# carried by the same real rotations as the operator, and the gradient pairs the two with the real signs of the rotations.
function _withdual(prop_cache::AbstractPropagationCache, overlapfunc; thread::Bool=true)
    nq = nqubits(prop_cache)
    singletonoverlap(term) = overlapfunc(_singletonvectorpaulisum(nq, term))

    dual_type = Base.promote_op(singletonoverlap, paulitype(prop_cache))
    isconcretetype(dual_type) || (dual_type = ComplexF64)
    coeff_type = _OpDualCoeff{coefftype(prop_cache),float(dual_type)}
    op_cache = PropagationCache(convertcoefftype(coeff_type, activesum(prop_cache)))
    withoverlap(term, coeff) = coeff_type(coeff.op, singletonoverlap(term))
    mapcoeffsbypair!(withoverlap, op_cache; thread)
    return op_cache
end

# The coefficient of a Pauli string in the operator sum and in the dual sum of the backward sweep. The gates act on both
# alike, and the truncations read only the coefficient of the operator, so the dual sum keeps exactly the Pauli strings
# that the operator sum keeps.
struct _OpDualCoeff{CO,CD}
    op::CO
    dual::CD
end

Base.:*(coeff::_OpDualCoeff, factor::Number) = _OpDualCoeff(coeff.op * factor, coeff.dual * factor)
Base.:*(factor::Number, coeff::_OpDualCoeff) = coeff * factor
Base.:+(coeff1::_OpDualCoeff, coeff2::_OpDualCoeff) = _OpDualCoeff(coeff1.op + coeff2.op, coeff1.dual + coeff2.dual)
Base.zero(::Type{_OpDualCoeff{CO,CD}}) where {CO,CD} = _OpDualCoeff(zero(CO), zero(CD))

# a coefficient of the operator alone, without a dual
Base.convert(::Type{_OpDualCoeff{CO,CD}}, coeff::Number) where {CO,CD} = _OpDualCoeff{CO,CD}(coeff, zero(CD))

# the truncations read the coefficient of the operator
truncatemincoeff(coeff::_OpDualCoeff, min_abs_coeff::Real) = truncatemincoeff(coeff.op, min_abs_coeff)
PropagationBase.tonumber(coeff::_OpDualCoeff) = coeff.op
PropagationBase.numcoefftype(::Type{_OpDualCoeff{CO,CD}}) where {CO,CD} = CO

# A length-1 VectorPauliSum for a single Pauli string, for feeding into `overlapfunc`.
function _singletonvectorpaulisum(nq::Int, term, coeff=1.0)
    return VectorPauliSum(nq, [term], [coeff])
end


# Gradient contribution for one gate: real((i/2) * dual_sum(commutator(generator, op_sum))).
# For a term that anticommutes with the generator, (i/2) times the commutator is its product with
# the generator, with the sign that the rotation gives that product (`paulirotationproduct`). So this
# is a single pass over the operator's terms, each adding its coefficient times the dual coefficient
# of its partner, with that sign.
function _generatorcommutatordot(gate_mask, op_cache; thread::Bool=true)
    return _generatorcommutatordot(StorageType(op_cache), gate_mask, op_cache; thread)
end

# A dictionary finds the partner in one lookup, in the zone that holds it for a multi sum, so a plain
# pass with a lookup per term is what it costs.
function _generatorcommutatordot(::StorageType, gate_mask, op_cache; thread::Bool=true)
    # a lookup needs the sum merged
    merge!(op_cache; thread)
    op_sum = activesum(op_cache)

    function commutatoroverlap(term, coeff)
        commutes(term, gate_mask) && return 0.0
        partner, sign = paulirotationproduct(gate_mask, term)
        return real(sign * coeff.op * getmergedcoeff(op_sum, partner).dual)
    end

    return mapreduce(commutatoroverlap, +, op_cache; init=0.0, thread)
end

# One sorted array is the case where a lookup per term hurts: `getmergedcoeff` binary-searches the
# whole sum, which is the most expensive single part of the backward sweep. The partner of a
# term is its XOR with the generator's mask, and XOR by a fixed mask keeps the order of two terms
# whenever they agree on the mask's own bits -- their first differing bit is then a bit the mask
# leaves alone. Terms carrying the same pattern on the mask therefore walk the sum forward to their
# partners, so one cursor per pattern replaces each search with a galloping step from where that
# pattern last matched. A mask with b set bits has 2^b patterns (four for a two-qubit rotation);
# patterns are picked up as they appear and a term that finds the cursor table full falls back to
# the plain search, so nothing depends on the number of patterns staying small.
function _generatorcommutatordot(::PropagationBase.ArrayStorage, gate_mask, op_cache; thread::Bool=true)
    merge!(op_cache; thread)

    op_terms, op_coeffs = activeterms(op_cache), activecoeffs(op_cache)
    @assert length(op_terms) == length(op_coeffs) "the operator sum's terms and coefficients disagree in length"
    isempty(op_terms) && return 0.0

    task_partitioner, n_tasks = PropagationBase._preparetasks(length(op_terms), thread)
    partials = Vector{Float64}(undef, n_tasks)

    function dot_chunk!(task_id)
        chunk = task_partitioner[task_id]
        partials[task_id] = _commutatordotrange(gate_mask, op_terms, op_coeffs, op_terms, op_coeffs, chunk.start, chunk.stop)
    end
    PropagationBase._eachtask(dot_chunk!, n_tasks)

    return sum(partials)
end

# A multi sum of arrays pairs each zone with the zone that holds the partners of its terms: the zone of a Pauli
# string is linear in it, so the generator's mask moves every term of a zone to the same zone.
function _generatorcommutatordot(::PropagationBase.MultiSumStorage{<:PropagationBase.ArrayStorage}, gate_mask, op_cache;
    thread::Bool=true)

    merge!(op_cache; thread)

    zone_caches = zonecaches(op_cache)
    partner_offset = zoneof(zonemap(op_cache), gate_mask) - 1
    partials = Vector{Float64}(undef, length(zone_caches))

    function dot_zone!(zone_id)
        zone_cache = zone_caches[zone_id]
        partner_cache = zone_caches[((zone_id - 1) ⊻ partner_offset) + 1]
        op_terms = activeterms(zone_cache)
        partials[zone_id] = _commutatordotrange(gate_mask, op_terms, activecoeffs(zone_cache),
            activeterms(partner_cache), activecoeffs(partner_cache), 1, length(op_terms))
    end
    PropagationBase._eachzone(dot_zone!, op_cache, thread)

    return sum(partials)
end

# how many patterns one task tracks at once; a two-qubit rotation needs four
const _MAX_DOT_CURSORS = 16

function _commutatordotrange(gate_mask::TT, op_terms, op_coeffs, partner_terms, partner_coeffs, lo::Int, hi::Int) where {TT}
    total = 0.0
    lo > hi && return total

    # everything the loop below reads unchecked, checked once here: the operator's chunk on both of
    # its arrays, and the coefficients of all the terms that the partners are searched among
    n_partners = length(partner_terms)
    checkbounds(op_terms, lo:hi)
    checkbounds(op_coeffs, lo:hi)
    checkbounds(partner_coeffs, 1:n_partners)

    patterns = Vector{TT}(undef, _MAX_DOT_CURSORS)  # the bits a group of terms carries on the mask
    cursors = fill(1, _MAX_DOT_CURSORS)             # where that group last found a partner
    n_cursors = 0

    @inbounds for ii in lo:hi
        term = op_terms[ii]
        commutes(term, gate_mask) && continue
        partner, sign = paulirotationproduct(gate_mask, term)

        pattern = term & gate_mask
        slot = 0
        for s in 1:n_cursors
            if patterns[s] == pattern
                slot = s
                break
            end
        end
        if slot == 0 && n_cursors < _MAX_DOT_CURSORS
            n_cursors += 1
            slot = n_cursors
            patterns[slot] = pattern
        end

        if slot == 0
            jj = searchsortedfirst(partner_terms, partner)
        else
            jj = _gallopingsearch(partner_terms, partner, cursors[slot], n_partners)
            cursors[slot] = jj
        end

        (jj <= n_partners && partner_terms[jj] == partner) || continue
        total += real(sign * op_coeffs[ii].op * partner_coeffs[jj].dual)
    end

    return total
end

# The first index at or after `from` whose term is not smaller than `key`, found by doubling a window
# out from `from` and bisecting the last one. Called with a cursor that only ever moves forward, this
# costs a handful of nearby reads instead of a binary search across the whole array.
@inline function _gallopingsearch(terms, key, from::Int, n::Int)
    # `n` is the length of `terms` and every index below stays in 1:n, so the reads are in bounds
    # whatever cursor the caller hands in
    from > n && return n + 1
    from = max(from, 1)
    @inbounds terms[from] >= key && return from

    step = 1
    lo = from + 1
    hi = min(from + step, n)
    @inbounds while hi < n && terms[hi] < key
        lo = hi + 1
        step <<= 1
        hi = min(from + step, n)
    end
    lo > hi && return n + 1

    return searchsortedfirst(terms, key, lo, hi, Base.Order.Forward)
end
