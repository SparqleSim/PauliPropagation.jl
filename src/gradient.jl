### gradient.jl
##
# Computes gradients of a Pauli propagation expectation value via backpropagation.
##
###


"""
    rewindgradient(circuit, psum::AbstractPauliSum, params, overlapfunc; thread=true, kwargs...)

Compute `overlapfunc(propagate(circuit, psum, params; kwargs...))` together with its gradient with
respect to `params`, in one paired forward and backward sweep costing O(length(circuit)) gate applications.
The only parametrized gates can be `PauliRotation`s and `GateLayer`s whose parametrized gates are `PauliRotation`s.
The gradient of a `GateLayer` is a vector of the derivatives by the parameters of its rotations, or a number if its parameter is a number.
All other gates must not be parametrized or frozen via `freeze(gate, param)` before inputting to `rewindgradient`.
Noise channels are not supported, frozen or not, because the backward sweep cannot undo them.
`overlapfunc` must be linear in the coefficients of the Pauli sum it is handed, e.g. any of
`overlapwithzero`, `overlapwithplus`, `overlapwithcomputational`, `overlapwithmaxmixed`, or
`overlapwithpaulisum`.
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
    # check that the only parameterized gates are PauliRotations and layers of them
    @assert all(_isdifferentiable, circuit) "All parameterized gates must be PauliRotations or GateLayers of them."
    @assert all(_isrewindable, circuit) "Noise channels are not supported because they cannot be undone."

    # forward sweep: ordinary Heisenberg propagation, exactly as in `propagate`.
    propagate!(circuit, forward_cache, params; thread, kwargs...)
    expec = overlapfunc(activesum(forward_cache))

    # the dual sum starts from the final operator, with overlapfunc applied to each of its Pauli strings individually,
    # and is carried beside the coefficients of the operator through the backward sweep
    cache = wrapdual(forward_cache, overlapfunc; thread)

    # backward sweep: undo gates in the circuit's own original order
    heisenberg = false
    undo_circuit, undo_params = _preparecircuit(circuit, params, heisenberg)
    state = _BackwardSweepState(cache, _zerogradient(undo_params), 0)
    PropagationBase._propagate!(_undostep!, undo_circuit, state, undo_params; thread, _unwrapkwargs(; kwargs...)...)

    return expec, state.grad
end


# The backward sweep undoes a gate by re-applying its Schrödinger-picture form,
# which only recovers the operator if the gate is invertible.
_isrewindable(gate) = true
_isrewindable(gate::ParametrizedNoiseChannel) = false
_isrewindable(gate::FrozenGate) = _isrewindable(gate.gate)
_isrewindable(layer::GateLayer) = all(_isrewindable, layer.gates)

# the gates whose parameters the backward sweep differentiates by
_isdifferentiable(gate::StaticGate) = true
_isdifferentiable(gate::ParametrizedGate) = false
_isdifferentiable(gate::PauliRotation) = true
_isdifferentiable(layer::GateLayer) = all(_isdifferentiable, layer.gates)


# a zero for the gradient by every parameter: a number, or a vector for a layer with one angle per rotation
_zerogradient(params) = [zero(float(param)) for param in params]

# a helper for customtruncfunc acting on the dual
function _unwrapkwargs(; customtruncfunc=nothing, kwargs...)
    if isnothing(customtruncfunc)
        return (; kwargs...)
    end
    opcustomtruncfunc(pstr, coeff) = customtruncfunc(pstr, coeff.coeff)
    return (; customtruncfunc=opcustomtruncfunc, kwargs...)
end


# State propagated through the backward sweep
# carries everything it needs to compute the gradient on the fly
mutable struct _BackwardSweepState{CT,GT}
    cache::CT
    grad::GT
    param_idx::Int
end

# the backward sweep is carried by this state instead of a cache, so the operator sum is what is
# counted and what keeps its zone workers up
PropagationBase._termcount(state::_BackwardSweepState) = length(state.cache)
PropagationBase._withworkers(f::F, state::_BackwardSweepState, thread::Bool) where {F} =
    PropagationBase._withworkers(f, state.cache, thread)

# Records the gradient component for PauliRotation
function _undostep!(gate::PauliRotation, state::_BackwardSweepState, theta; thread::Bool=true, kwargs...)
    gate_mask = symboltoint(paulitype(state.cache), gate.symbols, gate.qinds)

    state.param_idx += 1
    state.grad[state.param_idx] = _generatorcommutatordot(gate_mask, state.cache; thread)

    # the operator sum truncates normally, and the dual sum loses every Pauli string that it loses
    applymergetruncate!(gate, state.cache, theta; thread, kwargs...)

    return state
end

# just undoes the application of a StaticGate. No gradient recorded.
function _undostep!(gate::StaticGate, state::_BackwardSweepState; thread::Bool=true, kwargs...)
    applymergetruncate!(gate, state.cache; thread, kwargs...)
    return state
end

# Records the gradient of a GateLayer: the derivatives by the parameters of its rotations.
# Where `applymergetruncate!` would apply the layer class by class, one pass undoes it and adds up the gradients of its
# rotations, and otherwise its gates are undone one by one.
function _undostep!(layer::GateLayer, state::_BackwardSweepState, params; min_abs_coeff::Real=1e-10, max_weight::Real=Inf,
    max_freq::Real=Inf, max_sins::Real=Inf, min_rel_coeff=nothing, customtruncfunc=nothing, thread::Bool=true, kwargs...)

    if !_haslayerfastpath(layer, state.cache) || !isnothing(min_rel_coeff)
        gate_state = _BackwardSweepState(state.cache, zeros(countparameters(layer)), 0)
        function undogate!(gate, args...)
            _undostep!(gate, gate_state, args...;
                min_abs_coeff, max_weight, max_freq, max_sins, min_rel_coeff, customtruncfunc, thread, kwargs...)
            return
        end
        PropagationBase._foreachgate(undogate!, layer, params)
        rotation_grads = gate_state.grad
    else
        truncfunc = buildtruncfunc(state.cache; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread)
        rotation_grads = _undolayer!(layer, state.cache, params, truncfunc, min_abs_coeff; thread)
    end

    state.param_idx += 1
    state.grad[state.param_idx] = _layergradient(params, rotation_grads)
    return state
end

# the gradient of a layer from those of its parametrized rotations: a number for a number as its parameter
_layergradient(param::Number, rotation_grads) = only(rotation_grads)
_layergradient(params, rotation_grads) = rotation_grads


# The operator of `prop_cache` with its dual: every Pauli string keeps its coefficient and gets the overlap it has on its
# own as its coefficient in the dual sum. The dual coefficients are whatever `overlapfunc` returns for a single Pauli
# string -- real for every overlap the library ships. Nothing else in the backward sweep needs them complex: the dual is
# carried by the same real rotations as the operator, and the gradient pairs the two with the real signs of the rotations.
function wrapdual(prop_cache::AbstractPropagationCache, overlapfunc; thread::Bool=true)
    nq = nqubits(prop_cache)
    singletonoverlap(term) = overlapfunc(PauliString(nq, term, 1.0))

    dual_type = Base.promote_op(singletonoverlap, paulitype(prop_cache))
    if !isconcretetype(dual_type)
        dual_type = ComplexF64
    end
    coeff_type = _DualCoeff{coefftype(prop_cache),float(dual_type)}
    cache = PropagationCache(convertcoefftype(coeff_type, activesum(prop_cache)))
    withoverlap(term, coeff) = coeff_type(coeff.coeff, singletonoverlap(term))
    mapcoeffsbypair!(withoverlap, cache; thread)
    return cache
end

# The coefficient of a Pauli string in the operator sum and in the dual sum of the backward sweep. The gates act on both
# alike, and the truncations read only the coefficient of the operator, so the dual sum keeps exactly the Pauli strings
# that the operator sum keeps.
struct _DualCoeff{CT,DT}
    coeff::CT
    dual::DT
end

Base.:*(coeff::_DualCoeff, factor::Number) = _DualCoeff(coeff.coeff * factor, coeff.dual * factor)
Base.:*(factor::Number, coeff::_DualCoeff) = coeff * factor
Base.:+(coeff1::_DualCoeff, coeff2::_DualCoeff) = _DualCoeff(coeff1.coeff + coeff2.coeff, coeff1.dual + coeff2.dual)
Base.zero(::Type{_DualCoeff{CO,CD}}) where {CO,CD} = _DualCoeff(zero(CO), zero(CD))

# a coefficient of the operator alone, without a dual
Base.convert(::Type{_DualCoeff{CO,CD}}, coeff::Number) where {CO,CD} = _DualCoeff{CO,CD}(coeff, zero(CD))

# the truncations read the coefficient of the operator
truncatemincoeff(coeff::_DualCoeff, min_abs_coeff::Real) = truncatemincoeff(coeff.coeff, min_abs_coeff)
PropagationBase.tonumber(coeff::_DualCoeff) = coeff.coeff
PropagationBase.numcoefftype(::Type{_DualCoeff{CO,CD}}) where {CO,CD} = CO

# The classes of a layer of Pauli rotations are rotated with these coefficients as with numbers, and a Pauli string makes its partner
# only if the coefficient of the operator can reach the truncation threshold.
PropagationBase._rotatesinclasses(::Type{_DualCoeff{CO,CD}}) where {CO,CD} = CO <: Number && CD <: Number
Base.abs(coeff::_DualCoeff) = abs(coeff.coeff)


# Gradient contribution for one gate: real((i/2) * dual_sum(commutator(generator, op_sum))).
# For a term that anticommutes with the generator, (i/2) times the commutator is its product with
# the generator, with the sign that the rotation gives that product (`paulirotationproduct`). So this
# is a single pass over the operator's terms, each adding its coefficient times the dual coefficient
# of its parent term, with that sign.
function _generatorcommutatordot(gate_mask, cache; thread::Bool=true)
    return _generatorcommutatordot(StorageType(cache), gate_mask, cache; thread)
end

# The general case: we go through the Pauli strings of the operator and skip those that commute with the generator.
# Every other one has a parent term, its product with the generator, and contributes its own coefficient times the
# dual coefficient of that parent, with the sign of the product. We find the parent by looking it up in the merged sum.
function _generatorcommutatordot(::StorageType, gate_mask, cache; thread::Bool=true)
    # a lookup needs the sum merged
    merge!(cache; thread)
    op_sum = activesum(cache)

    function commutatoroverlap(term, coeff)
        if commutes(term, gate_mask)
            return 0.0
        end
        parent_term, sign = paulirotationproduct(gate_mask, term)
        return real(sign * coeff.coeff * getmergedcoeff(op_sum, parent_term).dual)
    end

    return mapreduce(commutatoroverlap, +, cache; init=0.0, thread)
end

# On one sorted array: the real part of Σ_P s_P c_P d_P' over the Pauli strings P of the operator that anticommute
# with the generator, with P' the parent term of P (its product with the generator, of sign s_P), c the coefficients
# of the operator and d those of the dual. Terms that share their bits under the gate mask form a group, and XOR with
# the mask keeps their order, so each group finds its parents by walking the array forward from the last one found.
function _generatorcommutatordot(::PropagationBase.ArrayStorage, gate_mask, cache; thread::Bool=true)
    merge!(cache; thread)

    terms, coeffs = activeterms(cache), activecoeffs(cache)
    @assert length(terms) == length(coeffs) "the operator sum's terms and coefficients disagree in length"
    if isempty(terms)
        return 0.0
    end

    task_partitioner, n_tasks = PropagationBase._preparetasks(length(terms), thread)
    task_sums = Vector{Float64}(undef, n_tasks)

    function dot_chunk!(task_id)
        chunk = task_partitioner[task_id]
        task_sums[task_id] = _commutatordotrange(gate_mask, terms, coeffs, terms, coeffs, chunk.start, chunk.stop)
    end
    PropagationBase._eachtask(dot_chunk!, n_tasks)

    return sum(task_sums)
end

# A multi sum of arrays pairs each zone with the zone that holds the parents of its terms: the zone of a Pauli
# string is linear in it, so the generator's mask moves every term of a zone to the same zone.
function _generatorcommutatordot(::PropagationBase.MultiSumStorage{<:PropagationBase.ArrayStorage}, gate_mask, cache;
    thread::Bool=true)

    merge!(cache; thread)

    zone_caches = zonecaches(cache)
    parent_zone_offset = zoneof(zonemap(cache), gate_mask) - 1
    task_sums = Vector{Float64}(undef, length(zone_caches))

    function dot_zone!(zone_id)
        zone_cache = zone_caches[zone_id]
        parent_cache = zone_caches[((zone_id - 1) ⊻ parent_zone_offset) + 1]
        terms = activeterms(zone_cache)
        task_sums[zone_id] = _commutatordotrange(gate_mask, terms, activecoeffs(zone_cache),
            activeterms(parent_cache), activecoeffs(parent_cache), 1, length(terms))
    end
    PropagationBase._eachzone(dot_zone!, cache, thread)

    return sum(task_sums)
end

# how many groups one task keeps a search start for; a two-qubit rotation has at most eight
const _MAX_DOT_GROUPS = 16

function _commutatordotrange(gate_mask::TT, terms, coeffs, parent_terms, parent_coeffs, lo::Int, hi::Int) where {TT}
    total = 0.0
    if lo > hi
        return total
    end

    # everything the loop below reads unchecked, checked once here: the operator's chunk on both of
    # its arrays, and the coefficients of all the terms that the parents are searched among
    n_parents = length(parent_terms)
    checkbounds(terms, lo:hi)
    checkbounds(coeffs, lo:hi)
    checkbounds(parent_coeffs, 1:n_parents)

    group_gate_bits = Vector{TT}(undef, _MAX_DOT_GROUPS)  # the bits the terms of a group carry under the mask
    group_search_starts = fill(1, _MAX_DOT_GROUPS)        # where that group last found a parent
    n_groups = 0

    @inbounds for ii in lo:hi
        term = terms[ii]
        if commutes(term, gate_mask)
            continue
        end
        parent_term, sign = paulirotationproduct(gate_mask, term)

        gate_bits = term & gate_mask
        group_idx = 0
        for g in 1:n_groups
            if group_gate_bits[g] == gate_bits
                group_idx = g
                break
            end
        end
        if group_idx == 0 && n_groups < _MAX_DOT_GROUPS
            n_groups += 1
            group_idx = n_groups
            group_gate_bits[group_idx] = gate_bits
        end

        if group_idx == 0
            jj = searchsortedfirst(parent_terms, parent_term)
        else
            jj = _gallopingsearch(parent_terms, parent_term, group_search_starts[group_idx], n_parents)
            group_search_starts[group_idx] = jj
        end

        if jj <= n_parents && parent_terms[jj] == parent_term
            total += real(sign * coeffs[ii].coeff * parent_coeffs[jj].dual)
        end
    end

    return total
end

# The first index at or after `from` whose term is not smaller than `key`, found by doubling a window
# out from `from` and bisecting the last one. Called with a search start that only ever moves forward, this
# costs a handful of nearby reads instead of a binary search across the whole array.
@inline function _gallopingsearch(terms, key, from::Int, n::Int)
    # `n` is the length of `terms` and every index below stays in 1:n, so the reads are in bounds
    # whatever search start the caller hands in
    if from > n
        return n + 1
    end
    from = max(from, 1)
    @inbounds if terms[from] >= key
        return from
    end

    step = 1
    lo = from + 1
    hi = min(from + step, n)
    @inbounds while hi < n && terms[hi] < key
        lo = hi + 1
        step <<= 1
        hi = min(from + step, n)
    end
    if lo > hi
        return n + 1
    end

    return searchsortedfirst(terms, key, lo, hi, Base.Order.Forward)
end


### Undoing a layer of Pauli rotations class by class

# Undoes the rotations of a layer as `_applylayer!` applies them, on the operator that carries the dual beside its
# coefficients, and returns the gradient of every parametrized rotation: what the pairs of Pauli strings that it mixes
# contribute, added up over the classes.
function _undolayer!(layer::GateLayer, cache, params, truncfunc::F, min_abs_coeff::Real; thread::Bool=true) where {F}
    rotations, angles = togates(layer, params)
    rotation_grads = zeros(length(rotations))
    TT = paulitype(cache)
    CT = coefftype(cache)
    workspace = PropagationBase._takeworkspace(TT, CT, PropagationBase.ClassScratch{TT,CT})
    try
        plan = _prepareclasses(rotations, angles, TT, nqubits(cache); min_abs_coeff)
        _undopass!(cache, plan, truncfunc, workspace, rotation_grads; thread)
    finally
        PropagationBase._putbackworkspace!(workspace)
    end
    is_parametrized = [gate isa ParametrizedGate for gate in layer.gates]
    return rotation_grads[is_parametrized]
end

# One pass as `_applypass!` makes it, in which every task adds up the gradients of the rotations in its own workspace,
# and those of all tasks are added to `rotation_grads` at the end.
function _undopass!(cache, plan, truncfunc::F, workspace, rotation_grads; thread::Bool=true) where {F}
    if length(cache) == 0
        return rotation_grads
    end
    sources = PropagationBase._passsources(StorageType(cache), cache, workspace, thread)
    n_tasks = length(sources)
    tasks = PropagationBase._ensurecount!(workspace.tasks, n_tasks)
    for task_id in 1:n_tasks
        fill!(resize!(tasks[task_id].scratch.gradient, length(rotation_grads)), 0.0)
    end

    classlabel(pstr) = PropagationBase._classlabel(plan, pstr)
    function rotateclasses!(sink, scratch, label_terms, label_coeffs)
        PropagationBase._rotateclasses!(_GradientSink(sink, scratch.gradient), scratch, plan, truncfunc, label_terms, label_coeffs)
        return sink
    end
    PropagationBase._applytolabels!(classlabel, rotateclasses!, cache, workspace, sources, thread)

    for task_id in 1:n_tasks
        rotation_grads .+= tasks[task_id].scratch.gradient
    end
    return rotation_grads
end

# The sink of a task in a gradient pass: it hands the Pauli strings on to the sink of the task, and adds what the two
# Pauli strings that a rotation mixes contribute to the gradient of that rotation.
struct _GradientSink{ST}
    sink::ST
    gradient::Vector{Float64}
end

@inline function PropagationBase._emit!(gradient_sink::_GradientSink, pstr, coeff)
    PropagationBase._emit!(gradient_sink.sink, pstr, coeff)
    return
end

# Each of the two contributes its coefficient in the operator times the dual coefficient of the other, with the sign that
# the rotation gives the other, as every Pauli string does in `_generatorcommutatordot`.
@inline function PropagationBase._addgradient!(gradient_sink::_GradientSink, rotation::Int, coeff::_DualCoeff, partner_coeff::_DualCoeff,
    sign_to_partner, sign_from_partner)

    gradient_sink.gradient[rotation] += real(sign_to_partner * coeff.coeff * partner_coeff.dual) +
                                        real(sign_from_partner * partner_coeff.coeff * coeff.dual)
    return
end
