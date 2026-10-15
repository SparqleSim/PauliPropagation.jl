### gatelayers.jl
##
# A file for layers of gates of any kind: which gates of a circuit are gathered into a layer, when two gates commute, and
# the propagation of a layer whose gates each map a Pauli string to one Pauli string, such as Clifford gates and Pauli
# noise. Such a layer is applied in one pass over the sum, with all of its gates applied to a Pauli string in one visit
# from lookups built from the transfer maps of the gates, which saves a pass over the sum per gate. A layer of any other
# gates on different qubits is applied in one pass as well, class by class, with the classes found from the transfer maps
# of its gates and every gate applied to a class through its own methods. Layers of Pauli rotations are propagated class
# by class too, see rotationlayers.jl.
##
###

### Layers of a circuit

"""
    tolayers(circuit; reorder=false)
    tolayers(circuit, params; reorder=false)

Groups consecutive gates of `circuit` that commute with each other into `GateLayer`s; a gate that shares no layer stays a gate.
By default gates are not reordered, so only gates that follow each other in `circuit` can end up in the same layer.
With `reorder=true`, a gate is first moved back to the earliest run of gates of its kind that it reaches passing only gates it commutes with, which regroups e.g. the RX, RZ, RX that `hardwareefficientcircuit` applies qubit by qubit into three layers over all qubits.
The layers applied in order give the same operator as the circuit either way; with truncation, a reordered circuit can give a different result than the circuit as written.
Keep the default to speed up propagation without changing the result of the circuit as written, as `Performance.propagate` does; use `reorder=true` where the layers must be as large as the gates allow, as [`symmetrypropagate`](@ref) does.
Layers already in the circuit, frozen or not, stay as they are, and no gate is moved past them.
Gates on different qubits commute, and whether two gates that share a qubit commute is decided by `commutes`.
Pauli rotations and other gates go into separate layers, because a layer of Pauli rotations and a layer of other gates are each propagated in their own way.
Also transforms the parameters of the circuit accordingly.
Returns the new equivalent layered circuit and its corresponding parameter vector.
"""
function tolayers(circuit, params; reorder::Bool=false)
    PropagationBase._checknumberofparams(circuit, params)
    if reorder
        circuit, params = _reordercommuting(circuit, params)
    end

    layered_circuit = Gate[]
    layered_params = []
    # the gates gathered for the next layer, and the parameters of those parametrized
    layer = Gate[]
    layer_params = []
    # the gathered gates on each qubit, so that a new gate is only compared with the gates it shares a qubit with
    gates_on_qubit = Dict{Int,Vector{Gate}}()
    param_index = 0
    for gate in circuit
        if isparametrized(gate)
            param_index += 1
        end

        if gate isa GateLayer
            # a layer is not nested into another layer
            _pushlayer!(layered_circuit, layered_params, layer, layer_params, gates_on_qubit)
            push!(layered_circuit, gate)
            if isparametrized(gate)
                push!(layered_params, params[param_index])
            end
        else
            if !_joinslayer(gate, layer, gates_on_qubit)
                _pushlayer!(layered_circuit, layered_params, layer, layer_params, gates_on_qubit)
            end
            push!(layer, gate)
            # add the gate to the gates on each of its qubits
            for qind in qinds(gate)
                push!(get!(Vector{Gate}, gates_on_qubit, qind), gate)
            end
            if isparametrized(gate)
                push!(layer_params, params[param_index])
            end
        end
    end
    _pushlayer!(layered_circuit, layered_params, layer, layer_params, gates_on_qubit)
    return layered_circuit, layered_params
end

function tolayers(circuit; reorder::Bool=false)
    # the layers do not depend on the parameters
    layered_circuit, _ = tolayers(circuit, zeros(countparameters(circuit)); reorder)
    return layered_circuit
end

# Pushes the gathered gates with their parameters, a single gate as it is and several as a layer, which takes a parameter
# if any of its gates does. Then nothing is gathered.
function _pushlayer!(layered_circuit, layered_params, layer, layer_params, gates_on_qubit)
    if length(layer) == 1
        push!(layered_circuit, only(layer))
        append!(layered_params, layer_params)
    elseif length(layer) > 1
        push!(layered_circuit, GateLayer(layer; guaranteed_commutes=true))
        if !isempty(layer_params)
            push!(layered_params, [param for param in layer_params])
        end
    end
    empty!(layer)
    empty!(layer_params)
    empty!(gates_on_qubit)
    return
end

# Whether a gate joins the gathered gates: it is of the same kind, a Pauli rotation or not, and commutes with the gathered
# gates on its qubits. The kinds stay apart because a layer of both kinds would apply its gates one after the other.
function _joinslayer(gate, layer, gates_on_qubit)
    if isempty(layer)
        return true
    end
    # the gathered gates are either all Pauli rotations or none, so the first gate tells which
    if _isrotation(gate) != _isrotation(first(layer))
        return false
    end
    # gates on other qubits always commute with it, so only the gates on its own qubits are compared
    for qind in qinds(gate)
        if !all(layer_gate -> commutes(gate, layer_gate), get(gates_on_qubit, qind, Gate[]))
            return false
        end
    end
    return true
end


### Reordering a circuit so that commuting gates follow each other

# The gates of `circuit` reordered so that gates that commute with each other follow each other: every gate is moved
# back to the earliest run of gates of its kind that it reaches passing only gates it commutes with, so that the runs
# are the layers `tolayers` then gathers. Layers given in the circuit, frozen or not, stay in place and no gate is
# moved past them. The parameters follow their gates.
function _reordercommuting(circuit, params)
    order = Int[]
    # the open runs since the last layer given in the circuit, as the positions of their gates, and for each run the
    # positions of its gates on each qubit, so that a gate is only compared with the gates it shares a qubit with
    runs = Vector{Int}[]
    runs_on_qubit = Dict{Int,Vector{Int}}[]
    for (position, gate) in enumerate(circuit)
        if _islayer(gate)
            foreach(run -> append!(order, run), runs)
            empty!(runs)
            empty!(runs_on_qubit)
            push!(order, position)
            continue
        end
        run = _joinedrun(gate, circuit, runs, runs_on_qubit)
        if run == 0
            push!(runs, Int[])
            push!(runs_on_qubit, Dict{Int,Vector{Int}}())
            run = length(runs)
        end
        push!(runs[run], position)
        for qind in qinds(gate)
            push!(get!(Vector{Int}, runs_on_qubit[run], qind), position)
        end
    end
    foreach(run -> append!(order, run), runs)
    return circuit[order], _reorderedparams(circuit, params, order)
end

# a layer given in the circuit, frozen with its parameters or not
_islayer(gate) = gate isa GateLayer || (gate isa FrozenGate && gate.gate isa GateLayer)

# The run that `gate` joins, or 0 if it starts a new one: the earliest run of its kind, a Pauli rotation or not, that
# it reaches walking back over runs whose gates on its qubits all commute with it.
function _joinedrun(gate, circuit, runs, runs_on_qubit)
    joined = 0
    for run in reverse(eachindex(runs))
        for qind in qinds(gate)
            all(position -> commutes(gate, circuit[position]), get(runs_on_qubit[run], qind, Int[])) || return joined
        end
        if _isrotation(gate) == _isrotation(circuit[first(runs[run])])
            joined = run
        end
    end
    return joined
end

# the parameters of the parametrized gates of `circuit` in the order `order` of the gates
function _reorderedparams(circuit, params, order)
    param_index = cumsum(isparametrized(gate) for gate in circuit)
    return [params[param_index[position]] for position in order if isparametrized(circuit[position])]
end


### Commutation of gates

"""
    commutes(gate_a::Gate, gate_b::Gate)

Returns whether the two gates commute with each other, for every value of their parameters.
By default this is whether the gates act on different qubits, from `qinds`.
Two Pauli rotations, a T gate among them, commute if their generators commute, and two Clifford gates if they map every Pauli string on their qubits alike in either order.
"""
function commutes(gate_a::Gate, gate_b::Gate)
    # gates that are Pauli rotations, such as a T gate, commute as their rotations do
    if _isrotation(gate_a) && _isrotation(gate_b)
        return commutes(_rotationof(gate_a), _rotationof(gate_b))
    end
    return isdisjoint(qinds(gate_a), qinds(gate_b))
end

# the generators commute if they have different Paulis, neither the identity, on an even number of shared qubits
function commutes(gate_a::PauliRotation, gate_b::PauliRotation)
    n_anticommuting = 0
    for (symbol_a, qind_a) in zip(gate_a.symbols, gate_a.qinds), (symbol_b, qind_b) in zip(gate_b.symbols, gate_b.qinds)
        if qind_a == qind_b && symbol_a != symbol_b && symbol_a != :I && symbol_b != :I
            n_anticommuting += 1
        end
    end
    return iseven(n_anticommuting)
end

# Clifford gates sharing a qubit commute if applying them in either order maps every Pauli string on their qubits alike,
# which they do if they map X and Z on each of the qubits alike, since a Clifford map is fixed by what it makes of those.
# The gates are compared on their qubits alone, numbered from one, so that the Pauli strings are small integers.
function commutes(gate_a::CliffordGate, gate_b::CliffordGate)
    if isdisjoint(gate_a.qinds, gate_b.qinds)
        return true
    end
    qinds = union(gate_a.qinds, gate_b.qinds)
    gate_a = CliffordGate(gate_a.symbol, indexin(gate_a.qinds, qinds))
    gate_b = CliffordGate(gate_b.symbol, indexin(gate_b.qinds, qinds))
    lookup_a = _preparecliffordmap(UInt64, gate_a)
    lookup_b = _preparecliffordmap(UInt64, gate_b)
    for qind in eachindex(qinds), pauli in (:X, :Z)
        pstr = setpauli(zero(UInt64), pauli, qind)
        after_a = only(apply(gate_a, pstr, 1, lookup_a))
        after_b = only(apply(gate_b, pstr, 1, lookup_b))
        if only(apply(gate_b, after_a..., lookup_b)) != only(apply(gate_a, after_b..., lookup_a))
            return false
        end
    end
    return true
end

# a frozen gate commutes like the gate it holds
commutes(gate_a::FrozenGate, gate_b::Gate) = commutes(gate_a.gate, gate_b)
commutes(gate_a::Gate, gate_b::FrozenGate) = commutes(gate_a, gate_b.gate)
commutes(gate_a::FrozenGate, gate_b::FrozenGate) = commutes(gate_a.gate, gate_b.gate)


### Propagation of a layer in one visit per Pauli string

# Whether the layer is applied with one visit of every Pauli string: the coefficients are of a type that a layer is
# applied to in one pass, and every gate maps a Pauli string to one Pauli string. A gate says so through
# `requiresmerging`, which is `false` for a gate that never creates duplicate Pauli strings, and the lookup built from
# its transfer map checks it.
function _hasmappath(layer::GateLayer, prop_cache::AbstractPropagationCache)
    return PropagationBase._appliesinclasses(coefftype(prop_cache)) && all(gate -> !requiresmerging(gate, prop_cache), layer.gates)
end

# What a gate does to a Pauli string, for every combination of the Paulis on its qubits: the bits it flips, and the factor
# it gives the coefficient. The combination of Paulis on `qinds`, as `getpauli(pstr, qinds)` reads it, plus one indexes both.
# A Pauli string that is the identity on the qubits is left as it is, so a gate is only applied to the Pauli strings that
# are not the identity on one of its qubits.
struct _LocalLookup{TT}
    qinds::Vector{Int}
    changes::Vector{TT}
    factors::Vector{Float64}
end

# The gates of a layer in its order: the bit shifts of their qubits and 4^K entries each in `changes` and `factors`,
# `K` the largest number of qubits of a gate, and for every qubit the gates acting on it, the gate `i` as the bit
# `i - 1` of a tuple of words.
struct _MapPlan{TT,W,K}
    shifts::Vector{NTuple{K,Int}}
    changes::Vector{TT}
    factors::Vector{Float64}
    qubit_gates::Vector{NTuple{W,UInt64}}

    function _MapPlan(shifts::Vector{NTuple{K,Int}}, changes::Vector{TT}, factors::Vector{Float64},
        qubit_gates::Vector{NTuple{W,UInt64}}) where {TT,W,K}
        plan = new{TT,W,K}(shifts, changes, factors, qubit_gates)
        _checkplan(plan)
        return plan
    end
end

# Applies the layer in one pass over the sum. Every Pauli string is visited once, and the gates acting on it are applied
# one after the other in the order of the layer, truncating after each as the gates one after the other would. As for a
# Clifford gate, nothing is truncated when the gates keep the magnitudes of the coefficients and no weight limit is given.
function _applymap!(layer::GateLayer, prop_cache::AbstractPropagationCache, params;
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, min_rel_coeff=nothing,
    customtruncfunc=nothing, thread::Bool=true, kwargs...)

    plan = _preparemap(layer, params, prop_cache; kwargs...)
    if isinf(max_weight) && isnothing(customtruncfunc) && _keepsmagnitudes(plan)
        map!(_applyactinggates(plan, nothing), prop_cache; thread)
    else
        truncfunc = buildtruncfunc(prop_cache; min_abs_coeff, max_weight, max_freq, max_sins, min_rel_coeff, customtruncfunc, thread)
        mapandtruncate!(_applyactinggates(plan, truncfunc), truncfunc, prop_cache; thread)
    end
    return prop_cache
end

_keepsmagnitudes(plan::_MapPlan) = all(factor -> abs(factor) == 1, plan.factors)

# The function that applies the gates of the plan acting on a Pauli string to it and its coefficient, in the order of the
# layer. It stops at the first gate after which `truncfunc` truncates the pair, which it returns as it is then, and
# `mapandtruncate!` truncates it by the same test. With `truncfunc = nothing`, nothing is truncated.
function _applyactinggates(plan::_MapPlan{TT,W}, truncfunc::F) where {TT,W,F}
    _checkplan(plan)
    function applyactinggates(pstr, coeff)
        acting = _actinggates(plan, pstr)
        for word_index in 1:W
            bits = acting[word_index]
            while bits != 0
                gate_index = 64 * (word_index - 1) + trailing_zeros(bits) + 1
                pstr, coeff = _applygate(plan, gate_index, pstr, coeff)
                if !isnothing(truncfunc) && truncfunc(pstr, coeff)
                    return pstr, coeff
                end
                bits &= bits - 1
            end
        end
        return pstr, coeff
    end
    return applyactinggates
end

# the gates of the plan acting on the Pauli string: those on a qubit where it is not the identity
@inline function _actinggates(plan::_MapPlan{TT,W}, pstr) where {TT,W}
    acting = ntuple(_ -> zero(UInt64), Val(W))
    for word_index in 0:_nwords(pstr)-1
        word = PropagationBase._wordat(pstr, 64 * word_index)
        # the lower bit of every qubit of the word that is not the identity
        support = (word | (word >> 1)) & 0x5555555555555555
        while support != 0
            qind = 32 * word_index + (trailing_zeros(support) >> 1) + 1
            acting = map(|, acting, plan.qubit_gates[qind])
            support &= support - 1
        end
    end
    return acting
end

# the number of 64-bit words of a Pauli string
_nwords(pstr) = cld(8 * sizeof(pstr), 64)

# the Pauli string and coefficient after the gate `gate_index` of the plan
@inline function _applygate(plan::_MapPlan{TT,W,K}, gate_index::Int, pstr, coeff) where {TT,W,K}
    # `_checkplan` ensures that the plan has the gate `gate_index` and 4^K entries for it
    @inbounds index = 4^K * (gate_index - 1) + _gatherpaulis(pstr, plan.shifts[gate_index]) + 1
    @inbounds return pstr ⊻ plan.changes[index], coeff * plan.factors[index]
end

# the plan of the layer on the Pauli strings of the cache, with the parameters of the parametrized gates and of the
# frozen ones
function _preparemap(layer::GateLayer, params, prop_cache::AbstractPropagationCache; kwargs...)
    TT = paulitype(prop_cache)
    lookups = _LocalLookup{TT}[]
    function pushlookup!(gate, param...)
        push!(lookups, _locallookup(gate, TT, param...; kwargs...))
        return
    end
    PropagationBase._foreachgate(pushlookup!, layer, params)
    return _MapPlan(lookups, nqubits(prop_cache))
end

# the plan from the lookups of the gates on Pauli strings on `nq` qubits, with the gates acting on every qubit
function _MapPlan(lookups::Vector{_LocalLookup{TT}}, nq::Int) where {TT}
    n_words = cld(length(lookups), 64)
    qubit_gates = fill(ntuple(_ -> zero(UInt64), n_words), nq)
    for (gate_index, lookup) in enumerate(lookups)
        _check_qind_range(nq, lookup.qinds)
        for qind in lookup.qinds
            qubit_gates[qind] = PropagationBase._withposition(qubit_gates[qind], gate_index - 1)
        end
    end
    K = maximum(lookup -> length(lookup.qinds), lookups; init=0)
    return _MapPlan(lookups, qubit_gates, Val(K))
end

# A gate on fewer than K qubits reads its last qubit again in place of the missing ones, and repeats its entries so that
# the Paulis read there do not change which entry it takes. `K` comes as a type, so that the tuples of shifts have a
# length known to the compiler.
function _MapPlan(lookups::Vector{_LocalLookup{TT}}, qubit_gates, ::Val{K}) where {TT,K}
    shifts = NTuple{K,Int}[]
    changes = TT[]
    factors = Float64[]
    for lookup in lookups
        n_qinds = length(lookup.qinds)
        push!(shifts, ntuple(ii -> _bitshiftfromsiteindex(lookup.qinds[min(ii, n_qinds)]), Val(K)))
        for paulis in 0:4^K-1
            index = paulis % 4^n_qinds + 1
            push!(changes, lookup.changes[index])
            push!(factors, lookup.factors[index])
        end
    end
    return _MapPlan(shifts, changes, factors, qubit_gates)
end

# `_applygate` reads the entries of the plan without bounds checks, so every gate needs its 4^K entries, and every gate
# acting on a qubit needs to be a gate of the plan. The plan is checked when it is built and again before it is applied.
function _checkplan(plan::_MapPlan{TT,W,K}) where {TT,W,K}
    n_entries = 4^K * length(plan.shifts)
    if length(plan.changes) != n_entries || length(plan.factors) != n_entries
        throw(ArgumentError(
            "The plan of a layer has $(length(plan.changes)) changes and $(length(plan.factors)) factors " *
            "instead of $n_entries for $(length(plan.shifts)) gates on up to $K qubits."
        ))
    end
    for (qind, gates) in enumerate(plan.qubit_gates)
        for (word_index, word) in enumerate(gates)
            # the number of the last gate of this word acting on the qubit, counted from one
            last_gate = 64 * word_index - leading_zeros(word)
            if word != 0 && last_gate > length(plan.shifts)
                throw(ArgumentError(
                    "The plan of a layer has $(length(plan.shifts)) gates, but more act on the qubit $qind."
                ))
            end
        end
    end
    return
end

# The lookup of a gate from its transfer map on its qubits. The map must send every Pauli string to one Pauli string and
# leave the identity as it is, because the gate is not applied to the Pauli strings that are the identity on its qubits.
function _locallookup(gate, ::Type{TT}, param...; kwargs...) where {TT}
    # `qinds` is an integer for a gate with a `qind` field
    gate_qinds = Int[qinds(gate)...]
    transfer_map = _localtransfermap(gate, param...; kwargs...)
    if _ncolumns(transfer_map) != 4^length(gate_qinds)
        throw(ArgumentError(lazy"The transfer map of $gate has $(_ncolumns(transfer_map)) columns instead of 4^$(length(gate_qinds)) for its qubits."))
    end

    changes = TT[]
    factors = Float64[]
    for (index, column) in enumerate(transfer_map)
        if length(column) != 1
            throw(ArgumentError(LazyString(
                gate, " maps a Pauli string to ", length(column), " Pauli strings, although `requiresmerging` says that it never ",
                "creates duplicate Pauli strings. A gate of a layer applied in one visit maps every Pauli string to one."
            )))
        end
        new_paulis, factor = only(column)
        if index == 1 && !(new_paulis == 0 && factor == 1)
            throw(ArgumentError(lazy"$gate changes the identity on its qubits, which a gate of a layer applied in one visit leaves as it is."))
        end
        push!(changes, setpauli(zero(TT), TT(Int(new_paulis) ⊻ (index - 1)), gate_qinds))
        push!(factors, factor)
    end
    return _LocalLookup(gate_qinds, changes, factors)
end


### Propagation of a layer of gates on different qubits, class by class

# Whether the layer is applied class by class: the sum can be applied to in one pass with its coefficients, the gates
# act on different qubits, and every gate can be applied to a part of the sum.
function _hasclasspath(layer::GateLayer, prop_cache::AbstractPropagationCache)
    return PropagationBase._canapplylayer(prop_cache) && _ondifferentqubits(layer) &&
           all(PropagationBase._appliesbyclass, layer.gates)
end

# whether the gates of the layer act on different qubits, as a layer built without `guaranteed_commutes` checks
_ondifferentqubits(layer::GateLayer) = !layer.guaranteed_commutes || isnothing(PropagationBase._sharedqubit(layer.gates))

# An imaginary-time rotation normalizes the sum by the coefficient of the identity, so it is not applied to a part of it
PropagationBase._appliesbyclass(::ImaginaryPauliRotation) = false
PropagationBase._appliesbyclass(gate::FrozenGate) = PropagationBase._appliesbyclass(gate.gate)

# The classes of a layer. The class of a Pauli string is the Pauli string with the Paulis on the qubits of every gate
# replaced by the first of their local class, which `representatives` does as a layer applied in one visit. The gates
# that map the identity on their qubits to more than the identity act on every class, the gate `i` as the bit `i - 1` of
# `acting_everywhere`.
struct _ClassPlan{TT,W,K}
    representatives::_MapPlan{TT,W,K}
    acting_everywhere::NTuple{W,UInt64}
end

# Applies the layer class by class. The Pauli strings are grouped by the representative of their class, which every
# gate maps into itself, and the gates acting on a class are applied to it one after the other in the order of the
# layer, each with `applymergetruncate!` and the keywords as `propagate!` applies it, so that the class is merged and
# truncated after every gate. The Pauli strings of a class that no gate acts on are truncated alike.
function _applyclasses!(layer::GateLayer, prop_cache::AbstractPropagationCache, params;
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, min_rel_coeff=nothing,
    customtruncfunc=nothing, thread::Bool=true, kwargs...)

    truncfunc = buildtruncfunc(prop_cache; min_abs_coeff, max_weight, max_freq, max_sins, min_rel_coeff, customtruncfunc, thread)
    plan = _prepareclasses(layer, params, prop_cache; kwargs...)
    _applyclasses!(plan, layer, prop_cache, params, truncfunc;
        min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
    return prop_cache
end

function _applyclasses!(plan::_ClassPlan{TT,W}, layer::GateLayer, prop_cache::AbstractPropagationCache, params,
    truncfunc::F; thread::Bool=true, kwargs...) where {TT,W,F}

    torepresentative = _applyactinggates(plan.representatives, nothing)
    representativeof(pstr) = first(torepresentative(pstr, 1.0))
    param_indices = _parameterindices(layer)

    function applygates!(class_cache, class_pstrs)
        # The gates act on the qubits where a Pauli string of the class is not the identity, since they leave the
        # identity on their qubits as it is, except those acting everywhere.
        support = zero(TT)
        for pstr in class_pstrs
            support |= pstr
        end
        acting = map(|, _actinggates(plan.representatives, support), plan.acting_everywhere)
        if all(iszero, acting)
            mergeandtruncate!(truncfunc, class_cache; thread=false)
        end
        for word_index in 1:W
            bits = acting[word_index]
            while bits != 0
                gate_index = 64 * (word_index - 1) + trailing_zeros(bits) + 1
                _applygatetoclass!(layer.gates[gate_index], class_cache, params, param_indices[gate_index]; kwargs...)
                bits &= bits - 1
            end
        end
        return class_cache
    end

    scratch_type = PauliClassScratch{TT,coefftype(prop_cache)}
    PropagationBase._applygatesbyclass!(applygates!, representativeof, prop_cache, scratch_type; thread)
    return prop_cache
end

# one gate of the layer on the Pauli strings of a class, with its parameter if it takes one
function _applygatetoclass!(gate, class_cache, params, param_index::Int; kwargs...)
    if isparametrized(gate)
        param = PropagationBase._gateparameter(params, param_index)
        applymergetruncate!(gate, class_cache, param; thread=false, kwargs...)
    else
        applymergetruncate!(gate, class_cache; thread=false, kwargs...)
    end
    return class_cache
end

# the position of the parameter of every gate in the parameter of the layer, 0 for a gate without one
function _parameterindices(layer::GateLayer)
    param_indices = zeros(Int, length(layer.gates))
    param_index = 0
    for (gate_index, gate) in enumerate(layer.gates)
        if isparametrized(gate)
            param_index += 1
            param_indices[gate_index] = param_index
        end
    end
    return param_indices
end

# the classes of the layer on the Pauli strings of the cache, from the transfer maps of its gates at their parameters
function _prepareclasses(layer::GateLayer, params, prop_cache::AbstractPropagationCache; kwargs...)
    TT = paulitype(prop_cache)
    lookups = _LocalLookup{TT}[]
    moving_identity = Int[]
    function pushgate!(gate, param...)
        transfer_map = _localtransfermap(gate, param...; kwargs...)
        push!(lookups, _classlookup(transfer_map, Int[qinds(gate)...], TT))
        # the first column is what the gate makes of the identity on its qubits
        if first(transfer_map) != [(0, 1.0)]
            push!(moving_identity, length(lookups))
        end
        return
    end
    PropagationBase._foreachgate(pushgate!, layer, params)

    representatives = _MapPlan(lookups, nqubits(prop_cache))
    acting_everywhere = ntuple(_ -> zero(UInt64), cld(length(lookups), 64))
    for gate_index in moving_identity
        acting_everywhere = PropagationBase._withposition(acting_everywhere, gate_index - 1)
    end
    return _ClassPlan(representatives, acting_everywhere)
end

# the lookup that replaces the Paulis on the qubits of a gate by the first of their local class
function _classlookup(transfer_map, gate_qinds::Vector{Int}, ::Type{TT}) where {TT}
    first_paulis = _localclasses(transfer_map)
    n_paulis = length(first_paulis)
    changes = [setpauli(zero(TT), TT(paulis ⊻ first_paulis[paulis+1]), gate_qinds) for paulis in 0:n_paulis-1]
    return _LocalLookup(gate_qinds, changes, ones(length(changes)))
end

# For every combination of Paulis on the qubits of a gate, the first of its local class: the smallest combination that
# the transfer map connects with it, directly or through others. Every entry of a column joins the class of the column
# with the class of the entry, at whichever of the two first combinations is smaller. An entry with a zero factor joins
# them as well, since the gate still makes its Pauli string.
function _localclasses(transfer_map)
    first_paulis = collect(0:_ncolumns(transfer_map)-1)
    function firstof(paulis)
        while first_paulis[paulis+1] != paulis
            paulis = first_paulis[paulis+1]
        end
        return paulis
    end
    for (index, column) in enumerate(transfer_map)
        for (new_paulis, _) in column
            column_first = firstof(index - 1)
            entry_first = firstof(Int(new_paulis))
            first_paulis[max(column_first, entry_first)+1] = min(column_first, entry_first)
        end
    end
    return [firstof(paulis) for paulis in 0:length(first_paulis)-1]
end


### Transfer maps of gates on their qubits

# The transfer map of a gate on its qubits, indexed by the Paulis on `qinds(gate)` as `getpauli(pstr, qinds)` reads them,
# with the parameter of a parametrized gate. The keywords of the propagation other than the truncations reach a gate
# that is applied to get its map, and a gate with a map of its own ignores them. A Clifford gate has its map in
# `clifford_map`, with one Pauli string and sign per column.
_localtransfermap(gate::CliffordGate; kwargs...) = TransferMap([[entry] for entry in clifford_map[gate.symbol]])

# Pauli noise keeps every Pauli string, and damps the coefficient of those that `isdamped` selects
function _localtransfermap(gate::PauliNoise, lambda; kwargs...)
    _check_noise_strength(PauliNoise, lambda)
    return TransferMap([[(UInt8(pauli), isdamped(gate, UInt8(pauli)) ? Float64(1 - lambda) : 1.0)] for pauli in 0:3])
end

# a frozen gate has the map of its gate at its parameter
_localtransfermap(gate::FrozenGate; kwargs...) = _localtransfermap(gate.gate, gate.parameter; kwargs...)

# any other gate gets its map from what it does to every Pauli string on its qubits
_localtransfermap(gate; kwargs...) = _transfermapbyapplying(gate; kwargs...)
_localtransfermap(gate, param; kwargs...) = _transfermapbyapplying(gate, param; kwargs...)

# Applies the gate to every Pauli string on its qubits, as a layer applies it to a class, and reads the Paulis on those
# qubits from the Pauli strings that come out, which must be the identity on the other qubits.
function _transfermapbyapplying(gate, param...; kwargs...)
    gate_qinds = Int[qinds(gate)...]
    nq = maximum(gate_qinds)
    # one type for the gates on the first 64 qubits, so that applying a gate is not compiled for each of their types
    TT = getinttype(max(nq, 64))
    # the type of the Pauli strings on the qubits of the gate alone
    LT = getinttype(length(gate_qinds))
    # the cache into which a layer loads its classes
    cache = PauliClassScratch{TT,Float64}().cache
    columns = Vector{Tuple{LT,Float64}}[]
    for paulis in 0:4^length(gate_qinds)-1
        pstr = setpauli(zero(TT), TT(paulis), gate_qinds)
        empty!(cache)
        add!(mainsum(cache), pstr, 1.0)
        applymergetruncate!(gate, cache, param...; min_abs_coeff=0.0, thread=false, kwargs...)
        column = Tuple{LT,Float64}[]
        for (new_pstr, coeff) in mainsum(cache)
            new_paulis = getpauli(new_pstr, gate_qinds)
            if setpauli(zero(TT), new_paulis, gate_qinds) != new_pstr
                throw(ArgumentError(lazy"$gate acts on qubits other than its qubits $gate_qinds."))
            end
            push!(column, (LT(new_paulis), coeff))
        end
        push!(columns, column)
    end
    return TransferMap(columns)
end
