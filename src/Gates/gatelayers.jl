### gatelayers.jl
##
# A file for layers of gates of any kind: which gates of a circuit are gathered into a layer, when two gates commute, and
# the propagation of a layer whose gates each map a Pauli string to one Pauli string, such as Clifford gates and Pauli
# noise. Such a layer is applied in one pass over the sum, with all of its gates applied to a Pauli string in one visit
# from lookups built from the transfer maps of the gates, which saves a pass over the sum per gate. Layers of Pauli
# rotations are propagated in one pass as well, coset by coset, see rotationlayers.jl.
##
###

### Layers of a circuit

"""
    tolayers(circuit)
    tolayers(circuit, params)

Groups consecutive gates of `circuit` that commute with each other into `GateLayer`s.
Gates are not reordered, so only gates that follow each other in `circuit` can end up in the same layer.
Gates on different qubits commute, and whether two gates that share a qubit commute is decided by `commutes`.
Pauli rotations and other gates go into separate layers, because a layer of Pauli rotations and a layer of other gates are each propagated in their own way.
Also transforms the parameters of the circuit accordingly.
Returns the new equivalent layered circuit and its corresponding parameter vector.
"""
function tolayers(circuit, params)
    PropagationBase._checknumberofparams(circuit, params)

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

function tolayers(circuit)
    # the layers do not depend on the parameters
    layered_circuit, _ = tolayers(circuit, zeros(countparameters(circuit)))
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


### Commutation of gates

"""
    commutes(gate_a::Gate, gate_b::Gate)

Returns whether the two gates commute with each other, for every value of their parameters.
By default this is whether the gates act on different qubits, from `qinds`.
Two Pauli rotations commute if their generators commute, and two Clifford gates if they map every Pauli string on their qubits alike in either order.
"""
commutes(gate_a::Gate, gate_b::Gate) = isdisjoint(qinds(gate_a), qinds(gate_b))

# the generators are compared as integers of the type for the highest qubit that the rotations act on
function commutes(gate_a::PauliRotation, gate_b::PauliRotation)
    TT = getinttype(max(maximum(gate_a.qinds), maximum(gate_b.qinds)))
    return commutes(symboltoint(TT, gate_a.symbols, gate_a.qinds), symboltoint(TT, gate_b.symbols, gate_b.qinds))
end

# Clifford gates sharing a qubit commute if applying them in either order maps every Pauli string on their qubits alike
function commutes(gate_a::CliffordGate, gate_b::CliffordGate)
    if isdisjoint(gate_a.qinds, gate_b.qinds)
        return true
    end
    qinds = union(gate_a.qinds, gate_b.qinds)
    TT = getinttype(maximum(qinds))
    lookup_a = _preparecliffordmap(TT, gate_a)
    lookup_b = _preparecliffordmap(TT, gate_b)
    for paulis in 0:4^length(qinds)-1
        pstr = setpauli(zero(TT), TT(paulis), qinds)
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

# Whether the layer is applied with one visit of every Pauli string: the coefficients are numbers, and every gate maps a
# Pauli string to one Pauli string. A gate says so through `requiresmerging`, which is `false` for a gate that never
# creates duplicate Pauli strings, and the lookup built from its transfer map checks it.
function _hasmappath(layer::GateLayer, prop_cache::AbstractPropagationCache)
    return coefftype(prop_cache) <: Number && all(gate -> !requiresmerging(gate, prop_cache), layer.gates)
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

# the lookups of the gates of a layer in its order, and for every qubit the gates acting on it, the gate `i` as the bit
# `i - 1` of a tuple of words
struct _MapPlan{TT,W}
    lookups::Vector{_LocalLookup{TT}}
    qubit_gates::Vector{NTuple{W,UInt64}}
end

# Applies the layer in one pass over the sum. Every Pauli string is visited once, and the gates acting on it are applied
# one after the other in the order of the layer, truncating after each as the gates one after the other would. As for a
# Clifford gate, nothing is truncated when the gates keep the magnitudes of the coefficients and no weight limit is given.
function _applymap!(layer::GateLayer, prop_cache::AbstractPropagationCache, params;
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing,
    thread::Bool=true)

    plan = _preparemap(layer, params, prop_cache)
    if isinf(max_weight) && isnothing(customtruncfunc) && all(_keepsmagnitudes, plan.lookups)
        map!(_applyactinggates(plan, nothing), prop_cache; thread)
    else
        truncfunc = buildtruncfunc(prop_cache; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread)
        mapandtruncate!(_applyactinggates(plan, truncfunc), truncfunc, prop_cache; thread)
    end
    return prop_cache
end

_keepsmagnitudes(lookup::_LocalLookup) = all(factor -> abs(factor) == 1, lookup.factors)

# The function that applies the gates of the plan acting on a Pauli string to it and its coefficient, in the order of the
# layer. It stops at the first gate after which `truncfunc` truncates the pair, which it returns as it is then, and
# `mapandtruncate!` truncates it by the same test. With `truncfunc = nothing`, nothing is truncated.
function _applyactinggates(plan::_MapPlan{TT,W}, truncfunc::F) where {TT,W,F}
    function applyactinggates(pstr, coeff)
        acting = _actinggates(plan, pstr)
        for word_index in 1:W
            bits = acting[word_index]
            while bits != 0
                lookup = plan.lookups[64*(word_index-1)+trailing_zeros(bits)+1]
                pstr, coeff = _applylookup(lookup, pstr, coeff)
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

# the Pauli string and coefficient after the gate of the lookup
@inline function _applylookup(lookup::_LocalLookup, pstr, coeff)
    index = _localpaulis(pstr, lookup.qinds) + 1
    return pstr ⊻ lookup.changes[index], coeff * lookup.factors[index]
end

# the Paulis of `pstr` on `qinds` as one integer, as `getpauli(pstr, qinds)` reads them
@inline function _localpaulis(pstr, qinds::Vector{Int})
    paulis = 0
    for (index, qind) in enumerate(qinds)
        paulis |= Int(_getpaulibits(pstr, qind)) << (2 * (index - 1))
    end
    return paulis
end

# the plan of the layer on the Pauli strings of the cache, with the parameters of the parametrized gates and of the
# frozen ones
function _preparemap(layer::GateLayer, params, prop_cache::AbstractPropagationCache)
    TT = paulitype(prop_cache)
    lookups = _LocalLookup{TT}[]
    function pushlookup!(gate, param...)
        push!(lookups, _locallookup(gate, TT, param...))
        return
    end
    PropagationBase._foreachgate(pushlookup!, layer, params)

    nq = nqubits(prop_cache)
    n_words = cld(length(lookups), 64)
    qubit_gates = fill(ntuple(_ -> zero(UInt64), n_words), nq)
    for (gate_index, lookup) in enumerate(lookups)
        _check_qind_range(nq, lookup.qinds)
        for qind in lookup.qinds
            qubit_gates[qind] = PropagationBase._withposition(qubit_gates[qind], gate_index - 1)
        end
    end
    return _MapPlan(lookups, qubit_gates)
end

# The lookup of a gate from its transfer map on its qubits. The map must send every Pauli string to one Pauli string and
# leave the identity as it is, because the gate is not applied to the Pauli strings that are the identity on its qubits.
function _locallookup(gate, ::Type{TT}, param...) where {TT}
    # `qinds` is an integer for a gate with a `qind` field
    gate_qinds = Int[qinds(gate)...]
    transfer_map = _localtransfermap(gate, param...)
    if _ncolumns(transfer_map) != 4^length(gate_qinds)
        throw(ArgumentError("The transfer map of $gate has $(_ncolumns(transfer_map)) columns instead of 4^$(length(gate_qinds)) for its qubits."))
    end

    changes = TT[]
    factors = Float64[]
    for (index, column) in enumerate(transfer_map)
        if length(column) != 1
            throw(ArgumentError(
                "$gate maps a Pauli string to $(length(column)) Pauli strings, although `requiresmerging` says that it never " *
                "creates duplicate Pauli strings. A gate of a layer applied in one visit maps every Pauli string to one."
            ))
        end
        new_paulis, factor = only(column)
        if index == 1 && !(new_paulis == 0 && factor == 1)
            throw(ArgumentError("$gate changes the identity on its qubits, which a gate of a layer applied in one visit leaves as it is."))
        end
        push!(changes, setpauli(zero(TT), TT(Int(new_paulis) ⊻ (index - 1)), gate_qinds))
        push!(factors, factor)
    end
    return _LocalLookup(gate_qinds, changes, factors)
end

# The transfer map of a gate on its qubits, indexed by the Paulis on `qinds(gate)` as `getpauli(pstr, qinds)` reads them,
# with the parameter of a parametrized gate. A Clifford gate has its map in `clifford_map`, with one Pauli string and sign
# per column.
_localtransfermap(gate::CliffordGate) = TransferMap([[entry] for entry in clifford_map[gate.symbol]])

# Pauli noise keeps every Pauli string, and damps the coefficient of those that `isdamped` selects
function _localtransfermap(gate::PauliNoise, lambda)
    _check_noise_strength(PauliNoise, lambda)
    return TransferMap([[(UInt8(pauli), isdamped(gate, UInt8(pauli)) ? Float64(1 - lambda) : 1.0)] for pauli in 0:3])
end

# a frozen gate has the map of its gate at its parameter
_localtransfermap(gate::FrozenGate) = _localtransfermap(gate.gate, gate.parameter)

# any other gate gets its map from what it does to every Pauli string on its qubits
_localtransfermap(gate) = _transfermapbyapplying(gate, nothing)
_localtransfermap(gate, param) = _transfermapbyapplying(gate, [param])

# Propagates every Pauli string on the qubits of the gate through the gate, as `totransfermap` does for a circuit, and
# reads the Paulis on those qubits from the Pauli strings that come out, which must be the identity on the other qubits.
function _transfermapbyapplying(gate, params)
    gate_qinds = Int[qinds(gate)...]
    nq = maximum(gate_qinds)
    TT = getinttype(nq)
    # the type of the Pauli strings on the qubits of the gate alone
    LT = getinttype(length(gate_qinds))
    columns = Vector{Tuple{LT,Float64}}[]
    for paulis in 0:4^length(gate_qinds)-1
        pstr = setpauli(zero(TT), TT(paulis), gate_qinds)
        psum = propagate([gate], PauliString(nq, pstr, 1.0), params; min_abs_coeff=0.0, thread=false)
        column = Tuple{LT,Float64}[]
        for (new_pstr, coeff) in psum
            new_paulis = _localpaulis(new_pstr, gate_qinds)
            if setpauli(zero(TT), TT(new_paulis), gate_qinds) != new_pstr
                throw(ArgumentError("$gate acts on qubits other than its qubits $gate_qinds."))
            end
            push!(column, (LT(new_paulis), coeff))
        end
        push!(columns, column)
    end
    return TransferMap(columns)
end
