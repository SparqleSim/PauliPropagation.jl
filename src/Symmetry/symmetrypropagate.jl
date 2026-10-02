### symmetrypropagate.jl
##
# Propagation with subgroup merging. The circuit is cut into layers of mutually commuting Pauli
# rotations and runs of other gates (`_commutinglayers`, Circuits/utils.jl). Every rotation layer
# gets a `SubgroupSchedule` (subgroups.jl); the driver applies each gate and merges under the
# scheduled group. A layer applied as a whole has the trivial group after every gate but the last.
##
###


## Layers given by the caller

# The layers as index vectors into the prepared circuit. Caller-given layers are contiguous ranges
# of the circuit as written, so that no gate is moved past one it may not commute with; the
# Heisenberg picture reverses the circuit, and with it the ranges.
function _layerindices(prepared_circuit, layers, heisenberg::Bool)
    layers === :auto && return _commutinglayers(prepared_circuit)
    m = length(prepared_circuit)
    ranges = [first(layer):last(layer) for layer in layers]
    valid = all(!isempty(r) && collect(r) == collect(l) for (r, l) in zip(ranges, layers)) &&
            reduce(vcat, ranges; init=Int[]) == 1:m
    valid || throw(ArgumentError("`layers` must be non-empty contiguous ranges that partition 1:$(m) in order, got $(layers)."))
    heisenberg || return [collect(r) for r in ranges]
    return [collect(m-last(r)+1:m-first(r)+1) for r in reverse(ranges)]
end


## The checks behind `check=true`

# a rotation layer must commute and be G-invariant for the merges inside and after it to be exact
function _checkrotationlayer(G::AbstractSiteSymmetry, layer::RotationLayer)
    iscommuting(layer) || throw(ArgumentError(
        "A caller-given layer of Pauli rotations does not commute. Split it, use `layers=:auto`, or pass `check=false`."))
    perm = _breakinggenerator(G, _angles(layer))
    perm === nothing || throw(ArgumentError(
        "A layer of Pauli rotations is not invariant under $(G): the site permutation $(perm) maps one of its " *
        "gates to a gate that is missing or has a different angle. Pass `check=false` to skip this test."))
    return nothing
end

# the qubits a gate acts on, in the order the gate lists them, or `nothing` if unknown
function _gatequbits(gate)
    gate isa FrozenGate && return _gatequbits(gate.gate)
    hasproperty(gate, :qinds) && return Tuple(gate.qinds)
    hasproperty(gate, :qind) && return (gate.qind,)
    return nothing
end

# Clifford gates that do not care about the order of their qubits
const _SYMMETRIC_CLIFFORDS = (:CZ, :SWAP, :ZZpihalf)

# A gate as (type, qubits, other fields, parameter) with its qubits replaced by their images under
# `perm`. A Pauli rotation is keyed by its permuted Pauli string instead. Other fields are
# compared as they are, so gates caching qubit-dependent data only match their own copies.
function _gatekey(gate, perm)
    inner = gate isa FrozenGate ? gate.gate : gate
    param = gate isa FrozenGate ? gate.parameter : nothing
    inner isa PauliRotation && return (PauliRotation, _permute(_paulistringof(getinttype(length(perm)), inner), perm), param)
    qubits = map(q -> perm[q], _gatequbits(gate))
    inner isa CliffordGate && inner.symbol in _SYMMETRIC_CLIFFORDS && (qubits = Tuple(sort(collect(qubits))))
    others = Tuple(getfield(inner, name) for name in fieldnames(typeof(inner)) if name ∉ (:qinds, :qind))
    return (typeof(inner), qubits, others, param)
end

# A layer of other gates is applied as given, so it is only known to be G-invariant if its gates
# act on disjoint qubits (then their order does not matter) and their multiset is G-invariant.
function _checkotherlayer(G::AbstractSiteSymmetry, gates)
    nq = nqubits(G)
    occupied = falses(nq)
    for gate in gates
        qubits = _gatequbits(gate)
        qubits === nothing && throw(ArgumentError(
            "Cannot check whether a layer with a $(typeof(gate)) is invariant under the symmetry; pass `check=false`."))
        for q in qubits
            occupied[q] && throw(ArgumentError(
                "Gates of a layer other than Pauli rotations share qubit $(q), so the order of the gates matters and the " *
                "invariance of the layer cannot be verified ($(typeof(gate)) at qubits $(qubits)). " *
                "Pass `check=false` if the layer is known to be invariant under the symmetry."))
            occupied[q] = true
        end
    end
    identity_perm = collect(1:nq)
    counts = Dict{Any,Int}()
    for gate in gates
        key = _gatekey(gate, identity_perm)
        counts[key] = get(counts, key, 0) + 1
    end
    for perm in generators(G), gate in gates
        get(counts, _gatekey(gate, perm), 0) == counts[_gatekey(gate, identity_perm)] || throw(ArgumentError(
            "A layer is not invariant under $(G): the site permutation $(perm) maps the gate $(gate) to a gate " *
            "that is missing or differs. Pass `check=false` to skip this test."))
    end
    return nothing
end


## The driver

# the keyword arguments that `buildtruncfunc` understands (`_TRUNCATION_KEYS`, Propagation/generics.jl),
# out of all the ones passed to the gates
_truncationkwargs(kwargs) = (; (key => value for (key, value) in pairs(kwargs) if key in _TRUNCATION_KEYS)...)

# Merge the cache under `group`, then truncate: `mapterms!` rewrites every Pauli string to its
# canonical form, `mergeandtruncate!` adds up the strings that became equal and drops what the
# truncation criteria reject.
function _mergetruncateunder!(group::AbstractSiteSymmetry, prop_cache::AbstractPauliPropagationCache; thread::Bool=true, kwargs...)
    mapterms!(canonicalform(group, paulitype(prop_cache)), prop_cache; thread)
    mergeandtruncate!(buildtruncfunc(prop_cache; _truncationkwargs(kwargs)..., thread), prop_cache; thread)
    return
end

# Apply one gate and merge under `group` unless it is trivial. When a merge follows, the
# coefficient criteria are applied after it rather than to the single-string amplitudes the gate
# produced (the explicit keywords after `kwargs...` override the caller's values; weight and
# frequency truncation still happen at the gate). Returns the numbers of strings before and after.
function _step!(gate, prop_cache::AbstractPauliPropagationCache, group::AbstractSiteSymmetry; kwargs...)
    if istrivial(group)
        applymergetruncate!(gate, prop_cache; kwargs...)
        before = after = length(prop_cache)
    else
        applymergetruncate!(gate, prop_cache; kwargs..., min_abs_coeff=0.0, min_rel_coeff=nothing, customtruncfunc=nothing)
        before = length(prop_cache)
        _mergetruncateunder!(group, prop_cache; kwargs...)
        after = length(prop_cache)
    end
    PropagationBase._recordsize!(prop_cache)
    return before, after
end

"""
    symmetrypropagate(G::AbstractSiteSymmetry, circuit, psum, thetas=nothing; kwargs...)
    symmetrypropagate!(G::AbstractSiteSymmetry, circuit, psum, thetas=nothing; kwargs...)
    symmetrypropagate!(G::AbstractSiteSymmetry, circuit, prop_cache, thetas=nothing; kwargs...)

Propagate `psum` through `circuit` like [`propagate`](@ref), merging Pauli strings related by
the symmetry group `G` as early as the circuit allows.

The circuit is split into layers of mutually commuting Pauli rotations and layers of other gates.
A rotation layer is applied one gate at a time in the order of its [`subgroupschedule`](@ref),
and after every gate the Pauli sum is merged under the stabilizer of the gates applied so far,
the subgroup of `G` that maps them onto themselves. Other layers are applied whole. After every
layer the sum is merged under `G`.

Without truncation the result is the merge under `G` of what `propagate` returns, provided every
layer and the state the propagated sum is finally contracted with are `G`-invariant. With
`check=true` the layers are verified: rotations must commute and be invariant, angles included;
other gates must act on disjoint qubits and be invariant as a set. As with any symmetry merge, the
returned sum is to be contracted with `G`-invariant states only.

# Keyword arguments
- `subgroups=true`: merge inside rotation layers. `false` applies every layer whole and merges
  under `G` only afterwards, the plain symmetry merging.
- `order=:auto`: the gate order inside a rotation layer, see [`subgroupschedule`](@ref).
- `check=true`: verify the layers as described above.
- `record=nothing`: a `Vector` to which a named tuple `(layer, step, group, before, after)` is
  pushed after every merge, with the numbers of Pauli strings before and after it.
- all other keyword arguments, e.g. `min_abs_coeff` or `thread`, are passed on to the gates.

# Example
```julia
nq = 6
topology = [(i, j) for i in 1:nq for j in i+1:nq]
circuit = heisenbergtrottercircuit(nq, 4; topology)
thetas = fill(0.05, countparameters(circuit))
psum = PauliSum(nq)
for i in 1:nq
    add!(psum, :X, i)
end
merged = symmetrypropagate(PermutationSymmetry(nq), circuit, psum, thetas)
overlapwithplus(merged) ≈ overlapwithplus(propagate(circuit, psum, thetas))
>>> true
```

# Extended help
- `layers=:auto` finds the layers as maximal sets of mutually commuting rotations, moving a
  rotation only past rotations it commutes with and never past other gates. Alternatively pass
  a vector of contiguous index ranges that partitions the circuit as written, in order.
- `heisenberg=true` as in `propagate`.
- With `check=false`, `order=:auto` means `:given`, since reordering relies on the commutation
  check. An explicit permutation as `order` applies to every rotation layer, which must then all
  have that length. Layers that cannot be verified, e.g. a ring of `CNOT`s, need `check=false`.
- With truncation the result differs from `propagate`, as for any reordering of the gates;
  coefficients are truncated after each merge rather than before. `max_freq` and `max_sins`
  need a sum whose coefficients are already wrapped in a `PauliFreqTracker`.
- As with `propagate`, use a `VectorPauliSum` (or a cache built on one) for large simulations.
"""
function symmetrypropagate!(G::AbstractSiteSymmetry, circuit, prop_cache::AbstractPauliPropagationCache, thetas=nothing;
    layers=:auto, order=:auto, subgroups::Bool=true, check::Bool=true, heisenberg::Bool=true, record=nothing, kwargs...)

    _checknqubits(G, prop_cache)
    if !check && order === :auto
        order = :given      # reordering relies on the commutation check
    end

    # as in `propagate` (a single gate becomes a circuit, the Heisenberg picture reverses circuit
    # and parameters), then every gate carries its own parameter
    circuit, thetas = _preparecircuit(circuit, thetas, heisenberg)
    circuit = freeze(circuit, thetas)
    TT = getinttype(nqubits(G))     # for the schedules; the merges use the cache's own Pauli type
    schedules = Dict{RotationLayer{TT},SubgroupSchedule}()   # identical layers share a schedule
    thread = get(kwargs, :thread, true)

    PropagationBase._withworkers(prop_cache, thread) do
        for (layer_index, indices) in enumerate(_layerindices(circuit, layers, heisenberg))
            gates = circuit[indices]
            if all(_isrotation, gates)
                layer = RotationLayer(TT, gates)
                schedule = get!(schedules, layer) do
                    check && _checkrotationlayer(G, layer)
                    subgroups ? subgroupschedule(G, layer; order) : _wholelayerschedule(G, length(layer))
                end
            else
                check && _checkotherlayer(G, gates)
                schedule = _wholelayerschedule(G, length(gates))
            end
            for (step, position) in enumerate(schedule.order)
                group = schedule.groups[step]
                before, after = _step!(gates[position], prop_cache, group; kwargs...)
                if record !== nothing && !istrivial(group)
                    push!(record, (layer=layer_index, step=step, group=group, before=before, after=after))
                end
            end
        end
    end
    return prop_cache
end

function symmetrypropagate!(G::AbstractSiteSymmetry, circuit, psum::AbstractPauliSum, thetas=nothing; kwargs...)
    prop_cache = PropagationCache(psum)
    symmetrypropagate!(G, circuit, prop_cache, thetas; kwargs...)
    return extractsum!(prop_cache, psum)
end

symmetrypropagate(G::AbstractSiteSymmetry, circuit, thing::Union{AbstractPauliSum,AbstractPauliPropagationCache}, thetas=nothing; kwargs...) =
    symmetrypropagate!(G, circuit, deepcopy(thing), thetas; kwargs...)
symmetrypropagate(G::AbstractSiteSymmetry, circuit, pstr::PauliString, thetas=nothing; kwargs...) =
    symmetrypropagate!(G, circuit, PauliSum(pstr), thetas; kwargs...)

@doc (@doc symmetrypropagate!) symmetrypropagate
