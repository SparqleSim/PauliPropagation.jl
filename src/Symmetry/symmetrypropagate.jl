### symmetrypropagate.jl
##
# Propagation with subgroup merging. The circuit is cut into layers of mutually commuting Pauli
# rotations and layers of other gates (`tolayers(circuit, thetas; reorder=true)`,
# Gates/gatelayers.jl). Every rotation layer gets a `SubgroupSchedule` (subgroups.jl); the driver
# applies each gate and merges under the scheduled group. A layer applied as a whole has the
# trivial group after every gate but the last.
##
###


## The gates of a layer

# the gates of an entry of a layered circuit, each frozen with its parameter
_frozengates(entry::GateLayer) = entry.gates                                     # a layer without parameters
_frozengates(entry::FrozenGate{<:GateLayer}) = freeze(entry.gate.gates, entry.parameter)
_frozengates(entry) = [entry]

# The gates the driver applies between merges, as blocks of frozen gates: a rotation layer, a
# rotation on its own, and other gates in the commuting layers `tolayers` found when they are
# checked. Without the check, consecutive other gates form one block merged only afterwards,
# the structure a caller asserting invariance can see.
function _gateblocks(layered, check::Bool)
    blocks = Vector{Gate}[]
    for entry in layered
        gates = _frozengates(entry)
        isempty(gates) && continue          # an empty layer given in the circuit
        if !check && !all(_isrotation, gates) && !isempty(blocks) && !all(_isrotation, blocks[end])
            append!(blocks[end], gates)
        else
            push!(blocks, collect(Gate, gates))
        end
    end
    return blocks
end


## The checks behind `check=true`

# a rotation layer must commute and be G-invariant for the merges inside and after it to be exact
function _checkrotationlayer(G::AbstractSiteSymmetry, layer::RotationLayer)
    iscommuting(layer) || throw(ArgumentError(
        "A layer of Pauli rotations given in the circuit does not commute. Build it with `PauliRotationLayer`, which checks, or pass `check=false`."))
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

# Clifford gates that do not care about the order of their qubits, as the Schrödinger picture transposes them
const _SYMMETRIC_CLIFFORDS = (:CZ, :SWAP, :ZZpihalf, :CZ_transpose, :SWAP_transpose, :ZZpihalf_transpose)

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

# A layer of other gates is applied as given and merged afterwards, so its gates must commute, so
# that their order does not matter, and their multiset must be G-invariant. `tolayers` gathers only
# commuting gates; a layer given in the circuit is verified here.
function _checkotherlayer(G::AbstractSiteSymmetry, gates)
    nq = nqubits(G)
    for gate in gates
        _gatequbits(gate) === nothing && throw(ArgumentError(
            "Cannot check whether a layer with a $(typeof(gate)) is invariant under the symmetry; pass `check=false`."))
    end
    for ii in eachindex(gates), jj in ii+1:lastindex(gates)
        commutes(gates[ii], gates[jj]) || throw(ArgumentError(
            "The gates $(gates[ii]) and $(gates[jj]) of a layer do not commute, so the order of the gates matters and " *
            "the invariance of the layer cannot be verified. Pass `check=false` if the layer is known to be invariant under the symmetry."))
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

The circuit is split into layers of mutually commuting Pauli rotations and layers of other gates
by [`tolayers`](@ref) with `reorder=true`, which gathers every gate into the earliest layer it
commutes into, so that e.g. the gates of a circuit written qubit by qubit form `G`-invariant
layers. A rotation layer is applied one gate at a time in the order of its [`subgroupschedule`](@ref),
and after every gate the Pauli sum is merged under the stabilizer of the gates applied so far,
the subgroup of `G` that maps them onto themselves. Other layers are applied whole. After every
layer the sum is merged under `G`.

Without truncation the result is the merge under `G` of what `propagate` returns, provided every
layer and the state the propagated sum is finally contracted with are `G`-invariant. With
`check=true` the layers are verified: rotations must commute and be invariant, angles included;
other gates must commute and be invariant as a set. As with any symmetry merge, the returned sum
is to be contracted with `G`-invariant states only.

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
- To control the layering, put `GateLayer`s (e.g. [`PauliRotationLayer`](@ref)) into the
  circuit: they are kept as they are, and no gate is moved past them. A given layer mixing
  rotations with other gates is applied whole and checked as a set of whole gates.
- `heisenberg=true` as in `propagate`.
- With `check=false`, `order=:auto` means `:given`, since reordering relies on the commutation
  check, and consecutive gates other than rotations are applied as one block merged only
  afterwards, since the caller cannot see the layers found among them. An explicit permutation
  as `order` applies to every rotation layer, which must then all have that length. Gates whose
  invariance cannot be verified, e.g. a `SWAP` written as three `CNOT`s, need `check=false`.
- With truncation the result differs from `propagate`, as for any reordering of the gates;
  coefficients are truncated after each merge rather than before. `max_freq` and `max_sins`
  need a sum whose coefficients are already wrapped in a `PauliFreqTracker`.
- As with `propagate`, use a `VectorPauliSum` (or a cache built on one) for large simulations.
"""
function symmetrypropagate!(G::AbstractSiteSymmetry, circuit, prop_cache::AbstractPauliPropagationCache, thetas=nothing;
    order=:auto, subgroups::Bool=true, check::Bool=true, heisenberg::Bool=true, record=nothing, kwargs...)

    _checknqubits(G, prop_cache)
    if !check && order === :auto
        order = :given      # reordering relies on the commutation check
    end

    # as in `propagate` (a single gate becomes a circuit, the Heisenberg picture reverses circuit
    # and parameters), then the gates are gathered into layers and every gate carries its parameter
    circuit, thetas = _preparecircuit(circuit, thetas, heisenberg)
    layered = freeze(tolayers(circuit, thetas; reorder=true)...)
    TT = getinttype(nqubits(G))     # for the schedules; the merges use the cache's own Pauli type
    schedules = Dict{RotationLayer{TT},SubgroupSchedule}()   # identical layers share a schedule
    thread = get(kwargs, :thread, true)

    PropagationBase._withworkers(prop_cache, thread) do
        for (layer_index, gates) in enumerate(_gateblocks(layered, check))
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
