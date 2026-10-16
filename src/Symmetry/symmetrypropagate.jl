### symmetrypropagate.jl
##
# Propagation with subgroup merging. `symmetrycircuit` cuts the circuit into layers of mutually
# commuting Pauli rotations and layers of other gates (`tolayers(circuit, thetas; reorder=true)`,
# Gates/gatelayers.jl), verifies them, and joins the plans of the layers (subgroups.jl) into one
# circuit of frozen gates and `SymmetryMerge` gates (groups.jl), which `symmetrypropagate` hands to
# `propagate`.
##
###


## The gates of a layer

# the gates of an entry of a layered circuit, each frozen with its parameter
_frozengates(entry::GateLayer) = entry.gates                                     # a layer without parameters
_frozengates(entry::FrozenGate{<:GateLayer}) = freeze(entry.gate.gates, entry.parameter)
_frozengates(entry) = [entry]

# The gates applied between merges, as blocks of frozen gates: a rotation layer, a rotation on
# its own, and other gates in the commuting layers `tolayers` found when they are checked. Without
# the check, consecutive other gates form one block merged only afterwards, the structure a caller
# asserting invariance can see.
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
function _checkrotationlayer(G::AbstractSiteSymmetry, pstrs, angles)
    all(commutes(pstrs[ii], pstrs[jj]) for ii in eachindex(pstrs) for jj in ii+1:lastindex(pstrs)) || throw(ArgumentError(
        "A layer of Pauli rotations given in the circuit does not commute. Build it with `PauliRotationLayer`, which checks, or pass `check=false`."))
    perm = _breakinggenerator(G, _angles(pstrs, angles))
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


## The plan of a circuit

"""
    symmetrycircuit(G::AbstractSiteSymmetry, circuit, thetas=nothing; heisenberg=true, order=:auto, subgroups=true, check=true)

The circuit that [`symmetrypropagate`](@ref) propagates: the gates of `circuit`, frozen with
`thetas`, in the order they are applied, with every gate after which the Pauli sum is merged
wrapped into a [`SymmetryMerge`](@ref) under the subgroup of `G` that merge uses.

The circuit is split into layers of mutually commuting Pauli rotations and layers of other gates
by [`tolayers`](@ref) with `reorder=true`, which gathers every gate into the earliest layer it
commutes into, so that e.g. the gates of a circuit written qubit by qubit form `G`-invariant
layers. Inside a layer of rotations (T gates among them, with their fixed angle) the gates are
applied in the order `order`, and after every gate
the sum is merged under the [`stabilizer`](@ref) of the gates applied so far, the subgroup of `G`
that maps them onto themselves, which is `G` again after the last gate. Other layers are applied
whole and merged under `G` after their last gate.

Like every circuit of the library, the plan is in the written order of the picture it is built
for and is to be propagated with the same `heisenberg`. With `heisenberg=true` (the default) the
gates are applied from the back, so the plan of a layer reads backwards; with `heisenberg=false`
it reads in the order the gates are applied.

# Keyword arguments
- `order=:auto`: the gate order inside a rotation layer: `:given` (as listed), `:lexicographic`
  (sorted by qubit indices, then Pauli symbols), `:orbits` (orbit by orbit of `G`) or an explicit
  permutation of the positions in the layer, which then applies to every rotation layer, all of
  which must have that length; with `heisenberg=true` the positions count from the back of the
  layer as written, since the circuit is reversed first. Reordering is allowed because the
  gates commute; without truncation it changes how large the sum gets in between, never the
  result. `:auto` is `:lexicographic` for a [`PermutationSymmetry`](@ref), where the staircase
  order `(1,2), (1,3), ...` leaves products of symmetric groups on contiguous blocks of sites,
  and `:orbits` otherwise: the gates of one `G`-orbit are applied together, so that `G` survives
  whenever an orbit is complete, walking each orbit along the cycles of one group element in the
  order that makes the prefixes of a cycle invariant under that element's subgroups. For the 8
  bonds of a ring this is `1, 5, 3, 7, 2, 6, 4, 8`, with `Z_2`, `Z_4`, `Z_2` and `Z_8` surviving.
- `subgroups=true`: merge inside rotation layers. `false` merges under `G` after every layer only,
  the plain symmetry merging.
- `check=true`: verify that rotation layers commute and are `G`-invariant, angles included, and
  that other gates commute and are invariant as a set. With `check=false`, `order=:auto` means
  `:given`, since reordering relies on the commutation check, and consecutive gates other than
  rotations form one block merged only afterwards, since the caller cannot see the layers found
  among them. Gates whose invariance cannot be verified, e.g. a `SWAP` written as three `CNOT`s,
  need `check=false`.

# Example
```julia
nq = 8
bonds = [PauliRotation([:Z, :Z], [i, mod1(i + 1, nq)]) for i in 1:nq]
plan = symmetrycircuit(TranslationSymmetry(nq), bonds, fill(0.1, nq); heisenberg=false)
[gate.group for gate in plan if gate isa SymmetryMerge]
>>> 4-element Vector{PauliPropagation.AbstractSiteSymmetry}:
 SiteSymmetry(8, order 2)
 SiteSymmetry(8, order 4)
 SiteSymmetry(8, order 2)
 TranslationSymmetry(8)
```

# Extended help
- To control the layering, put `GateLayer`s (e.g. [`PauliRotationLayer`](@ref)) into the
  circuit: they are kept as they are, and no gate is moved past them. A given layer mixing
  rotations with other gates is applied whole and checked as a set of whole gates.
- The plan is frozen: it takes no parameters, since the merges depend on the angles.
- A circuit that already holds `SymmetryMerge` gates, also inside a `GateLayer`, is refused;
  propagate it as it is.
"""
function symmetrycircuit(G::AbstractSiteSymmetry, circuit, thetas=nothing;
    heisenberg::Bool=true, order=:auto, subgroups::Bool=true, check::Bool=true)

    circuit, thetas = PropagationBase._promotecircandparams(circuit, thetas)
    PropagationBase._checknumberofparams(circuit, thetas)
    if !check && order === :auto
        order = :given      # reordering relies on the commutation check
    end
    # the plan is built in the order the gates are applied; the Heisenberg picture reverses the
    # circuit, and the plan is reversed back so that `propagate` reverses it once more
    heisenberg || return _plan(G, circuit, thetas; order, subgroups, check)
    return toheisenberg(_plan(G, toheisenberg(circuit, thetas)...; order, subgroups, check))
end

# the plans of the layers of a circuit in propagation order, joined; identical layers share a plan
function _plan(G::AbstractSiteSymmetry, circuit, thetas; order, subgroups::Bool, check::Bool)
    layered = freeze(tolayers(circuit, thetas; reorder=true)...)
    TT = getinttype(nqubits(G))
    # identical layers share a plan, gate objects included: a frozen rotation is fixed by its Pauli
    # string and angle, so the first layer's gates serve every layer with the same strings and angles
    plans = Dict{Tuple{Vector{TT},Vector{Float64}},Vector{Gate}}()
    plan = Gate[]
    for gates in _gateblocks(layered, check)
        any(gate -> gate isa SymmetryMerge, gates) && throw(ArgumentError(
            "The circuit already holds `SymmetryMerge` gates; propagate it as it is."))
        if all(_isrotation, gates)
            pstrs, angles = _rotationdata(TT, gates)
            append!(plan, get!(plans, (pstrs, angles)) do
                check && _checkrotationlayer(G, pstrs, angles)
                subgroups ? _rotationplan(G, gates, pstrs, angles, order) : _wholelayerplan(G, gates)
            end)
        else
            check && _checkotherlayer(G, gates)
            append!(plan, _wholelayerplan(G, gates))
        end
    end
    return plan
end


## Propagation

"""
    symmetrypropagate(G::AbstractSiteSymmetry, circuit, psum, thetas=nothing; kwargs...)
    symmetrypropagate!(G::AbstractSiteSymmetry, circuit, psum, thetas=nothing; kwargs...)

Propagate `psum` (a Pauli sum, a `PauliString` or a propagation cache) through `circuit` like
[`propagate`](@ref), merging Pauli strings related by the symmetry group `G` as early as the
circuit allows: `propagate` of the [`symmetrycircuit`](@ref) of `G` and `circuit`.

Without truncation the result is the merge under `G` of what `propagate` returns, provided every
layer and the state the propagated sum is finally contracted with are `G`-invariant; `check=true`
verifies the layers. As with any symmetry merge, the returned sum is to be contracted with
`G`-invariant states only.

The keywords `order`, `subgroups` and `check` are those of `symmetrycircuit`, `heisenberg` is as
in `propagate`, and all others, e.g. `min_abs_coeff` or `thread`, are passed on to the gates.

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
- With truncation the result differs from `propagate`, as for any reordering of the gates;
  coefficients are truncated after each merge rather than before.
- `@countpaulis` holds two entries per merge, the sizes before and after it, so `@peakpaulis` is
  the size just before a merge where that is the largest.
- `max_freq` and `max_sins` wrap the coefficients as `propagate` does; `symmetrypropagate!`
  requires them wrapped, as `propagate!` does.
- As with `propagate`, use a `VectorPauliSum` (or a cache built on one) for large simulations.
  `Performance.propagate(symmetrycircuit(G, circuit, thetas), psum)` applies the runs of gates
  between merges as layers where that pays, e.g. whole layers with `subgroups=false`.
"""
function symmetrypropagate!(G::AbstractSiteSymmetry, circuit, thing, thetas=nothing;
    heisenberg::Bool=true, order=:auto, subgroups::Bool=true, check::Bool=true, kwargs...)
    _checknqubits(G, thing)
    plan = symmetrycircuit(G, circuit, thetas; heisenberg, order, subgroups, check)
    return propagate!(plan, thing; heisenberg, kwargs...)
end

function symmetrypropagate(G::AbstractSiteSymmetry, circuit, thing, thetas=nothing;
    heisenberg::Bool=true, order=:auto, subgroups::Bool=true, check::Bool=true, kwargs...)
    _checknqubits(G, thing)
    plan = symmetrycircuit(G, circuit, thetas; heisenberg, order, subgroups, check)
    # `propagate` has no out-of-place method for a cache
    thing isa AbstractPauliPropagationCache && return propagate!(plan, deepcopy(thing); heisenberg, kwargs...)
    return propagate(plan, thing; heisenberg, kwargs...)
end

@doc (@doc symmetrypropagate!) symmetrypropagate
