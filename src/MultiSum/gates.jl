### Propagation

"""
    propagate(circuit, mpsum::MultiVectorPauliSum, thetas=nothing; kwargs...)
    propagate!(circuit, mpsum::MultiVectorPauliSum, thetas=nothing; kwargs...)

Propagate a `MultiVectorPauliSum` through `circuit`. Threading comes from the zones running in
parallel, so `thread` here decides whether the zone loops are threaded, not whether any single zone
operation is.
"""
PropagationBase.propagate(circuit, mpsum::MultiVectorPauliSum, thetas=nothing; kwargs...) =
    propagate!(circuit, deepcopy(mpsum), thetas; kwargs...)

function PropagationBase.propagate!(circuit, mpsum::MultiVectorPauliSum, thetas=nothing; heisenberg::Bool=true, kwargs...)
    circuit, thetas = PauliPropagation._preparecircuit(circuit, thetas, heisenberg)
    return PB._propagate!(applygate!, circuit, mpsum, thetas; kwargs...)
end

function _eachzone(f::F, n_zones::Int, thread::Bool) where {F}
    if thread
        @threads for zone_id in 1:n_zones
            f(zone_id)
        end
    else
        for zone_id in 1:n_zones
            f(zone_id)
        end
    end
    return
end


### Pauli rotations

"""
    applygate!(gate::PauliRotation, mpsum::MultiVectorPauliSum, theta; kwargs...)

Every zone branches its own terms and parks the products in its outbox, then every zone merges the
outbox addressed to it. Zone assignment is linear in the Pauli string, so the gate permutes the zones
and every zone has exactly one outbox to pick up.

Set `n_zones` to twice the thread count: the permutation pairs the zones off, and a task that owns
both halves of a pair needs no barrier and hands the outbox it just wrote to a merge on the same core.
"""
function applygate!(gate::PauliRotation, mpsum::MultiVectorPauliSum, theta;
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf,
    customtruncfunc=nothing, thread::Bool=true, kwargs...)

    PauliPropagation._check_qind_range(mpsum.nqubits, gate.qinds)

    xor_mask = symboltoint(paulitype(mpsum), gate.symbols, gate.qinds)
    gate_mask = PF._gatemask(xor_mask, paulis(mainsum(mpsum.zones[1])))
    zone_shift = _zonebits(xor_mask, mpsum.zonemasks)

    kept_val, new_val = cos(theta), sin(theta)
    truncfunc(pstr, coeff) = PF._coefftruncfunc(pstr, coeff; min_abs_coeff, max_freq, max_sins, customtruncfunc)

    # a pair needs nothing from outside itself, so one task per pair needs no barrier at all --
    # worth it only while there are still enough pairs to keep every thread busy
    pairs = _zonepairs(nzones(mpsum), zone_shift)
    if !thread || length(pairs) >= nthreads()
        _eachzone(length(pairs), thread) do ii
            source, owner = pairs[ii]
            _branchintooutbox!(mpsum, source, owner, gate_mask, kept_val, new_val, max_weight)
            source != owner && _branchintooutbox!(mpsum, owner, source, gate_mask, kept_val, new_val, max_weight)

            # both zones must finish branching before either takes delivery
            _mergefromoutbox!(mpsum, owner, source, xor_mask, truncfunc)
            source != owner && _mergefromoutbox!(mpsum, source, owner, xor_mask, truncfunc)
        end
        return mpsum
    end

    _eachzone(nzones(mpsum), thread) do source
        _branchintooutbox!(mpsum, source, _partner(source, zone_shift), gate_mask, kept_val, new_val, max_weight)
    end

    _eachzone(nzones(mpsum), thread) do owner
        _mergefromoutbox!(mpsum, owner, _partner(owner, zone_shift), xor_mask, truncfunc)
    end

    return mpsum
end

@inline _partner(zone_id::Int, zone_shift::Int) = ((zone_id - 1) ⊻ zone_shift) + 1

# The gate permutes the zones, so they pair off and a pair needs nothing from outside itself.
# One task per pair removes both barriers; a shift of zero leaves every zone on its own.
function _zonepairs(n_zones::Int, zone_shift::Int)
    zone_shift == 0 && return [(zone_id, zone_id) for zone_id in 1:n_zones]
    return [(zone_id, _partner(zone_id, zone_shift)) for zone_id in 1:n_zones
            if zone_id - 1 < zone_id - 1 ⊻ zone_shift]
end

function _branchintooutbox!(mpsum::MultiVectorPauliSum, source::Int, owner::Int, gate_mask, kept_val, new_val, max_weight)
    zone = mpsum.zones[source]
    n_terms = activesize(zone)

    fill!(view(mpsum.counts, :, source), 0)
    mpsum.outbox_ordered[source] = sortedprefix(mainsum(zone)) == n_terms
    n_terms == 0 && return

    outbox = mpsum.outboxes[source]
    length(outbox) < n_terms && resize!(outbox, n_terms + n_terms >> 1)

    mpsum.counts[owner, source] = _branchwrite!(
        paulis(outbox), coefficients(outbox), paulis(mainsum(zone)), coefficients(mainsum(zone)),
        n_terms, gate_mask, kept_val, new_val, max_weight)

    return
end

# A branching term keeps its Pauli string, so it stays in this zone and only has its coefficient
# scaled. Its product goes to the outbox unless it is too heavy, which merging cannot undo.
@inline function _branchwrite!(out_terms, out_coeffs, zone_terms, zone_coeffs, n_terms::Int,
    gate_mask, kept_val, new_val, max_weight)

    pos = 1

    GC.@preserve zone_terms begin
        bytes = PF._bytesof(zone_terms, gate_mask)

        @inbounds for ii in 1:n_terms
            PF._gatecommutesat(gate_mask, zone_terms, bytes, ii) && continue

            coeff = zone_coeffs[ii]
            zone_coeffs[ii] = coeff * kept_val

            new_pstr, sign = PF._gateproduct(gate_mask, zone_terms[ii], bytes, ii)
            PF._truncateweight(new_pstr, max_weight) && continue

            out_terms[pos] = new_pstr
            out_coeffs[pos] = coeff * new_val * sign
            pos += 1
        end
    end

    return pos - 1
end

# The outbox holds `parent ⊻ xor_mask` over a sorted, duplicate-free zone, which is what the XOR
# passes need; the free tail of this zone's own arrays serves as their second buffer.
function _mergefromoutbox!(mpsum::MultiVectorPauliSum, owner::Int, source::Int, xor_mask, truncfunc::F) where {F}
    n_tail = mpsum.counts[owner, source]
    n_tail == 0 && return

    zone = mpsum.zones[owner]
    outbox = mpsum.outboxes[source]
    n_old = activesize(zone)
    n_new = n_old + n_tail

    capacity(zone) < n_new && resize!(zone, n_new + n_new >> 1)
    main_terms, main_coeffs, aux_terms, aux_coeffs = PB._mainauxarrays(zone)

    ordered = mpsum.outbox_ordered[source] && n_old > 0 && n_tail >= PB._MIN_XOR_TAIL
    groups = ordered ? PB._xorplan(xor_mask, main_terms) : nothing

    if groups === nothing
        copyto!(main_terms, n_old + 1, paulis(outbox), 1, n_tail)
        copyto!(main_coeffs, n_old + 1, coefficients(outbox), 1, n_tail)
        setactivesize!(zone, n_new)
        merge!(zone; thread=false, truncfunc)
        return
    end

    tail_terms, tail_coeffs = PB._xorsorttail!(groups,
        view(paulis(outbox), 1:n_tail), view(coefficients(outbox), 1:n_tail),
        view(main_terms, n_old+1:n_new), view(main_coeffs, n_old+1:n_new); thread=false)

    PB._mergesortedhead!(zone, aux_terms, aux_coeffs, main_terms, main_coeffs, n_old,
        tail_terms, tail_coeffs, n_tail, truncfunc, false, Val(true))

    return
end


### Pauli noise

"""
    applygate!(gate::PauliNoise, mpsum::MultiVectorPauliSum, lambda; kwargs...)

Pauli noise leaves every Pauli string where it is, so no zone talks to any other and each one runs the
library's single-threaded fused overload.
"""
function applygate!(gate::PauliNoise, mpsum::MultiVectorPauliSum, lambda; thread::Bool=true, kwargs...)
    _eachzone(nzones(mpsum), thread) do zone_id
        applymergetruncate!(gate, mpsum.zones[zone_id], lambda; fused=true, thread=false, kwargs...)
    end
    return mpsum
end


### Clifford gates

"""
    applygate!(gate::CliffordGate, mpsum::MultiVectorPauliSum; kwargs...)

A Clifford gate maps each Pauli string to one other, so nothing branches and nothing collides, but the
new owner is not related to the old one. Each zone transforms its terms and sorts them into its outbox
by owner, then each zone gathers the segment addressed to it and sorts it.
"""
function applygate!(gate::CliffordGate, mpsum::MultiVectorPauliSum;
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf,
    customtruncfunc=nothing, thread::Bool=true, kwargs...)

    PauliPropagation._check_qind_range(mpsum.nqubits, gate.qinds)

    lookup_map = clifford_map[gate.symbol]
    truncfunc(pstr, coeff) = PF._fusedtruncfunc(pstr, coeff; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc)

    _eachzone(nzones(mpsum), thread) do source
        _scatterintooutbox!(mpsum, source, gate, lookup_map, truncfunc)
    end

    _eachzone(nzones(mpsum), thread) do owner
        _gatheroutboxes!(mpsum, owner)
    end

    return mpsum
end

function _scatterintooutbox!(mpsum::MultiVectorPauliSum, source::Int, gate::CliffordGate, lookup_map, truncfunc::F) where {F}
    zone = mpsum.zones[source]
    n_terms = activesize(zone)
    counts = view(mpsum.counts, :, source)

    fill!(counts, 0)
    n_terms == 0 && return

    zone_terms, zone_coeffs = paulis(mainsum(zone)), coefficients(mainsum(zone))

    # transform in place, then count how many terms each zone is owed
    @inbounds for ii in 1:n_terms
        pstr, coeff = only(apply(gate, zone_terms[ii], zone_coeffs[ii], lookup_map))
        zone_terms[ii], zone_coeffs[ii] = pstr, coeff
        truncfunc(pstr, coeff) || (counts[_zoneof(pstr, mpsum.zonemasks)] += 1)
    end

    outbox = mpsum.outboxes[source]
    length(outbox) < n_terms && resize!(outbox, n_terms)
    out_terms, out_coeffs = paulis(outbox), coefficients(outbox)

    positions = _segmentstarts(counts)
    @inbounds for ii in 1:n_terms
        pstr, coeff = zone_terms[ii], zone_coeffs[ii]
        truncfunc(pstr, coeff) && continue

        owner = _zoneof(pstr, mpsum.zonemasks)
        out_terms[positions[owner]] = pstr
        out_coeffs[positions[owner]] = coeff
        positions[owner] += 1
    end

    return
end

function _gatheroutboxes!(mpsum::MultiVectorPauliSum, owner::Int)
    zone = mpsum.zones[owner]
    n_total = sum(view(mpsum.counts, owner, :))

    setsortedprefix!(mainsum(zone), 0)
    setactivesize!(zone, n_total)
    n_total == 0 && return

    capacity(zone) < n_total && resize!(zone, n_total)
    zone_terms, zone_coeffs = paulis(mainsum(zone)), coefficients(mainsum(zone))

    pos = 1
    for source in 1:nzones(mpsum)
        n_segment = mpsum.counts[owner, source]
        n_segment == 0 && continue

        start = 1 + sum(view(mpsum.counts, 1:owner-1, source))
        copyto!(zone_terms, pos, paulis(mpsum.outboxes[source]), start, n_segment)
        copyto!(zone_coeffs, pos, coefficients(mpsum.outboxes[source]), start, n_segment)
        pos += n_segment
    end

    # a Clifford gate sends distinct Pauli strings to distinct Pauli strings, so sorting is all it takes
    sortbyterm!(zone; thread=false)
    setsortedprefix!(mainsum(zone), n_total)

    return
end

function _segmentstarts(counts)
    starts = similar(counts, Int)
    pos = 1
    for owner in eachindex(counts)
        starts[owner] = pos
        pos += counts[owner]
    end
    return starts
end
