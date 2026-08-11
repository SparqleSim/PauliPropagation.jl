###
##
# Applying a gate to a MultiSum in two passes: every zone makes its terms and parks them with their
# owners, then every zone takes delivery of the terms addressed to it and merges them in.
##
###

"""
    propagate(circuit, msum::MultiSum, thetas=nothing; kwargs...)
    propagate!(circuit, msum::MultiSum, thetas=nothing; kwargs...)
    propagate!(circuit, prop_cache::MultiSumPropagationCache, thetas=nothing; kwargs...)

Propagate a `MultiSum` through `circuit`, exactly like any other term sum of the library.
`thread` decides whether the zones run in parallel; every operation inside a zone is single-threaded.
"""
function PropagationBase.propagate!(circuit, prop_cache::MultiSumPropagationCache, thetas=nothing;
    max_freq::Real=Inf, max_sins::Real=Inf, heisenberg::Bool=true, kwargs...)

    _checkfreqandsinfields(prop_cache, max_freq, max_sins)
    circuit, thetas = _preparecircuit(circuit, thetas, heisenberg)

    return PropagationBase._propagate!(circuit, prop_cache, thetas; max_freq, max_sins, kwargs...)
end

"""
    staysinzone(gate)::Bool

Whether `gate` leaves every term in the zone that owns it, in which case every zone applies it with
the machinery of the sum it carries and no term travels. Defaults to `false`, and can be overloaded
for custom gates that only rescale coefficients.
"""
staysinzone(gate) = false
staysinzone(::PauliNoise) = true


### Gates that move every term

"""
    applytoall!(gate, prop_cache::MultiSumPropagationCache, args...; kwargs...)

Applies `gate` to every term of every zone via `apply`, parking the terms it makes with the zones that
own them. This is the generic path, taken by every gate that does not branch off a copy of the terms
it is applied to.
"""
function PropagationBase.applytoall!(gate, prop_cache::MultiSumPropagationCache, args...;
    thread::Bool=true, kwargs...)

    if staysinzone(gate)
        _eachzone(prop_cache, thread) do zone_id
            applytoall!(gate, prop_cache.zonecaches[zone_id], args...; thread=false, kwargs...)
        end
        return _syncsums!(prop_cache)
    end

    _eachzone(prop_cache, thread) do source
        _movezone!(gate, prop_cache, source, args...; kwargs...)
    end

    return _collectzones!(prop_cache; thread)
end

"""
    applytoall!(gate::CliffordGate, prop_cache::MultiSumPropagationCache; kwargs...)

Provides the Clifford lookup map to the generic `applytoall!`, like the library does for its own
propagation caches.
"""
function PropagationBase.applytoall!(gate::CliffordGate, prop_cache::MultiSumPropagationCache; kwargs...)
    _check_qind_range(nsites(prop_cache), gate.qinds)
    return applytoall!(gate, prop_cache, clifford_map[gate.symbol]; kwargs...)
end

PropagationBase.applytoall!(gate::TGate, prop_cache::MultiSumPropagationCache; kwargs...) =
    applytoall!(PauliRotation(:Z, gate.qind), prop_cache, π / 4; kwargs...)

PropagationBase.applytoall!(gate::FrozenGate, prop_cache::MultiSumPropagationCache; kwargs...) =
    applytoall!(gate.gate, prop_cache, gate.parameter; kwargs...)


### Gates that branch

"""
    applytoall!(gate::PauliRotation, prop_cache::MultiSumPropagationCache, theta; kwargs...)

Every zone rescales the terms that branch and parks the terms they make in its outbox. The zone
assignment is linear in the term, so the gate permutes the zones and every zone receives from exactly
one other zone.
"""
PropagationBase.applytoall!(gate::PauliRotation, prop_cache::MultiSumPropagationCache, theta; kwargs...) =
    _branchtoall!(gate, prop_cache, cos(theta), sin(theta), Val(:PauliRotation); kwargs...)

PropagationBase.applytoall!(gate::ImaginaryPauliRotation, prop_cache::MultiSumPropagationCache, tau; kwargs...) =
    _branchtoall!(gate, prop_cache, cosh(tau), sinh(tau), Val(:ImaginaryPauliRotation); kwargs...)

"""
    applymergetruncate!(gate::ImaginaryPauliRotation, prop_cache::MultiSumPropagationCache, tau; normalize_coeffs=true, kwargs...)

Applies the gate, merges, and normalizes by the coefficient of the identity term, which lives in
whichever zone owns it.
"""
function PropagationBase.applymergetruncate!(gate::ImaginaryPauliRotation, prop_cache::MultiSumPropagationCache, tau;
    normalize_coeffs::Bool=true, kwargs...)

    applytoall!(gate, prop_cache, tau; kwargs...)
    merge!(prop_cache; kwargs...)

    if normalize_coeffs
        mult!(prop_cache, 1 / getcoeff(activesum(prop_cache), zero(termtype(prop_cache))))
    end

    truncate!(prop_cache; kwargs...)

    return prop_cache
end

"""
    applytoall!(gate::AmplitudeDampingNoise, prop_cache::MultiSumPropagationCache, gamma; kwargs...)

Every zone rescales its terms and parks the terms that its Z Paulis damp into with their owners.
"""
function PropagationBase.applytoall!(gate::AmplitudeDampingNoise, prop_cache::MultiSumPropagationCache, gamma;
    thread::Bool=true, kwargs...)

    _check_qind_range(nsites(prop_cache), gate.qind)
    _check_noise_strength(AmplitudeDampingNoise, gamma)

    _eachzone(prop_cache, thread) do source
        _dampzone!(prop_cache, source, gate.qind, gamma)
    end

    return _collectzones!(prop_cache; thread)
end


### The two passes

function _branchtoall!(gate, prop_cache::MultiSumPropagationCache, kept_val, new_val, gatetype::Val;
    thread::Bool=true, kwargs...)

    _check_qind_range(nsites(prop_cache), gate.qinds)
    gate_mask = symboltoint(termtype(prop_cache), gate.symbols, gate.qinds)

    _eachzone(prop_cache, thread) do source
        _branchzone!(prop_cache, source, gate_mask, kept_val, new_val, gatetype)
    end

    return _collectzones!(prop_cache; thread)
end

# A branching term keeps its own term and has only its coefficient rescaled, so it stays in this zone.
# The term it branches off goes to the outbox.
function _branchzone!(prop_cache::MultiSumPropagationCache, source::Int, gate_mask, kept_val, new_val, gatetype::Val)
    storage_type = StorageType(prop_cache)
    box = _xorbox!(prop_cache, source, gate_mask)
    zonecache = prop_cache.zonecaches[source]
    zone_coeffs = coefficients(zonecache)

    for (ii, (term, coeff)) in enumerate(zip(terms(zonecache), zone_coeffs))
        _branches(gatetype, commutes(gate_mask, term)) || continue

        new_term, sign = paulirotationproduct(gate_mask, term)
        _setcoeff!(storage_type, mainsum(zonecache), zone_coeffs, ii, term, coeff * kept_val)
        _park!(storage_type, box, new_term, coeff * new_val * sign)
    end

    return
end

@inline _branches(::Val{:PauliRotation}, does_commute) = !does_commute
@inline _branches(::Val{:ImaginaryPauliRotation}, does_commute) = does_commute

function _dampzone!(prop_cache::MultiSumPropagationCache, source::Int, qind::Integer, gamma)
    storage_type = StorageType(prop_cache)
    z_mask = symboltoint(termtype(prop_cache), :Z, qind)
    box = _xorbox!(prop_cache, source, z_mask)
    zonecache = prop_cache.zonecaches[source]
    zone_coeffs = coefficients(zonecache)

    for (ii, (term, coeff)) in enumerate(zip(terms(zonecache), zone_coeffs))
        pauli = getpauli(term, qind)
        pauli == 0 && continue

        if pauli == 3
            # Z damps into the identity on that site, which is this term ⊻ z_mask
            _setcoeff!(storage_type, mainsum(zonecache), zone_coeffs, ii, term, coeff * (1 - gamma))
            _park!(storage_type, box, term ⊻ z_mask, coeff * gamma)
        else
            _setcoeff!(storage_type, mainsum(zonecache), zone_coeffs, ii, term, coeff * sqrt(1 - gamma))
        end
    end

    return
end

# Every term this zone holds is moved to whichever zone owns the terms the gate makes from it.
function _movezone!(gate, prop_cache::MultiSumPropagationCache, source::Int, args...; kwargs...)
    outbox = empty!(prop_cache.outboxes[source])
    zonecache = prop_cache.zonecaches[source]

    for (term, coeff) in zip(terms(zonecache), coefficients(zonecache))
        for (new_term, new_coeff) in apply(gate, term, coeff, args...; kwargs...)
            _park!(outbox, new_term, new_coeff)
        end
    end

    _emptyzone!(StorageType(prop_cache), zonecache)

    return
end

# Second pass: every zone appends what the outboxes hold for it. Merging is left to `merge!`.
function _collectzones!(prop_cache::MultiSumPropagationCache; thread::Bool=true)
    _eachzone(prop_cache, thread) do owner
        _deliver!(StorageType(prop_cache), prop_cache.zonecaches[owner],
            (outbox.zones[owner] for outbox in prop_cache.outboxes))
    end
    return prop_cache
end


### Zone-local storage handling

# A fixed ⊻ mask moves every term of a zone into one and the same zone, so a gate that branches by
# that mask has a single box to park in and never routes a term.
@inline function _xorbox!(prop_cache::MultiSumPropagationCache, source::Int, mask)
    outbox = empty!(prop_cache.outboxes[source])
    return @inbounds outbox.zones[((source - 1) ⊻ _zonebits(mask, outbox.zonemasks))+1]
end

_deliver!(::DictStorage, zonecache, boxes) = foreach(box -> add!(mainsum(zonecache), box), boxes)

function _deliver!(::ArrayStorage, zonecache, boxes)
    n_old = activesize(zonecache)
    n_new = n_old + sum(length, boxes)
    n_new == n_old && return

    capacity(zonecache) < n_new && resize!(zonecache, n_new + n_new >> 1)
    zone_terms, zone_coeffs = terms(mainsum(zonecache)), coefficients(mainsum(zonecache))

    pos = n_old + 1
    for box in boxes
        copyto!(zone_terms, pos, terms(box), 1, length(box))
        copyto!(zone_coeffs, pos, coefficients(box), 1, length(box))
        pos += length(box)
    end

    setactivesize!(zonecache, n_new)

    return
end

_emptyzone!(::DictStorage, zonecache) = empty!(mainsum(zonecache))
_emptyzone!(::ArrayStorage, zonecache) = (setactivesize!(zonecache, 0); setsortedprefix!(mainsum(zonecache), 0))

@inline _setcoeff!(::DictStorage, term_sum, coeffs, ii::Int, term, coeff) = set!(term_sum, term, coeff)
@inline _setcoeff!(::ArrayStorage, term_sum, coeffs, ii::Int, term, coeff) = (coeffs[ii] = coeff)
