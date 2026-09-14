###
##
# The indexed cache split over work zones, one owner per Pauli string.
#
# Two things come out of this. The table probes and the coefficient writes are the only random
# accesses in the design, and splitting the sum keeps each zone's share of them inside the caches of
# the core that owns it, which is what the design lives or dies on once the sum outgrows L3. And
# because the zone assignment is linear over GF(2), a rotation sends every term of a zone into one
# and the same other zone, so it decomposes into independent zone pairs that share nothing -- no
# outbox, no routing pass, and nothing to lock.
##
###

"""
    MultiIndexedPauliPropagationCache

An `IndexedPauliPropagationCache` per work zone, with `PropagationBase.ZoneMap` deciding which zone
owns which Pauli string. Zone counts must be powers of two; `defaultnzones()` is one zone per
thread, which is the right default for parallelism but usually too few to keep a large sum in cache.
"""
struct MultiIndexedPauliPropagationCache{ZC<:IndexedPauliPropagationCache,ZM} <: PauliPropagation.AbstractPauliPropagationCache
    nqubits::Int
    zonecaches::Vector{ZC}
    zonemap::ZM
end

function MultiIndexedPauliPropagationCache(psum::PauliPropagation.AbstractPauliSum, n_zones::Integer=PropagationBase.defaultnzones())
    nq = nqubits(psum)
    zonemap = PropagationBase.ZoneMap(paulitype(psum), n_zones)

    zonesums = [VectorPauliSum(numcoefftype(psum), nq) for _ in 1:n_zones]
    for (pstr, coeff) in merge(psum)
        zone = zonesums[PropagationBase.zoneof(zonemap, pstr)]
        push!(terms(zone), pstr)
        push!(coefficients(zone), coeff)
    end

    return MultiIndexedPauliPropagationCache(nq, map(IndexedPauliPropagationCache, zonesums), zonemap)
end

MultiIndexedPauliPropagationCache(pstr::PauliString, n_zones::Integer=PropagationBase.defaultnzones()) =
    MultiIndexedPauliPropagationCache(PauliSum(pstr), n_zones)

zonecaches(cache::MultiIndexedPauliPropagationCache) = cache.zonecaches
PropagationBase.nzones(cache::MultiIndexedPauliPropagationCache) = length(cache.zonecaches)
PauliPropagation.nqubits(cache::MultiIndexedPauliPropagationCache) = cache.nqubits
PauliPropagation.paulitype(cache::MultiIndexedPauliPropagationCache) = paulitype(first(cache.zonecaches))

Base.length(cache::MultiIndexedPauliPropagationCache) = sum(zone -> zone.active_size, cache.zonecaches)
nliveterms(cache::MultiIndexedPauliPropagationCache) = sum(nliveterms, cache.zonecaches)

function Base.show(io::IO, cache::MultiIndexedPauliPropagationCache)
    print(io, "MultiIndexedPauliPropagationCache with $(nliveterms(cache)) terms ")
    print(io, "over $(nzones(cache)) zones on $(nqubits(cache)) qubits")
    return
end

function PauliPropagation.getcoeff(cache::MultiIndexedPauliPropagationCache, pstr)
    return getcoeff(cache.zonecaches[PropagationBase.zoneof(cache.zonemap, pstr)], pstr)
end

compact!(cache::MultiIndexedPauliPropagationCache) = (foreach(compact!, cache.zonecaches); cache)

function PauliPropagation.VectorPauliSum(cache::MultiIndexedPauliPropagationCache)
    compact!(cache)
    vpsum = VectorPauliSum(numcoefftype(mainsum(first(cache.zonecaches))), cache.nqubits)
    sizehint!(vpsum, length(cache))
    for zone in cache.zonecaches
        append!(terms(vpsum), view(terms(mainsum(zone)), 1:zone.active_size))
        append!(coefficients(vpsum), view(coefficients(mainsum(zone)), 1:zone.active_size))
    end
    return vpsum
end

PauliPropagation.PauliSum(cache::MultiIndexedPauliPropagationCache) = PauliSum(VectorPauliSum(cache))
PropagationBase.extractsum!(cache::MultiIndexedPauliPropagationCache) = VectorPauliSum(cache)

### Gates

"""
    applymergetruncate!(gate::PauliRotation, cache::MultiIndexedPauliPropagationCache, theta; kwargs...)

Apply one Pauli rotation to a zoned indexed cache, as independent work over the zone pairs the
rotation connects.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.PauliRotation, cache::MultiIndexedPauliPropagationCache, theta;
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, thread::Bool=true, kwargs...)

    PauliPropagation._check_qind_range(nqubits(cache), gate.qinds)
    gate_mask = PauliPropagation.symboltoint(paulitype(cache), gate.symbols, gate.qinds)
    _zonedrotate!(cache, gate_mask, cos(theta), sin(theta), min_abs_coeff, max_weight, Val(:PauliRotation), thread)
    return cache
end

"""
    applymergetruncate!(gate::ImaginaryPauliRotation, cache::MultiIndexedPauliPropagationCache, tau; normalize_coeffs=true, kwargs...)

Apply one imaginary Pauli rotation to a zoned indexed cache. The identity Pauli string that the
normalization divides by lives in one zone alone, and `getcoeff` finds it there.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.ImaginaryPauliRotation, cache::MultiIndexedPauliPropagationCache, tau;
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, normalize_coeffs::Bool=true, thread::Bool=true, kwargs...)

    PauliPropagation._check_qind_range(nqubits(cache), gate.qinds)
    gate_mask = PauliPropagation.symboltoint(paulitype(cache), gate.symbols, gate.qinds)
    _zonedrotate!(cache, gate_mask, cosh(tau), sinh(tau), min_abs_coeff, max_weight, Val(:ImaginaryPauliRotation), thread)

    if normalize_coeffs
        scale = 1 / getcoeff(cache, zero(paulitype(cache)))
        _eachzone(cache, thread) do z
            zone = cache.zonecaches[z]
            zone_coeffs = coefficients(mainsum(zone))
            @inbounds for i in 1:zone.active_size
                zone_coeffs[i] *= scale
            end
        end
    end

    return cache
end

PauliPropagation.applymergetruncate!(gate::PauliPropagation.FrozenGate, cache::MultiIndexedPauliPropagationCache; kwargs...) =
    PauliPropagation.applymergetruncate!(gate.gate, cache, gate.parameter; kwargs...)

"""
    applymergetruncate!(gate::PauliNoise, cache::MultiIndexedPauliPropagationCache, lambda; kwargs...)

Apply a Pauli noise channel to a zoned indexed cache. Noise only rescales coefficients, so every
zone handles its own terms.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.PauliNoise, cache::MultiIndexedPauliPropagationCache, lambda;
    thread::Bool=true, kwargs...)

    _eachzone(cache, thread) do z
        PauliPropagation.applymergetruncate!(gate, cache.zonecaches[z], lambda; kwargs...)
    end
    return cache
end

function PauliPropagation.applymergetruncate!(gate, cache::MultiIndexedPauliPropagationCache, args...; kwargs...)
    throw(ArgumentError("$(typeof(gate)) is not implemented for MultiIndexedPauliPropagationCache."))
end

### The rotation, zone pair by zone pair

# The zone a term lands in is its own zone XORed with the zone bits of the generator, so the zones
# split into pairs that exchange terms only with each other, or -- when the generator carries no
# zone bits at all -- into single zones that keep everything they make.
function _zonedrotate!(cache::MultiIndexedPauliPropagationCache, gate_mask::TT, kept_val, new_val,
    min_abs_coeff::Real, max_weight::Real, gatetype::Val, thread::Bool) where {TT}

    caches = cache.zonecaches
    columns = columnsof(gate_mask)
    offset = PropagationBase._zonebits(gate_mask, cache.zonemap.masks)
    local_mask = _PERF._gatemask(gate_mask, terms(mainsum(first(caches))))

    n_old = [zone.active_size for zone in caches]
    n_branching = zeros(Int, length(caches))

    _eachzone(cache, thread) do z
        zone = caches[z]
        appendterms!(zone.index, terms(mainsum(zone)), n_old[z])
        n_branching[z] = _markbranching!(zone.marks, zone.index, columns, n_old[z], gatetype)
    end

    _eachpair(cache, offset, thread) do a, b
        # each of the two walks may append every term the other one branches
        _reserve!(caches[a], n_old[a] + n_branching[b])
        _clearhandled!(caches[a], n_old[a])
        if b != a
            _reserve!(caches[b], n_old[b] + n_branching[a])
            _clearhandled!(caches[b], n_old[b])
        end

        _emitproducts!(caches[a], caches[b], n_old[a], local_mask, kept_val, new_val, min_abs_coeff, max_weight, gatetype)
        b == a || _emitproducts!(caches[b], caches[a], n_old[b], local_mask, kept_val, new_val, min_abs_coeff, max_weight, gatetype)

        _maybecompact!(caches[a])
        b == a || _maybecompact!(caches[b])
    end

    return cache
end

# Runs `zonefunc(a, b)` once per zone pair that `offset` connects, or once per zone with `a == b`
# when the generator keeps every term in its own zone.
function _eachpair(zonefunc::F, cache::MultiIndexedPauliPropagationCache, offset::Int, thread::Bool) where {F}
    n = nzones(cache)
    if offset == 0
        _eachzone(cache, thread) do z
            zonefunc(z, z)
        end
        return cache
    end

    # the lower zone of each pair names it, so each pair comes up exactly once
    lower = [z for z in 1:n if (z - 1) < ((z - 1) ⊻ offset)]
    _overitems(lower, thread) do z
        zonefunc(z, ((z - 1) ⊻ offset) + 1)
    end
    return cache
end

_eachzone(zonefunc::F, cache::MultiIndexedPauliPropagationCache, thread::Bool) where {F} =
    _overitems(zonefunc, 1:nzones(cache), thread)

function _overitems(itemfunc::F, items, thread::Bool) where {F}
    if thread && Threads.nthreads() > 1 && length(items) > 1
        Threads.@threads for item in items
            itemfunc(item)
        end
    else
        for item in items
            itemfunc(item)
        end
    end
    return
end
