###
##
# A propagation cache for a MultiSum: one cache per zone, plus one outbox per zone for the terms
# a gate sends to zones it does not own.
##
###

"""
    MultiSumPropagationCache(msum::MultiSum)

Propagation cache for a `MultiSum`. It carries the propagation cache of every zone, so each zone
propagates with the machinery of the sum it carries, and one outbox per zone. An outbox is itself a
`MultiSum`: a zone parks every term it makes under the zone that owns it, and the owners take
delivery once every zone has finished making terms.

There is no auxiliary sum on this level, since every zone brings its own.
"""
struct MultiSumPropagationCache{MS<:MultiSum,ZC<:AbstractPropagationCache} <: AbstractPropagationCache
    msum::MS
    zonecaches::Vector{ZC}
    outboxes::Vector{MS}
end

function MultiSumPropagationCache(msum::MultiSum)
    zonecaches = map(PropagationCache, msum.zones)
    outboxes = [similar(msum) for _ in 1:nzones(msum)]
    return MultiSumPropagationCache(msum, zonecaches, outboxes)
end

PropagationBase.PropagationCache(msum::MultiSum) = MultiSumPropagationCache(msum)

nzones(prop_cache::MultiSumPropagationCache) = length(prop_cache.zonecaches)
zonesizes(prop_cache::MultiSumPropagationCache) = map(length, prop_cache.zonecaches)

function Base.show(io::IO, prop_cache::MultiSumPropagationCache)
    println(io, "MultiSumPropagationCache with $(length(prop_cache)) terms over $(nzones(prop_cache)) zones:")
    println(io, "  zone sizes: ", zonesizes(prop_cache))
end


### The propagation cache interface

PropagationBase.mainsum(prop_cache::MultiSumPropagationCache) = prop_cache.msum
PropagationBase.activesum(prop_cache::MultiSumPropagationCache) =
    MultiSum(map(activesum, prop_cache.zonecaches), prop_cache.msum.zonemasks)

PropagationBase.StorageType(prop_cache::MultiSumPropagationCache) = StorageType(prop_cache.msum)
PropagationBase.nsites(prop_cache::MultiSumPropagationCache) = nsites(prop_cache.msum)
PropagationBase.termtype(prop_cache::MultiSumPropagationCache) = termtype(prop_cache.msum)
PropagationBase.coefftype(prop_cache::MultiSumPropagationCache) = coefftype(prop_cache.msum)
PropagationBase.numcoefftype(prop_cache::MultiSumPropagationCache) = numcoefftype(prop_cache.msum)

Base.length(prop_cache::MultiSumPropagationCache) = sum(length, prop_cache.zonecaches)
Base.isempty(prop_cache::MultiSumPropagationCache) = all(isempty, prop_cache.zonecaches)
PropagationBase.capacity(prop_cache::MultiSumPropagationCache) = sum(capacity, prop_cache.zonecaches)

PropagationBase.terms(prop_cache::MultiSumPropagationCache) = terms(activesum(prop_cache))
PropagationBase.coefficients(prop_cache::MultiSumPropagationCache) = coefficients(activesum(prop_cache))
PropagationBase.maxabscoeff(prop_cache::MultiSumPropagationCache) = maximum(maxabscoeff, prop_cache.zonecaches)

nqubits(prop_cache::MultiSumPropagationCache) = nqubits(prop_cache.msum)
paulis(prop_cache::MultiSumPropagationCache) = terms(prop_cache)
paulitype(prop_cache::MultiSumPropagationCache) = termtype(prop_cache)

function PropagationBase.mult!(prop_cache::MultiSumPropagationCache, scalar::Number)
    foreach(zonecache -> mult!(zonecache, scalar), prop_cache.zonecaches)
    return prop_cache
end

function PropagationBase.extractsum!(prop_cache::MultiSumPropagationCache)
    foreach(extractsum!, prop_cache.zonecaches)
    return _syncsums!(prop_cache).msum
end

"""
    resize!(prop_cache::MultiSumPropagationCache, n_new::Int)

Give the zones room for `n_new` terms between them. The zone assignment is a hash, so the shares are
equal.

A zone holds the terms addressed to it next to the terms it already has, so its own share has to
cover that peak, not just the terms that survive the gate.
"""
function Base.resize!(prop_cache::MultiSumPropagationCache, n_new::Int)
    per_zone = cld(n_new, nzones(prop_cache))
    foreach(zonecache -> _reserve!(StorageType(prop_cache), zonecache, per_zone), prop_cache.zonecaches)
    return prop_cache
end


### Merging and truncating, zone by zone

function Base.merge!(prop_cache::MultiSumPropagationCache; thread::Bool=true, kwargs...)
    _eachzone(prop_cache, thread) do zone_id
        merge!(prop_cache.zonecaches[zone_id]; thread=false, kwargs...)
    end
    return _syncsums!(prop_cache)
end

# `min_rel_coeff` is relative to the largest coefficient anywhere, so it is resolved before the zones
# are truncated against it
function PropagationBase.truncate!(prop_cache::MultiSumPropagationCache;
    min_abs_coeff::Real=1e-10, min_rel_coeff=nothing, thread::Bool=true, kwargs...)

    min_coeff = isnothing(min_rel_coeff) ? min_abs_coeff : max(min_rel_coeff * maxabscoeff(prop_cache), min_abs_coeff)

    _eachzone(prop_cache, thread) do zone_id
        truncate!(prop_cache.zonecaches[zone_id]; min_abs_coeff=min_coeff, thread=false, kwargs...)
    end

    return _syncsums!(prop_cache)
end

function PropagationBase.truncate!(truncfunc::F, prop_cache::MultiSumPropagationCache;
    thread::Bool=true, kwargs...) where {F<:Function}

    _eachzone(prop_cache, thread) do zone_id
        truncate!(truncfunc, prop_cache.zonecaches[zone_id]; thread=false, kwargs...)
    end

    return _syncsums!(prop_cache)
end


### Working the zones

# every zone is read and written by one thread only, so parallelism comes from the zones alone
function _eachzone(zonefunc::F, prop_cache::MultiSumPropagationCache, thread::Bool) where {F}
    if thread
        @threads for zone_id in 1:nzones(prop_cache)
            zonefunc(zone_id)
        end
    else
        for zone_id in 1:nzones(prop_cache)
            zonefunc(zone_id)
        end
    end
    return prop_cache
end

# a zone cache swaps its sums as it works, so the MultiSum's zones follow it
function _syncsums!(prop_cache::MultiSumPropagationCache)
    for (zone_id, zonecache) in enumerate(prop_cache.zonecaches)
        prop_cache.msum.zones[zone_id] = mainsum(zonecache)
    end
    return prop_cache
end

_reserve!(::DictStorage, zonecache, n_new::Int) = sizehint!(storage(mainsum(zonecache)), n_new)
_reserve!(::ArrayStorage, zonecache, n_new::Int) = resize!(zonecache, n_new)
