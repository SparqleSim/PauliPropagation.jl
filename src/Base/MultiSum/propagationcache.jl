###
##
# The propagation cache of a multi sum carries the propagation cache of every zone, so each zone
# propagates with the machinery of the sum it carries, plus one outbox per zone for the terms a gate
# sends to zones it does not own. There is no auxiliary sum on this level, since every zone brings its
# own.
##
###

"""
    zonecaches(prop_cache::AbstractPropagationCache)

The propagation caches of the zones. Defaults to the `zonecaches` field of `prop_cache`.
"""
zonecaches(prop_cache::AbstractPropagationCache) = prop_cache.zonecaches

"""
    outboxes(prop_cache::AbstractPropagationCache)

The outbox of every zone, each a multi sum in which a zone parks the terms it makes under the zone
that owns them. Defaults to the `outboxes` field of `prop_cache`.
"""
outboxes(prop_cache::AbstractPropagationCache) = prop_cache.outboxes

zones(prop_cache::AbstractPropagationCache) = zones(mainsum(prop_cache))
zonemap(prop_cache::AbstractPropagationCache) = zonemap(mainsum(prop_cache))
nzones(prop_cache::AbstractPropagationCache) = length(zonecaches(prop_cache))
zonesizes(prop_cache::AbstractPropagationCache) = map(length, zonecaches(prop_cache))


### The propagation cache interface, zone by zone

_length(::MultiSumStorage, prop_cache::AbstractPropagationCache) = sum(length, zonecaches(prop_cache))
_capacity(::MultiSumStorage, prop_cache::AbstractPropagationCache) = sum(capacity, zonecaches(prop_cache))

_terms(::MultiSumStorage, prop_cache::AbstractPropagationCache) = terms(activesum(prop_cache))
_coefficients(::MultiSumStorage, prop_cache::AbstractPropagationCache) = coefficients(activesum(prop_cache))

_maxabscoeff(::MultiSumStorage, prop_cache::AbstractPropagationCache) = maximum(maxabscoeff, zonecaches(prop_cache))

function _extractsum!(::MultiSumStorage, prop_cache::AbstractPropagationCache)
    foreach(extractsum!, zonecaches(prop_cache))
    return mainsum(_syncsums!(prop_cache))
end

function _merge!(::MultiSumStorage, prop_cache::AbstractPropagationCache; thread::Bool=true, kwargs...)
    _eachzone(prop_cache, thread) do zone_id
        merge!(zonecaches(prop_cache)[zone_id]; thread=false, kwargs...)
    end
    return _syncsums!(prop_cache)
end

function _truncate!(::MultiSumStorage, truncfunc::F, prop_cache::AbstractPropagationCache;
    thread::Bool=true, kwargs...) where {F<:Function}

    _eachzone(prop_cache, thread) do zone_id
        truncate!(truncfunc, zonecaches(prop_cache)[zone_id]; thread=false, kwargs...)
    end

    return _syncsums!(prop_cache)
end

"""
    _resizezones!(prop_cache::AbstractPropagationCache, n_new::Int)

Give the zones room for `n_new` terms between them. The zone assignment is a hash, so the shares are
equal.

A zone holds the terms addressed to it next to the terms it already has, so its own share has to
cover that peak, not just the terms that survive the gate.
"""
function _resizezones!(prop_cache::AbstractPropagationCache, n_new::Int)
    per_zone = cld(n_new, nzones(prop_cache))
    foreach(zonecache -> _reserve!(zonestorage(prop_cache), zonecache, per_zone), zonecaches(prop_cache))
    return prop_cache
end

_reserve!(::DictStorage, zonecache, n_new::Int) = sizehint!(storage(mainsum(zonecache)), n_new)
_reserve!(::ArrayStorage, zonecache, n_new::Int) = resize!(zonecache, n_new)


### Working the zones

# every zone is read and written by one thread only, so parallelism comes from the zones alone
function _eachzone(zonefunc::F, prop_cache::AbstractPropagationCache, thread::Bool) where {F}
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

# a zone cache swaps its sums as it works, so the multi sum's zones follow it
function _syncsums!(prop_cache::AbstractPropagationCache)
    zone_sums = zones(prop_cache)
    for (zone_id, zonecache) in enumerate(zonecaches(prop_cache))
        zone_sums[zone_id] = mainsum(zonecache)
    end
    return prop_cache
end
