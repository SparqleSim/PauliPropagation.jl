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

Get the propagation caches of the zones.
Defaults to the `zonecaches` field of `prop_cache`.
"""
zonecaches(prop_cache::AbstractPropagationCache) = prop_cache.zonecaches

"""
    outboxes(prop_cache::AbstractPropagationCache)

Get the outbox of every zone, each of which is a multi sum in which a zone collects the terms it creates for the zones that own them.
Defaults to the `outboxes` field of `prop_cache`.
"""
outboxes(prop_cache::AbstractPropagationCache) = prop_cache.outboxes

zones(prop_cache::AbstractPropagationCache) = zones(mainsum(prop_cache))
zonemap(prop_cache::AbstractPropagationCache) = zonemap(mainsum(prop_cache))
nzones(prop_cache::AbstractPropagationCache) = length(zonecaches(prop_cache))

# a zone cache reports what it holds, where the zone itself reports the length of the arrays under it
zonesizes(prop_cache::AbstractPropagationCache) = map(length, zonecaches(prop_cache))


### The propagation cache interface, zone by zone

_length(::MultiSumStorage, prop_cache::AbstractPropagationCache) = sum(length, zonecaches(prop_cache))
_capacity(::MultiSumStorage, prop_cache::AbstractPropagationCache) = sum(capacity, zonecaches(prop_cache))

_terms(::MultiSumStorage, prop_cache::AbstractPropagationCache) =
    Iterators.flatten(terms(zonecache) for zonecache in zonecaches(prop_cache))
_coefficients(::MultiSumStorage, prop_cache::AbstractPropagationCache) =
    Iterators.flatten(coefficients(zonecache) for zonecache in zonecaches(prop_cache))

# the zones carry their own auxiliary sums, so the types are read off the main sum alone
_termtype(::MultiSumStorage, prop_cache::AbstractPropagationCache) = termtype(mainsum(prop_cache))
_coefftype(::MultiSumStorage, prop_cache::AbstractPropagationCache) = coefftype(mainsum(prop_cache))
_numcoefftype(::MultiSumStorage, prop_cache::AbstractPropagationCache) = numcoefftype(mainsum(prop_cache))

_activesum(::MultiSumStorage, prop_cache::AbstractPropagationCache) =
    Base.typename(typeof(mainsum(prop_cache))).wrapper(nsites(prop_cache), map(activesum, zonecaches(prop_cache)), zonemap(prop_cache))

# the zone assignment spreads the terms evenly, so the zones take equal shares of the room
function _resize!(::MultiSumStorage, prop_cache::AbstractPropagationCache, n_new::Int)
    per_zone = cld(n_new, nzones(prop_cache))
    foreach(zonecache -> resize!(zonecache, per_zone), zonecaches(prop_cache))
    return prop_cache
end

function _extractsum!(::MultiSumStorage, prop_cache::AbstractPropagationCache)
    foreach(extractsum!, zonecaches(prop_cache))
    return mainsum(_syncsums!(prop_cache))
end

### Working the zones

# Every zone is read and written by one thread only, so all parallelism comes from the zones. A
# sum below one task's worth of terms is worked in turn: a round costs tens of microseconds and
# more with every thread, where a zone that small takes one.
function _eachzone(zonefunc::F, thing, thread::Bool) where {F}
    if !thread || length(thing) < _MIN_ELEMS_PER_TASK
        for zone_id in 1:nzones(thing)
            zonefunc(zone_id)
        end
    else
        _eachtask(zonefunc, nzones(thing))
    end
    return thing
end

# a zone cache swaps its sums as it works, so the multi sum's zones follow it
function _syncsums!(prop_cache::AbstractPropagationCache)
    zone_sums = zones(prop_cache)
    for (zone_id, zonecache) in enumerate(zonecaches(prop_cache))
        zone_sums[zone_id] = mainsum(zonecache)
    end
    return prop_cache
end

# a box is emptied by the zone that takes delivery, so every box is empty when a gate picks it up
_deliver!(zonecache, box) = (add!(zonecache, box); empty!(box); zonecache)

function _checkauxempty(::MultiSumStorage, prop_cache::AbstractPropagationCache)
    for outbox in outboxes(prop_cache)
        if !isempty(outbox)
            _throwunmerged()
        end
    end
    return prop_cache
end

# a zone takes delivery of what the other zones parked in their outboxes for it
function _deliverto!(prop_cache::AbstractPropagationCache, owner::Int)
    zonecache = zonecaches(prop_cache)[owner]
    for outbox in outboxes(prop_cache)
        _deliver!(zonecache, zones(outbox)[owner])
    end
    return zonecache
end

function _deliverboxes!(prop_cache::AbstractPropagationCache; thread::Bool=true)
    deliver_to_zone!(owner) = _deliverto!(prop_cache, owner)
    return _eachzone(deliver_to_zone!, prop_cache, thread)
end
