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

function _merge!(::MultiSumStorage, prop_cache::AbstractPropagationCache; thread::Bool=true, kwargs...)
    merge_zone!(zone_id) = merge!(zonecaches(prop_cache)[zone_id]; thread=false, kwargs...)
    _eachzone(merge_zone!, prop_cache, thread)
    return _syncsums!(prop_cache)
end

# the slots of a zone follow those of the zones before it
function _mapslots!(::MultiSumStorage, weight_func::W, new_coeff_func::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {W,F}
    total_zone_weight(zonecache) = mapreducecoeffs(weight_func, +, zonecache; thread=false)
    zone_weights = _zonevalues(total_zone_weight, real(numcoefftype(prop_cache)), prop_cache, thread)
    zone_slot_starts = pushfirst!(cumsum(zone_weights), zero(eltype(zone_weights)))

    map_zone_slots!(zone_id) = _map_shifted_slots!(weight_func, new_coeff_func, zonecaches(prop_cache)[zone_id], zone_slot_starts[zone_id])
    _eachzone(map_zone_slots!, prop_cache, thread)

    return prop_cache
end

# `mapslots!` on one zone, with its slots starting at `zone_slot_start` instead of at zero
function _map_shifted_slots!(weight_func::W, new_coeff_func::F, zonecache, zone_slot_start) where {W,F}
    shifted_new_coeff_func(coeff, slot_start, slot_end) = new_coeff_func(coeff, zone_slot_start + slot_start, zone_slot_start + slot_end)
    return mapslots!(weight_func, shifted_new_coeff_func, zonecache; thread=false)
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

# a propagation over a multi sum keeps its workers up from the first gate to the last
_withworkers(::MultiSumStorage, f::F) where {F} = withworkers(f)

# a zone cache swaps its sums as it works, so the multi sum's zones follow it
function _syncsums!(prop_cache::AbstractPropagationCache)
    zone_sums = zones(prop_cache)
    for (zone_id, zonecache) in enumerate(zonecaches(prop_cache))
        zone_sums[zone_id] = mainsum(zonecache)
    end
    return prop_cache
end
