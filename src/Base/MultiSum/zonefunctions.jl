###
##
# Applying a gate to a multi sum in two passes: every zone makes its terms and parks them with their
# owners, then every zone takes delivery of the terms addressed to it and merges them in.
# A gate that branches by a fixed mask takes the second pass of `xorbranch!` below instead.
##
###

"""
    staysinzone(gate)::Bool

Whether `gate` leaves every term in the zone that owns it, in which case every zone applies the gate with the machinery of the sum it carries and no term is moved between zones.
Defaults to `false`, and can be overloaded for custom gates that only rescale coefficients.
"""
staysinzone(gate) = false

# a multi sum applies a gate zone by zone
_applytoall!(::MultiSumStorage, gate, prop_cache::AbstractPropagationCache, args...; kwargs...) =
    applytoallzones!(gate, prop_cache, args...; kwargs...)

"""
    applytoallzones!(gate, prop_cache, args...; thread=true, kwargs...)

Apply `gate` to every term of every zone via `apply()`, collecting the terms it creates in the outboxes of the zones that own them.
This is the generic path, taken by every gate that does not move all of the terms it branches by the same bitmask.
"""
function applytoallzones!(gate, prop_cache::AbstractPropagationCache, args...;
    thread::Bool=true, kwargs...)

    if staysinzone(gate)
        apply_in_zone!(zone_id) = applytoall!(gate, zonecaches(prop_cache)[zone_id], args...; thread=false, kwargs...)
        _eachzone(apply_in_zone!, prop_cache, thread)
        return _syncsums!(prop_cache)
    end

    move_zone!(source) = _movezone!(gate, prop_cache, source, args...; kwargs...)
    _eachzone(move_zone!, prop_cache, thread)

    # every zone appends what the outboxes hold for it, and merging is left to `merge!`
    deliver_to_zone!(owner) = foreach(outbox -> _deliver!(zonecaches(prop_cache)[owner], zones(outbox)[owner]), outboxes(prop_cache))
    _eachzone(deliver_to_zone!, prop_cache, thread)

    return prop_cache
end

# Every term this zone holds is moved to whichever zone owns the terms the gate makes from it.
function _movezone!(gate, prop_cache::AbstractPropagationCache, source::Int, args...; kwargs...)
    outbox = outboxes(prop_cache)[source]
    zonecache = zonecaches(prop_cache)[source]

    for (term, coeff) in zonecache
        for (new_term, new_coeff) in apply(gate, term, coeff, args...; kwargs...)
            push!(outbox, new_term, new_coeff)
        end
    end

    empty!(zonecache)

    return
end

### Branching by a fixed mask

# Because the zone assignment is linear in the term, `⊻ mask` permutes the zones: every zone writes
# the terms it creates into a single box, and takes delivery from a single zone.
function _xorbranch!(::MultiSumStorage, rule::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true, truncfunc=nothing) where {F}
    zone_storage = zonestorage(prop_cache)
    sorted_zones = _sortedzones(zone_storage, prop_cache)

    branch_zone!(zone_id) = _branchzone!(zone_storage, rule, zonecaches(prop_cache)[zone_id], _branchbox(prop_cache, zone_id), mask)
    _eachzone(branch_zone!, prop_cache, thread)

    # The box a zone collects is its tail already, in the parent order of the zone that made it, so
    # an array zone sorts it in from where it is instead of taking delivery first.
    function collect_branch!(owner)
        source = _xortarget(zonemap(prop_cache), owner, mask)
        _mergebox!(zone_storage, zonecaches(prop_cache)[owner], _branchbox(prop_cache, source), mask, sorted_zones, source; truncfunc)
    end
    _eachzone(collect_branch!, prop_cache, thread)

    return _syncsums!(prop_cache)
end

# a dict zone keeps its terms where they are and sets the new ones in its box
_branchzone!(::StorageType, rule::F, zonecache, box, mask) where {F} = _branchdict!(rule, mainsum(zonecache), box, mask)

# an array zone writes the new terms straight into its box, in the order of their parents
function _branchzone!(::ArrayStorage, rule::F, zonecache, box, mask) where {F}
    n_old = activesize(zonecache)

    # a term makes at most one new term, so the zone's size bounds what the box has to hold
    length(box) < n_old && resize!(box, n_old)
    n_new = _branchwrite!(rule, terms(box), coefficients(box), 1,
        terms(mainsum(zonecache)), coefficients(mainsum(zonecache)), 1, n_old, mask, Val(true))
    resize!(box, n_new)

    return zonecache
end


### Merging in what a gate that branches by a fixed mask parked

# A zone that is sorted throughout hands its terms to a single other zone in ascending order, so the
# tail that zone takes delivery of is `mask ⊻ ascending` and sorts by XOR passes instead of by
# comparison. Merging here leaves `merge!` nothing to do afterwards.
_sortedzones(::StorageType, prop_cache::AbstractPropagationCache) = nothing

_sortedzones(::ArrayStorage, prop_cache::AbstractPropagationCache) =
    [sortedprefix(mainsum(zonecache)) == activesize(zonecache) for zonecache in zonecaches(prop_cache)]

# a dict zone merges as it takes delivery, where an array zone sorts the box in and merges it
function _mergebox!(::StorageType, zonecache, box, mask, sorted_zones, source::Int; truncfunc=nothing)
    _deliver!(zonecache, box)
    truncfunc === nothing || truncate!(truncfunc, zonecache; thread=false)
    return zonecache
end

_mergebox!(::ArrayStorage, zonecache, box, mask, sorted_zones, source::Int; truncfunc=nothing) =
    xorsortedboxmerge!(zonecache, box, mask, (@inbounds sorted_zones[source]); thread=false, truncfunc)


### Zone-local storage handling

# ⊻-ing by `mask` maps zone `source` onto this zone, and this zone back onto `source`
@inline _xortarget(zone_map::ZoneMap, source::Int, mask) =
    ((source - 1) ⊻ _zonebits(mask, zone_map.masks)) + 1

# A zone that branches by a fixed mask sends everything it makes to a single zone, so one box holds
# it. Which zone that is moves with the mask, so parking in the box of the moment would leave every
# box of every outbox grown to the size of a zone.
@inline _branchbox(prop_cache::AbstractPropagationCache, zone_id::Int) =
    @inbounds first(zones(outboxes(prop_cache)[zone_id]))

# a box is emptied by the zone that takes delivery, so every box is empty when a gate picks it up
_deliver!(zonecache, box) = (add!(zonecache, box); empty!(box); zonecache)
