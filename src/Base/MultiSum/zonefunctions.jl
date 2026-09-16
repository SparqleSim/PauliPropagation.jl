###
##
# Applying a gate to a multi sum in two passes: every zone makes its terms and parks them with their
# owners, then every zone takes delivery of the terms addressed to it and merges them in.
##
###

"""
    staysinzone(gate)::Bool

Whether `gate` leaves every term in the zone that owns it, in which case every zone applies the gate with the machinery of the sum it carries and no term is moved between zones.
Defaults to `false`, and can be overloaded for custom gates that only rescale coefficients.
"""
staysinzone(gate) = false

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

"""
    applyxorbranch!(branchfunc, prop_cache, mask; thread=true, kwargs...)

Apply a gate that moves every term it branches by the same `⊻ mask`.
For every term, `branchfunc(term, coeff)` returns `nothing` to leave the term untouched, or `(kept_coeff, new_coeff, branches)`, where the term keeps `kept_coeff` and, if `branches`, the term `term ⊻ mask` is collected with `new_coeff`.
Because the zone assignment is linear in the term, the gate permutes the zones, so that every zone writes into a single box of its outbox and receives from a single zone.
"""
function applyxorbranch!(branchfunc::F, prop_cache::AbstractPropagationCache, mask;
    thread::Bool=true, kwargs...) where {F<:Function}

    branch_zone!(source) = _branchzone!(branchfunc, prop_cache, source, mask)
    return _branchpasses!(branch_zone!, prop_cache, mask; thread, kwargs...)
end

"""
    applyxorbranchzones!(zonefunc, prop_cache, mask; thread=true, kwargs...)

Version of `applyxorbranch!()` with the first pass left to the caller.
`zonefunc(zonecache, box)` applies the gate to one entire zone and writes the terms it branches into `box`, instead of being handed one term at a time.
The zone that owns those terms then collects the box and merges it in.
Further `kwargs` are passed on to the merge, including `truncfunc`.
"""
function applyxorbranchzones!(zonefunc::F, prop_cache::AbstractPropagationCache, mask;
    thread::Bool=true, kwargs...) where {F<:Function}

    branch_zone!(source) = zonefunc(zonecaches(prop_cache)[source], _branchbox(prop_cache, source))
    return _branchpasses!(branch_zone!, prop_cache, mask; thread, kwargs...)
end

# every zone makes its terms and parks them, then every zone takes delivery and merges
function _branchpasses!(passfunc::F, prop_cache::AbstractPropagationCache, mask;
    thread::Bool=true, kwargs...) where {F<:Function}

    sorted_zones = _sortedzones(zonestorage(prop_cache), prop_cache)

    _eachzone(passfunc, prop_cache, thread)

    # The gate permutes the zones, so every zone collects from a single zone and touches no zone but
    # those two. The box it collects is its tail already, in the parent order of the zone that made
    # it, so an array zone sorts it in from where it is instead of taking delivery first.
    function collect_branch!(owner)
        source = _xortarget(zonemap(prop_cache), owner, mask)
        _mergebox!(zonestorage(prop_cache), zonecaches(prop_cache)[owner], _branchbox(prop_cache, source), mask, sorted_zones, source; kwargs...)
    end
    _eachzone(collect_branch!, prop_cache, thread)

    return _syncsums!(prop_cache)
end


### The two passes

# A fixed ⊻ mask moves every term of a zone into one and the same zone, so the gate has a single box
# to park in and never routes a term. A branching term keeps its own term and has only its
# coefficient rescaled, so it stays in this zone.
function _branchzone!(branchfunc::F, prop_cache::AbstractPropagationCache, source::Int, mask) where {F}
    zone_storage = zonestorage(prop_cache)
    zonecache = zonecaches(prop_cache)[source]
    box = _branchbox(prop_cache, source)

    for (ii, (term, coeff)) in enumerate(zonecache)
        branched = branchfunc(term, coeff)
        isnothing(branched) && continue

        kept_coeff, new_coeff, branches = branched
        _setcoeff!(zone_storage, zonecache, ii, term, kept_coeff)
        branches && _push!(zone_storage, box, term ⊻ mask, new_coeff)
    end

    return
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


### Merging in what a gate that branches by a fixed mask parked

# A zone that is sorted throughout hands its terms to a single other zone in ascending order, so the
# tail that zone takes delivery of is `mask ⊻ ascending` and sorts by XOR passes instead of by
# comparison. Merging here leaves `merge!` nothing to do afterwards.
_sortedzones(::StorageType, prop_cache::AbstractPropagationCache) = nothing

_sortedzones(::ArrayStorage, prop_cache::AbstractPropagationCache) =
    [sortedprefix(mainsum(zonecache)) == activesize(zonecache) for zonecache in zonecaches(prop_cache)]

# a dict zone merges as it takes delivery, where an array zone sorts the box in and merges it
_mergebox!(::StorageType, zonecache, box, mask, sorted_zones, source::Int; kwargs...) =
    _deliver!(zonecache, box)

_mergebox!(::ArrayStorage, zonecache, box, mask, sorted_zones, source::Int; kwargs...) =
    xorsortedboxmerge!(zonecache, box, mask, (@inbounds sorted_zones[source]); thread=false, kwargs...)


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

# one loop serves both storages, so it hands over what either of them writes by: a dict the term,
# an array the index
@inline _setcoeff!(::DictStorage, zonecache, ii::Int, term, coeff) = set!(mainsum(zonecache), term, coeff)
@inline _setcoeff!(::ArrayStorage, zonecache, ii::Int, term, coeff) = (coefficients(mainsum(zonecache))[ii] = coeff)
