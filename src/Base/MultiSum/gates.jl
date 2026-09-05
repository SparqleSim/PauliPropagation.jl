###
##
# Applying a gate to a multi sum in two passes: every zone makes its terms and parks them with their
# owners, then every zone takes delivery of the terms addressed to it and merges them in.
##
###

"""
    staysinzone(gate)::Bool

Whether `gate` leaves every term in the zone that owns it, in which case every zone applies it with
the machinery of the sum it carries and no term travels. Defaults to `false`, and can be overloaded
for custom gates that only rescale coefficients.
"""
staysinzone(gate) = false

"""
    applytoallzones!(gate, prop_cache, args...; thread=true, kwargs...)

Applies `gate` to every term of every zone via `apply`, parking the terms it makes with the zones that
own them. This is the generic path, taken by every gate that does not branch off a copy of the terms
it is applied to.
"""
function applytoallzones!(gate, prop_cache::AbstractPropagationCache, args...;
    thread::Bool=true, kwargs...)

    if staysinzone(gate)
        _eachzone(prop_cache, thread) do zone_id
            applytoall!(gate, zonecaches(prop_cache)[zone_id], args...; thread=false, kwargs...)
        end
        return _syncsums!(prop_cache)
    end

    _eachzone(prop_cache, thread) do source
        _movezone!(gate, prop_cache, source, args...; kwargs...)
    end

    return _collectzones!(prop_cache; thread)
end

"""
    applyxorbranch!(branchfunc, prop_cache, mask; thread=true, kwargs...)

Applies a gate that moves every term it branches by the same `⊻ mask`.

For every term, `branchfunc(term, coeff)` returns `nothing` to leave it untouched, or
`(kept_coeff, new_coeff, branches)`: the term keeps `kept_coeff`, and if `branches` the term
`term ⊻ mask` is parked with `new_coeff`.

The zone assignment is linear in the term, so the gate permutes the zones and every zone parks into a
single box and receives from a single zone.
"""
function applyxorbranch!(branchfunc::F, prop_cache::AbstractPropagationCache, mask;
    thread::Bool=true, kwargs...) where {F<:Function}

    return _branchpasses!(prop_cache, mask; thread, kwargs...) do source
        _branchzone!(branchfunc, prop_cache, source, mask)
    end
end

"""
    applyxorbranchzones!(zonefunc, prop_cache, mask; thread=true, kwargs...)

[`applyxorbranch!`](@ref) with the first pass left to the caller: `zonefunc(zonecache, box)` applies
the gate to one zone and writes what it branches into `box`, rather than being handed one term at a
time. The zone that owns those terms then takes delivery of the box and merges it in.

`kwargs` reach the merge, `truncfunc` included.
"""
function applyxorbranchzones!(zonefunc::F, prop_cache::AbstractPropagationCache, mask;
    thread::Bool=true, kwargs...) where {F<:Function}

    return _branchpasses!(prop_cache, mask; thread, kwargs...) do source
        zonefunc(zonecaches(prop_cache)[source], _branchbox(prop_cache, source))
    end
end

# every zone makes its terms and parks them, then every zone takes delivery and merges
function _branchpasses!(passfunc::F, prop_cache::AbstractPropagationCache, mask;
    thread::Bool=true, kwargs...) where {F<:Function}

    sorted_zones = _sortedzones(zonestorage(prop_cache), prop_cache)

    _eachzone(prop_cache, thread) do source
        passfunc(source)
    end

    _collectbranch!(prop_cache, mask; thread)

    return _mergebranch!(zonestorage(prop_cache), prop_cache, mask, sorted_zones; thread, kwargs...)
end


### The two passes

# A fixed ⊻ mask moves every term of a zone into one and the same zone, so the gate has a single box
# to park in and never routes a term.
function _branchzone!(branchfunc::F, prop_cache::AbstractPropagationCache, source::Int, mask) where {F}
    zone_storage = zonestorage(prop_cache)
    box = _branchbox(prop_cache, source)

    return _branchterms!(branchfunc, prop_cache, source, mask) do new_term, new_coeff
        _pushterm!(zone_storage, box, new_term, new_coeff)
    end
end

# A branching term keeps its own term and has only its coefficient rescaled, so it stays in this zone.
# The term it branches off is parked by `parkfunc`.
function _branchterms!(parkfunc::P, branchfunc::F, prop_cache::AbstractPropagationCache,
    source::Int, mask) where {P,F}

    zone_storage = zonestorage(prop_cache)
    zonecache = zonecaches(prop_cache)[source]
    zone_coeffs = coefficients(zonecache)

    for (ii, (term, coeff)) in enumerate(zip(terms(zonecache), zone_coeffs))
        branched = branchfunc(term, coeff)
        isnothing(branched) && continue

        kept_coeff, new_coeff, branches = branched
        _setcoeff!(zone_storage, mainsum(zonecache), zone_coeffs, ii, term, kept_coeff)
        branches && parkfunc(term ⊻ mask, new_coeff)
    end

    return
end

# Every term this zone holds is moved to whichever zone owns the terms the gate makes from it.
function _movezone!(gate, prop_cache::AbstractPropagationCache, source::Int, args...; kwargs...)
    outbox = outboxes(prop_cache)[source]
    zonecache = zonecaches(prop_cache)[source]

    for (term, coeff) in zip(terms(zonecache), coefficients(zonecache))
        for (new_term, new_coeff) in apply(gate, term, coeff, args...; kwargs...)
            _park!(outbox, new_term, new_coeff)
        end
    end

    _emptyzone!(zonestorage(prop_cache), zonecache)

    return
end

# Second pass: every zone appends what the outboxes hold for it. Merging is left to `merge!`.
function _collectzones!(prop_cache::AbstractPropagationCache; thread::Bool=true)
    _eachzone(prop_cache, thread) do owner
        _deliver!(zonestorage(prop_cache), zonecaches(prop_cache)[owner],
            (zones(outbox)[owner] for outbox in outboxes(prop_cache)))
    end
    return prop_cache
end

# The gate permutes the zones, so every zone has a single zone to collect from.
function _collectbranch!(prop_cache::AbstractPropagationCache, mask; thread::Bool=true)
    zone_map = zonemap(prop_cache)
    _eachzone(prop_cache, thread) do owner
        box = _branchbox(prop_cache, _xortarget(zone_map, owner, mask))
        _deliver!(zonestorage(prop_cache), zonecaches(prop_cache)[owner], (box,))
    end
    return prop_cache
end


### Merging what a gate that branches by a fixed mask appended

# A zone that is sorted throughout hands its terms to a single other zone in ascending order, so the
# tail that zone takes delivery of is `mask ⊻ ascending` and sorts by XOR passes instead of by
# comparison. Merging here leaves `merge!` nothing to do afterwards.
_sortedzones(::StorageType, prop_cache::AbstractPropagationCache) = nothing

_sortedzones(::ArrayStorage, prop_cache::AbstractPropagationCache) =
    [sortedprefix(mainsum(zonecache)) == activesize(zonecache) for zonecache in zonecaches(prop_cache)]

_mergebranch!(::StorageType, prop_cache::AbstractPropagationCache, mask, sorted_zones; kwargs...) = prop_cache

function _mergebranch!(::ArrayStorage, prop_cache::AbstractPropagationCache,
    mask, sorted_zones; thread::Bool=true, kwargs...)

    zone_map = zonemap(prop_cache)
    _eachzone(prop_cache, thread) do owner
        source = _xortarget(zone_map, owner, mask)
        xorsortedtailmerge!(zonecaches(prop_cache)[owner], mask, @inbounds sorted_zones[source];
            thread=false, kwargs...)
    end

    return _syncsums!(prop_cache)
end


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
_deliver!(::DictStorage, zonecache, boxes) = foreach(box -> (add!(mainsum(zonecache), box); empty!(box)), boxes)

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
        empty!(box)
    end

    setactivesize!(zonecache, n_new)

    return
end

_emptyzone!(::DictStorage, zonecache) = empty!(mainsum(zonecache))
_emptyzone!(::ArrayStorage, zonecache) = (setactivesize!(zonecache, 0); setsortedprefix!(mainsum(zonecache), 0))

# one loop serves both storages, so it hands over everything either of them needs: a dict writes by
# term, an array by index into the coefficients its caller hoisted out of the loop
@inline _setcoeff!(::DictStorage, term_sum, coeffs, ii::Int, term, coeff) = set!(term_sum, term, coeff)
@inline _setcoeff!(::ArrayStorage, term_sum, coeffs, ii::Int, term, coeff) = (coeffs[ii] = coeff)
