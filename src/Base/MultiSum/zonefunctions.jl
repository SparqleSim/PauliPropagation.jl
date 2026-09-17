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

# a box is emptied by the zone that takes delivery, so every box is empty when a gate picks it up
_deliver!(zonecache, box) = (add!(zonecache, box); empty!(box); zonecache)
