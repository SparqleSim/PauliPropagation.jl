###
##
# The propagation cache of a MultiPauliSum: one cache per zone, plus one outbox per zone for the Pauli
# strings a gate sends to zones it does not own.
##
###

"""
    MultiPauliPropagationCache(msum::MultiPauliSum)

Propagation cache for a `MultiPauliSum`.
It carries the propagation cache of every zone, so that each zone propagates with the machinery of the Pauli sum type it carries, and one outbox per zone.
An outbox is itself a `MultiPauliSum` in which a zone collects every Pauli string it creates for the zone that owns it, and which the owning zones pick up once all zones have finished applying the gate.
This cache carries no auxiliary sum of its own, because every zone brings one.
"""
struct MultiPauliPropagationCache{MS<:MultiPauliSum,ZC<:AbstractPauliPropagationCache} <: AbstractPauliPropagationCache
    msum::MS
    zonecaches::Vector{ZC}
    outboxes::Vector{MS}
end

function MultiPauliPropagationCache(msum::MultiPauliSum)
    zonecaches = map(PropagationCache, zones(msum))
    outboxes = [similar(msum) for _ in 1:nzones(msum)]
    return MultiPauliPropagationCache(msum, zonecaches, outboxes)
end

PropagationBase.PropagationCache(msum::MultiPauliSum) = MultiPauliPropagationCache(msum)

PropagationBase.mainsum(prop_cache::MultiPauliPropagationCache) = prop_cache.msum

function Base.show(io::IO, prop_cache::MultiPauliPropagationCache)
    println(io, "MultiPauliPropagationCache with $(length(prop_cache)) terms over $(nzones(prop_cache)) zones:")
    println(io, "  zone sizes: ", zonesizes(prop_cache))
end
