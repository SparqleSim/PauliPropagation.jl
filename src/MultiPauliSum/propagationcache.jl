###
##
# The propagation cache of a MultiPauliSum: one cache per zone, plus one outbox per zone for the Pauli
# strings a gate sends to zones it does not own.
##
###

"""
    MultiPauliPropagationCache(msum::MultiPauliSum)

Propagation cache for a `MultiPauliSum`. It carries the propagation cache of every zone, so each zone
propagates with the machinery of the sum it carries, and one outbox per zone. An outbox is itself a
`MultiPauliSum`: a zone parks every Pauli string it makes under the zone that owns it, and the owners take
delivery once every zone has finished making Pauli strings.

There is no auxiliary sum on this level, since every zone brings its own.
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

PropagationBase.activesum(prop_cache::MultiPauliPropagationCache) =
    MultiPauliSum(nqubits(prop_cache), map(activesum, zonecaches(prop_cache)), zonemap(prop_cache))

# the zones carry their own auxiliary sums, so the types are read off the main sum alone
PropagationBase.termtype(prop_cache::MultiPauliPropagationCache) = termtype(mainsum(prop_cache))
PropagationBase.coefftype(prop_cache::MultiPauliPropagationCache) = coefftype(mainsum(prop_cache))
PropagationBase.numcoefftype(prop_cache::MultiPauliPropagationCache) = numcoefftype(mainsum(prop_cache))

Base.resize!(prop_cache::MultiPauliPropagationCache, n_new::Int) = PropagationBase._resizezones!(prop_cache, n_new)

function Base.show(io::IO, prop_cache::MultiPauliPropagationCache)
    println(io, "MultiPauliPropagationCache with $(length(prop_cache)) terms over $(nzones(prop_cache)) zones:")
    println(io, "  zone sizes: ", zonesizes(prop_cache))
end
