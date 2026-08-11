###
##
# A term sum split over one work zone per thread, with an owning zone per term.
# Everything here is basis-agnostic: a concrete multi sum only carries its zones and zone masks,
# and the storage trait routes the term sum interface through the owning zone.
##
###

"""
    MultiSumStorage(zonestorage::StorageType) <: StorageType

Storage trait of a term sum split over work zones, each a term sum of the carried type that a single
thread owns. `zonestorage` is the storage trait of the zones, which every zone-local operation
dispatches on.

A term sum carries this trait by returning its zones from `storage` and its zone masks from
`zonemasks`.
"""
struct MultiSumStorage{ST<:StorageType} <: StorageType
    zonestorage::ST
end

_storagetype(zones::Vector{<:AbstractTermSum}) = MultiSumStorage(StorageType(first(zones)))

"""
    zonestorage(thing)

The storage trait of the zones of `thing`, which is a multi sum or a cache propagating one.
"""
zonestorage(thing) = StorageType(thing).zonestorage

"""
    zones(msum::AbstractTermSum)

The term sums that `msum` is split over. Defaults to `storage(msum)`.
"""
zones(msum::AbstractTermSum) = storage(msum)

"""
    zonemap(msum::AbstractTermSum)

The [`ZoneMap`](@ref) that assigns a term to its zone. Defaults to the `zonemap` field of `msum`.
"""
zonemap(msum::AbstractTermSum) = msum.zonemap

"""
    nzones(msum::AbstractTermSum)

The number of work zones that `msum` is split over.
"""
nzones(msum::AbstractTermSum) = length(zones(msum))

"""
    zonesizes(msum::AbstractTermSum)

The number of terms in each zone of `msum`.
"""
zonesizes(msum::AbstractTermSum) = map(length, zones(msum))


### Zone assignment

"""
    ZoneMap(TermType, n_zones)

Assigns every term to one of `n_zones` zones. Each bit of an intermediate bucket index is a parity of
the term under a fixed mask, so the buckets read every site and no zone is favoured.

A power-of-two `n_zones` gives an [`XorZoneMap`](@ref), whose zone index is the bucket index itself.
Any other count gives a [`FoldedZoneMap`](@ref), which folds more buckets than zones onto the zones.
"""
abstract type ZoneMap end

"""
    XorZoneMap(masks)

Zone map over `2^length(masks)` zones, whose zone index is the vector of mask parities of the term.
The assignment is then linear, `zoneof(t ⊻ m) - 1 == (zoneof(t) - 1) ⊻ (zoneof(m) - 1)`, so a gate
that moves every term it branches by the same `⊻ m` permutes the zones: each zone sends all of them to
exactly one other zone, and each zone receives from exactly one. This is what
[`applyxorbranch!`](@ref) parks into a single box on.
"""
struct XorZoneMap{TT} <: ZoneMap
    masks::Vector{TT}
end

"""
    FoldedZoneMap(masks, zoneofbucket)

Zone map over any number of zones, which `zoneofbucket` assigns the `2^length(masks)` buckets to in
turn. `⊻` is only closed on the buckets, so folding breaks the linearity of [`XorZoneMap`](@ref) and a
gate that branches by a fixed mask has to route every term it makes.
"""
struct FoldedZoneMap{TT} <: ZoneMap
    masks::Vector{TT}
    zoneofbucket::Vector{Int}
end

# enough buckets per zone that folding them evenly leaves the zones within a few percent of each other
const _BUCKETS_PER_ZONE_BITS = 5
const _MAX_BUCKET_BITS = 16

function ZoneMap(::Type{TT}, n_zones::Integer) where {TT}
    n_zones >= 1 || throw(ArgumentError("n_zones must be positive, got $n_zones."))

    zone_bits = trailing_zeros(nextpow(2, n_zones))
    ispow2(n_zones) && return XorZoneMap(_zonemasks(TT, zone_bits))

    n_bits = max(zone_bits, min(_MAX_BUCKET_BITS, zone_bits + _BUCKETS_PER_ZONE_BITS))
    zoneofbucket = [bucket % Int(n_zones) + 1 for bucket in 0:(1<<n_bits-1)]
    return FoldedZoneMap(_zonemasks(TT, n_bits), zoneofbucket)
end

"""
    isxorlinear(zone_map::ZoneMap)

Whether `⊻`-ing every term by the same mask permutes the zones of `zone_map`.
"""
isxorlinear(::XorZoneMap) = true
isxorlinear(::FoldedZoneMap) = false
isxorlinear(thing) = isxorlinear(zonemap(thing))

"""
    defaultnzones()

The number of zones to split over when none is given: as many as there are threads.
"""
defaultnzones() = maxtasks(true)

"""
    zoneof(msum, term)
    zoneof(zone_map::ZoneMap, term)

The zone that owns `term`.
"""
@inline zoneof(zone_map::XorZoneMap, term) = _zonebits(term, zone_map.masks) + 1
@inline zoneof(zone_map::FoldedZoneMap, term) = @inbounds zone_map.zoneofbucket[_zonebits(term, zone_map.masks)+1]
@inline zoneof(msum, term) = zoneof(zonemap(msum), term)

@inline _zone(msum, term) = @inbounds zones(msum)[zoneof(msum, term)]

@inline function _zonebits(term, zonemasks)
    bits = 0
    @inbounds for r in eachindex(zonemasks)
        bits |= Int(isodd(count_ones(term & zonemasks[r]))) << (r - 1)
    end
    return bits
end

const _ZONEMASK_SEED = 0x9e3779b97f4a7c15

# any fixed set of masks works; pseudo-random ones read every site, so no zone is favoured
function _zonemasks(::Type{TT}, n_bits::Int) where {TT}
    masks = TT[]
    state = _ZONEMASK_SEED

    for _ in 1:n_bits
        mask = zero(TT)
        for word in 0:cld(sizeof(TT), 8)-1
            state += _ZONEMASK_SEED
            mask |= (_mix64(state) % TT) << (64 * word)
        end
        push!(masks, mask)
    end

    return masks
end

@inline function _mix64(x::UInt64)
    z = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end


### The term sum interface, routed through the owning zone

_terms(::MultiSumStorage, msum::AbstractTermSum) = Iterators.flatten(terms(zone) for zone in zones(msum))
_coefficients(::MultiSumStorage, msum::AbstractTermSum) = Iterators.flatten(coefficients(zone) for zone in zones(msum))
@inline _iterate(::MultiSumStorage, msum::AbstractTermSum, args...) = iterate(Iterators.flatten(zones(msum)), args...)

_getcoeff(::MultiSumStorage, msum::AbstractTermSum, trm) = getcoeff(_zone(msum, trm), trm)
_getmergedcoeff(::MultiSumStorage, msum::AbstractTermSum, trm) = getmergedcoeff(_zone(msum, trm), trm)

@inline _add!(::MultiSumStorage, msum::AbstractTermSum, term, coeff) = (add!(_zone(msum, term), term, coeff); msum)
@inline _set!(::MultiSumStorage, msum::AbstractTermSum, term, coeff) = (set!(_zone(msum, term), term, coeff); msum)
_delete!(::MultiSumStorage, msum::AbstractTermSum, term) = (delete!(_zone(msum, term), term); msum)

_mult!(::MultiSumStorage, msum::AbstractTermSum, scalar::Number) = (foreach(zone -> mult!(zone, scalar), zones(msum)); msum)
_empty!(::MultiSumStorage, msum::AbstractTermSum) = (foreach(empty!, zones(msum)); msum)
_copy!(::MultiSumStorage, dst_msum::AbstractTermSum, src_msum::AbstractTermSum) = (foreach(copy!, zones(dst_msum), zones(src_msum)); dst_msum)

_merge!(::MultiSumStorage, msum::AbstractTermSum) = (foreach(merge!, zones(msum)); msum)

function _truncate!(::MultiSumStorage, truncfunc::F, msum::AbstractTermSum; kwargs...) where {F<:Function}
    foreach(zone -> truncate!(truncfunc, zone; kwargs...), zones(msum))
    return msum
end

_maxabscoeff(::MultiSumStorage, msum::AbstractTermSum) = maximum(maxabscoeff, zones(msum))


### Parking terms without deduplication

@inline _park!(msum::AbstractTermSum, term, coeff) = _park!(zonestorage(msum), _zone(msum, term), term, coeff)

@inline _park!(::DictStorage, term_sum, term, coeff) = add!(term_sum, term, coeff)

@inline function _park!(::ArrayStorage, term_sum, term, coeff)
    push!(terms(term_sum), term)
    push!(coefficients(term_sum), coeff)
    return term_sum
end

"""
    emptylike(term_sum::AbstractTermSum)

An empty term sum of the type of `term_sum`, term type included. A zone must carry the very type it
was seeded from, so it is rebuilt from the empty storage rather than from the coefficient type and
the number of sites, and `similar` is no help because it keeps the length.

The defaults assume the constructor `TS(nsites, storage...)`. Overload this for a type that carries
more than that.
"""
emptylike(term_sum::AbstractTermSum) = _emptylike(StorageType(term_sum), term_sum)
_emptylike(::DictStorage, term_sum::TS) where {TS} = Base.typename(TS).wrapper(nsites(term_sum), empty(storage(term_sum)))
_emptylike(::ArrayStorage, term_sum::TS) where {TS} = Base.typename(TS).wrapper(nsites(term_sum), empty(terms(term_sum)), empty(coefficients(term_sum)))
