###
##
# A term sum split over one work zone per thread, with an owning zone per term.
# Everything here is basis-agnostic: a concrete multi sum only carries its zones and its zone map,
# and the storage trait routes the term sum interface through the owning zone.
##
###

"""
    MultiSumStorage(zonestorage::StorageType) <: StorageType

Storage type of a term sum that is split over work zones, each of which is a term sum of the carried type that one thread owns.
`zonestorage` is the storage type of the zones, on which all zone-local operations dispatch.
A term sum carries this storage type by returning its zones from `storage()`, and additionally needs to define `zonemap()` and `withzones()`.
"""
struct MultiSumStorage{ST<:StorageType} <: StorageType
    zonestorage::ST
end

_storagetype(zones::Vector{<:AbstractTermSum}) = MultiSumStorage(StorageType(first(zones)))

"""
    zonestorage(thing)

Get the storage type of the zones of `thing`, which is either a multi sum or a propagation cache of one.
"""
zonestorage(thing) = StorageType(thing).zonestorage

"""
    zones(msum::AbstractTermSum)

Get the term sums that `msum` is split over, which is the storage that a multi sum carries.
"""
zones(msum::AbstractTermSum) = _zones(StorageType(msum), msum)
_zones(::MultiSumStorage, msum::AbstractTermSum) = storage(msum)

"""
    zonemap(msum::AbstractTermSum)

Get the `ZoneMap` that assigns each term to the zone that owns it.
Defaults to the `zonemap` field of `msum`.
"""
zonemap(msum::AbstractTermSum) = msum.zonemap

"""
    withzones(msum::AbstractTermSum, new_zones)

Return `msum` with its zones replaced by `new_zones` and everything else carried over.
This is the only function that needs to know the concrete type of a multi sum.
The zone-wise `similar()`, `emptylike()` and `activesum()` are built on it.
"""
withzones(msum::TS, new_zones) where {TS<:AbstractTermSum} = _thrownotimplemented(TS, :withzones)

"""
    nzones(msum::AbstractTermSum)

Get the number of work zones that `msum` is split over.
"""
nzones(msum::AbstractTermSum) = length(zones(msum))

"""
    zonesizes(msum::AbstractTermSum)

Get the number of terms in each zone of `msum`.
"""
zonesizes(msum::AbstractTermSum) = map(length, zones(msum))


### Zone assignment

"""
    ZoneMap(TermType, n_zones)

Assignment of terms of type `TermType` to one of `n_zones` zones, where `n_zones` must be a power of two.
Each bit of the zone index is the parity of the term under a fixed mask, which reads every site and spreads the terms evenly over the zones.
This makes the assignment linear over GF(2), that is `zoneof(t ⊻ m) - 1 == (zoneof(t) - 1) ⊻ (zoneof(m) - 1)`.
A gate that moves all of the terms it branches by the same `⊻ m` therefore permutes the zones, with each zone sending to and receiving from exactly one other zone.
This is what `applyxorbranch!()` relies on.
"""
struct ZoneMap{TT}
    masks::Vector{TT}
end

# the masks are the whole map: how many there are is the zone count, and what they are is the assignment
Base.:(==)(zone_map1::ZoneMap, zone_map2::ZoneMap) = zone_map1.masks == zone_map2.masks

function ZoneMap(::Type{TT}, n_zones::Integer) where {TT}
    n_zones >= 1 || throw(ArgumentError("n_zones must be positive, got $n_zones."))
    ispow2(n_zones) || throw(ArgumentError(
        "n_zones must be a power of two, got $n_zones. The nearest are $(prevpow(2, n_zones)) and $(nextpow(2, n_zones))."))

    return ZoneMap(_zonemasks(TT, trailing_zeros(Int(n_zones))))
end

"""
    defaultnzones()

Get the number of zones to split over when none is given, which is two per thread rounded up to a power of two, and a single zone when single-threaded.
Rounding down instead would leave threads idle for a whole round whenever the number of threads is not a power of two.
"""
defaultnzones() = nextpow(2, 2 * maxtasks(true) - 1)

"""
    zoneof(msum, term)
    zoneof(zone_map::ZoneMap, term)

Get the index of the zone that owns `term`.
"""
@inline zoneof(zone_map::ZoneMap, term) = _zonebits(term, zone_map.masks) + 1
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
function _copy!(::MultiSumStorage, dst_msum::AbstractTermSum, src_msum::AbstractTermSum)
    # a zone only holds the terms it owns, so copying across differing assignments loses ownership
    zonemap(dst_msum) == zonemap(src_msum) ||
        throw(ArgumentError("Cannot copy between multi sums with different zone assignments."))

    foreach(copy!, zones(dst_msum), zones(src_msum))
    return dst_msum
end

# a term sum merges and truncates without a `thread` argument, so these run the zones in turn and let
# each zone thread inside. The zone-parallel versions are the ones on the propagation cache.
_merge!(::MultiSumStorage, msum::AbstractTermSum) = (foreach(merge!, zones(msum)); msum)

function _truncate!(::MultiSumStorage, truncfunc::F, msum::AbstractTermSum; kwargs...) where {F<:Function}
    foreach(zone -> truncate!(truncfunc, zone; kwargs...), zones(msum))
    return msum
end

_maxabscoeff(::MultiSumStorage, msum::AbstractTermSum) = maximum(maxabscoeff, zones(msum))

# the zones are iterated one after the other, which carries neither a length nor an element type
_length(::MultiSumStorage, msum::AbstractTermSum) = sum(length, zones(msum))
_termtype(::MultiSumStorage, msum::AbstractTermSum) = termtype(first(zones(msum)))
_coefftype(::MultiSumStorage, msum::AbstractTermSum) = coefftype(first(zones(msum)))

# an empty multi sum of the same type is exactly what `emptylike` builds
_similar(::MultiSumStorage, msum::AbstractTermSum) = emptylike(msum)

# the zone assignment is a hash, so the zones take equal shares of the hint
_sizehint!(::MultiSumStorage, msum::AbstractTermSum, n) =
    (foreach(zone -> sizehint!(zone, cld(n, nzones(msum))), zones(msum)); msum)

# a p-norm over the zones' p-norms is the p-norm over all coefficients
_norm(::MultiSumStorage, msum::AbstractTermSum, L::Real) = LinearAlgebra.norm([norm(zone, L) for zone in zones(msum)], L)


### Parking terms with their owners

# Appends the term to the zone that owns it. The storage trait is passed rather than derived, since
# every zone of a multi sum carries the same one.
@inline _park!(msum::AbstractTermSum, term, coeff) = _pushterm!(zonestorage(msum), _zone(msum, term), term, coeff)

"""
    emptylike(term_sum::AbstractTermSum)

Create an empty term sum of the same type as `term_sum`, including its term type.
This differs from `similar()`, which for array-based term sums returns a term sum of the same length as `term_sum`.
The default implementations assume the constructor `TS(nsites, storage...)` and can be overloaded for types that carry more than that.
"""
emptylike(term_sum::AbstractTermSum) = _emptylike(StorageType(term_sum), term_sum)
_emptylike(::MultiSumStorage, msum::AbstractTermSum) = withzones(msum, map(emptylike, zones(msum)))
_emptylike(::DictStorage, term_sum::TS) where {TS} = Base.typename(TS).wrapper(nsites(term_sum), empty(storage(term_sum)))
_emptylike(::ArrayStorage, term_sum::TS) where {TS} = Base.typename(TS).wrapper(nsites(term_sum), empty(terms(term_sum)), empty(coefficients(term_sum)))
