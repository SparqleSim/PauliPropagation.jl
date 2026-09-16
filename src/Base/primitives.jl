###
##
# Backend- and container-agnostic primitive operations on term sums and caches.
##
###


# `map` and `filter` work on `(term, coefficient)` pairs. Their `terms` and `coeffs`
# variants work on one half of the pair only. Every public operation dispatches through
# `StorageType`, so a new storage backend can provide efficient implementations without
# changing its term sum or propagation cache type. A storage uses cache scratch only when
# its operation needs it.


_activesum(term_sum::AbstractTermSum) = term_sum
_activesum(prop_cache::AbstractPropagationCache) = mainsum(prop_cache)


"""
    map(transform, term_sum::AbstractTermSum; thread=true)
    map(transform, prop_cache::AbstractPropagationCache; thread=true)

Apply `transform(term, coefficient)` to every active term and coefficient pair, returning a copy.
`transform` must return a `(term, coefficient)` pair of values accepted by the destination.
"""
Base.map(transform, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    Base.map!(transform, deepcopy(thing); thread)

"""
    map!(transform, term_sum::AbstractTermSum; thread=true)
    map!(transform, prop_cache::AbstractPropagationCache; thread=true)

Replace every active `(term, coefficient)` pair by the pair returned from
`transform(term, coefficient)`. Array-backed storage updates its active entries in place,
while dictionary-backed caches write transformed pairs into their auxiliary sum.
"""
Base.map!(transform, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    _map!(StorageType(thing), transform, thing; thread)

function _map!(::DictStorage, transform, term_sum::AbstractTermSum; thread::Bool=true)
    prop_cache = PropagationCache(term_sum)
    Base.map!(transform, prop_cache; thread)
    return extractsum!(prop_cache, term_sum)
end

function _map!(::DictStorage, transform, prop_cache::AbstractPropagationCache; thread::Bool=true)
    output_sum = auxsum(prop_cache)
    empty!(output_sum)
    sizehint!(output_sum, length(prop_cache))

    for (term, coefficient) in prop_cache
        mapped_term, mapped_coefficient = transform(term, coefficient)
        add!(output_sum, mapped_term, mapped_coefficient)
    end

    return swapsums!(prop_cache)
end

function _map!(::ArrayStorage, transform, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true)
    active_terms = terms(thing)
    active_coefficients = coefficients(thing)

    AK.foreachindex(active_terms; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do index
        term, coefficient = transform(active_terms[index], active_coefficients[index])
        active_terms[index] = term
        active_coefficients[index] = coefficient
    end

    setsortedprefix!(_activesum(thing), 0)
    return thing
end

_map!(::StorageType, transform, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    _thrownotimplemented(thing, :map!)


"""
    mapterms(transform, term_sum::AbstractTermSum; thread=true)
    mapterms(transform, prop_cache::AbstractPropagationCache; thread=true)

Apply `transform(term)` to every active term, returning a copy and leaving coefficients unchanged.
"""
mapterms(transform, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    mapterms!(transform, deepcopy(thing); thread)

"""
    mapterms!(transform, term_sum::AbstractTermSum; thread=true)
    mapterms!(transform, prop_cache::AbstractPropagationCache; thread=true)

Replace every active term by `transform(term)`, leaving coefficients unchanged.
"""
mapterms!(transform, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    _mapterms!(StorageType(thing), transform, thing; thread)

_mapterms!(::StorageType, transform, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    Base.map!((term, coefficient) -> (transform(term), coefficient), thing; thread)


"""
    mapcoeffs(transform, term_sum::AbstractTermSum; thread=true)
    mapcoeffs(transform, prop_cache::AbstractPropagationCache; thread=true)

Apply `transform(coefficient)` to every active coefficient, returning a copy and leaving terms unchanged.
"""
mapcoeffs(transform, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    mapcoeffs!(transform, deepcopy(thing); thread)

"""
    mapcoeffs!(transform, term_sum::AbstractTermSum; thread=true)
    mapcoeffs!(transform, prop_cache::AbstractPropagationCache; thread=true)

Replace every active coefficient by `transform(coefficient)`, leaving terms unchanged. This operation
updates coefficients in place and does not require cache scratch storage.
"""
mapcoeffs!(transform, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    _mapcoeffs!(StorageType(thing), transform, thing; thread)

function _mapcoeffs!(::DictStorage, transform, term_sum::AbstractTermSum; thread::Bool=true)
    Base.map!(transform, coefficients(term_sum))
    return term_sum
end

function _mapcoeffs!(::DictStorage, transform, prop_cache::AbstractPropagationCache; thread::Bool=true)
    Base.map!(transform, coefficients(prop_cache))
    return prop_cache
end

function _mapcoeffs!(::ArrayStorage, transform, thing::AbstractTermSum; thread::Bool=true)
    return _mapactivecoeffs!(transform, thing; thread)
end

function _mapcoeffs!(::ArrayStorage, transform, thing::AbstractPropagationCache; thread::Bool=true)
    return _mapactivecoeffs!(transform, thing; thread)
end

function _mapactivecoeffs!(transform, thing; thread::Bool=true)
    active_coefficients = coefficients(thing)
    AK.foreachindex(active_coefficients; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do index
        active_coefficients[index] = transform(active_coefficients[index])
    end
    return thing
end

function _mapcoeffs!(::StorageType, transform, term_sum::AbstractTermSum; thread::Bool=true)
    for (term, coefficient) in term_sum
        set!(term_sum, term, transform(coefficient))
    end
    return term_sum
end

_mapcoeffs!(::StorageType, transform, prop_cache::AbstractPropagationCache; thread::Bool=true) =
    _thrownotimplemented(prop_cache, :mapcoeffs!)


"""
    filter(keep, term_sum::AbstractTermSum; thread=true)
    filter(keep, prop_cache::AbstractPropagationCache; thread=true)

Keep the active `(term, coefficient)` pairs for which `keep(term, coefficient)` returns `true`,
returning a copy.
"""
Base.filter(keep, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    Base.filter!(keep, deepcopy(thing); thread)

"""
    filter!(keep, term_sum::AbstractTermSum; thread=true)
    filter!(keep, prop_cache::AbstractPropagationCache; thread=true)

Remove active `(term, coefficient)` pairs for which `keep(term, coefficient)` returns `false`.
"""
Base.filter!(keep, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    _filter!(StorageType(thing), keep, thing; thread)

function _filter!(::DictStorage, keep, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true)
    Base.filter!(entry -> keep(entry.first, entry.second), storage(_activesum(thing)))
    return thing
end

function _filter!(::ArrayStorage, keep, term_sum::AbstractTermSum; thread::Bool=true)
    prop_cache = PropagationCache(term_sum)
    Base.filter!(keep, prop_cache; thread)
    return extractsum!(prop_cache, term_sum)
end

function _filter!(::ArrayStorage, keep, prop_cache::AbstractPropagationCache; thread::Bool=true)
    isempty(prop_cache) && return prop_cache

    flag!(keep, prop_cache; thread)
    filterviaflags!(prop_cache; thread)
    return prop_cache
end

_filter!(::StorageType, keep, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    _thrownotimplemented(thing, :filter!)


"""
    sortterms(term_sum::AbstractTermSum; kwargs...)
    sortterms(prop_cache::AbstractPropagationCache; kwargs...)

Return a copy whose active terms have been sorted. For multi sums, terms are sorted within each
zone, whose ownership partition is retained.
"""
sortterms(thing::Union{AbstractTermSum,AbstractPropagationCache}; kwargs...) =
    sortterms!(deepcopy(thing); kwargs...)

"""
    sortterms!(term_sum::AbstractTermSum; lt=isless, by=identity, rev=false, order=Base.Forward, thread=true)
    sortterms!(prop_cache::AbstractPropagationCache; lt=isless, by=identity, rev=false, order=Base.Forward, thread=true)

Sort active terms. Dictionary-backed sums retain their unordered storage and are returned unchanged.
"""
sortterms!(thing::Union{AbstractTermSum,AbstractPropagationCache}; lt=isless, by=identity, rev::Bool=false,
    order=Base.Forward, thread::Bool=true) =
    _sortterms!(StorageType(thing), thing; lt, by, rev, order, thread)

_sortterms!(::DictStorage, thing::Union{AbstractTermSum,AbstractPropagationCache}; kwargs...) = thing

function _sortterms!(::ArrayStorage, term_sum::AbstractTermSum; kwargs...)
    prop_cache = PropagationCache(term_sum)
    sortterms!(prop_cache; kwargs...)
    return extractsum!(prop_cache, term_sum)
end

function _sortterms!(::ArrayStorage, prop_cache::AbstractPropagationCache; lt=isless, by=identity,
    rev::Bool=false, order=Base.Forward, thread::Bool=true)

    sortbyterm!(prop_cache; lt, by, rev, order, thread)
    return prop_cache
end

_sortterms!(::StorageType, thing::Union{AbstractTermSum,AbstractPropagationCache}; kwargs...) =
    _thrownotimplemented(thing, :sortterms!)


"""
    sortcoeffs(term_sum::AbstractTermSum; kwargs...)
    sortcoeffs(prop_cache::AbstractPropagationCache; kwargs...)

Return a copy whose active coefficients have been sorted. For multi sums, coefficients are sorted
within each zone, whose ownership partition is retained.
"""
sortcoeffs(thing::Union{AbstractTermSum,AbstractPropagationCache}; kwargs...) =
    sortcoeffs!(deepcopy(thing); kwargs...)

"""
    sortcoeffs!(term_sum::AbstractTermSum; lt=isless, by=identity, rev=false, order=Base.Forward, thread=true)
    sortcoeffs!(prop_cache::AbstractPropagationCache; lt=isless, by=identity, rev=false, order=Base.Forward, thread=true)

Sort active coefficients, moving their terms with them. Dictionary-backed sums retain their unordered
storage and are returned unchanged.
"""
sortcoeffs!(thing::Union{AbstractTermSum,AbstractPropagationCache}; lt=isless, by=identity, rev::Bool=false,
    order=Base.Forward, thread::Bool=true) =
    _sortcoeffs!(StorageType(thing), thing; lt, by, rev, order, thread)

_sortcoeffs!(::DictStorage, thing::Union{AbstractTermSum,AbstractPropagationCache}; kwargs...) = thing

function _sortcoeffs!(::ArrayStorage, term_sum::AbstractTermSum; kwargs...)
    prop_cache = PropagationCache(term_sum)
    sortcoeffs!(prop_cache; kwargs...)
    return extractsum!(prop_cache, term_sum)
end

function _sortcoeffs!(::ArrayStorage, prop_cache::AbstractPropagationCache; lt=isless, by=identity,
    rev::Bool=false, order=Base.Forward, thread::Bool=true)

    AK.sortperm!(activeindices(prop_cache), activecoeffs(prop_cache); lt, by, rev, order,
        max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
    permuteviaindices!(prop_cache; thread)
    return prop_cache
end

_sortcoeffs!(::StorageType, thing::Union{AbstractTermSum,AbstractPropagationCache}; kwargs...) =
    _thrownotimplemented(thing, :sortcoeffs!)
