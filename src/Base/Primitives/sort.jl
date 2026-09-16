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

    AK.sortperm!(activeindices(prop_cache), activeterms(prop_cache); lt, by, rev, order,
        max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
    return permuteviaindices!(prop_cache; thread)
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
    return permuteviaindices!(prop_cache; thread)
end

_sortcoeffs!(::StorageType, thing::Union{AbstractTermSum,AbstractPropagationCache}; kwargs...) =
    _thrownotimplemented(thing, :sortcoeffs!)
