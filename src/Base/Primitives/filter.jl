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

function _filter!(::DictStorage, keep, term_sum::AbstractTermSum; thread::Bool=true)
    Base.filter!(entry -> keep(entry.first, entry.second), storage(term_sum))
    return term_sum
end

function _filter!(::DictStorage, keep, prop_cache::AbstractPropagationCache; thread::Bool=true)
    Base.filter!(entry -> keep(entry.first, entry.second), storage(mainsum(prop_cache)))
    return prop_cache
end

function _filter!(::ArrayStorage, keep, term_sum::AbstractTermSum; thread::Bool=true)
    prop_cache = PropagationCache(term_sum)
    Base.filter!(keep, prop_cache; thread)
    return extractsum!(prop_cache, term_sum)
end

function _filter!(::ArrayStorage, keep::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    keep_or_drop(term, coefficient) = keep(term, coefficient) ? coefficient : nothing
    return mapcoeffsbypair!(keep_or_drop, prop_cache; thread)
end

_filter!(::StorageType, keep, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    _thrownotimplemented(thing, :filter!)


"""
    filterterms(keep, term_sum::AbstractTermSum; thread=true)
    filterterms(keep, prop_cache::AbstractPropagationCache; thread=true)

Keep active terms for which `keep(term)` returns `true`, returning a copy.
"""
filterterms(keep, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    filterterms!(keep, deepcopy(thing); thread)

"""
    filterterms!(keep, term_sum::AbstractTermSum; thread=true)
    filterterms!(keep, prop_cache::AbstractPropagationCache; thread=true)

Remove active terms for which `keep(term)` returns `false`.
"""
filterterms!(keep, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    Base.filter!((term, coefficient) -> keep(term), thing; thread)


"""
    filtercoeffs(keep, term_sum::AbstractTermSum; thread=true)
    filtercoeffs(keep, prop_cache::AbstractPropagationCache; thread=true)

Keep active coefficients for which `keep(coefficient)` returns `true`, returning a copy.
"""
filtercoeffs(keep, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    filtercoeffs!(keep, deepcopy(thing); thread)

"""
    filtercoeffs!(keep, term_sum::AbstractTermSum; thread=true)
    filtercoeffs!(keep, prop_cache::AbstractPropagationCache; thread=true)

Remove active coefficients for which `keep(coefficient)` returns `false`.
"""
filtercoeffs!(keep, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    Base.filter!((term, coefficient) -> keep(coefficient), thing; thread)


### Multi-sum storage

function _filter!(::MultiSumStorage, keep, msum::AbstractTermSum; thread::Bool=true)
    prop_cache = PropagationCache(msum)
    filter!(keep, prop_cache; thread)
    return extractsum!(prop_cache, msum)
end

function _filter!(::MultiSumStorage, keep, prop_cache::AbstractPropagationCache; thread::Bool=true)
    filter_zone!(zone_id) = filter!(keep, zonecaches(prop_cache)[zone_id]; thread=false)
    _eachzone(filter_zone!, prop_cache, thread)
    return _syncsums!(prop_cache)
end
