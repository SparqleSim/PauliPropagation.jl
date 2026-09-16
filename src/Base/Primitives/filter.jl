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

function _filter!(::ArrayStorage, keep, prop_cache::AbstractPropagationCache; thread::Bool=true)
    isempty(prop_cache) && return prop_cache

    flag!(keep, prop_cache; thread)
    return filterviaflags!(prop_cache; thread)
end

_filter!(::StorageType, keep, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    _thrownotimplemented(thing, :filter!)
