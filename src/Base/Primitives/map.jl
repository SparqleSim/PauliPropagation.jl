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

function _map!(::ArrayStorage, transform, term_sum::AbstractTermSum; thread::Bool=true)
    source_terms = terms(term_sum)
    source_coefficients = coefficients(term_sum)

    AK.foreachindex(source_terms; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do index
        term, coefficient = transform(source_terms[index], source_coefficients[index])
        source_terms[index] = term
        source_coefficients[index] = coefficient
    end

    setsortedprefix!(term_sum, 0)
    return term_sum
end

function _map!(::ArrayStorage, transform, prop_cache::AbstractPropagationCache; thread::Bool=true)
    source_terms = terms(prop_cache)
    source_coefficients = coefficients(prop_cache)

    AK.foreachindex(source_terms; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do index
        term, coefficient = transform(source_terms[index], source_coefficients[index])
        source_terms[index] = term
        source_coefficients[index] = coefficient
    end

    setsortedprefix!(mainsum(prop_cache), 0)
    return prop_cache
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
