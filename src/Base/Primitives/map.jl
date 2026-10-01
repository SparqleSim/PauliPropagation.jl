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
while any other cache writes the transformed pairs through `flatmap!`.
"""
Base.map!(transform, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    _map!(StorageType(thing), transform, thing; thread)

function _map!(::StorageType, transform::F, term_sum::AbstractTermSum; thread::Bool=true) where {F}
    prop_cache = PropagationCache(term_sum)
    Base.map!(transform, prop_cache; thread)
    return extractsum!(prop_cache, term_sum)
end

function _map!(::StorageType, transform::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    map_to_pair(term, coefficient) = (transform(term, coefficient),)
    return flatmap!(map_to_pair, prop_cache; thread)
end

function _map!(::ArrayStorage, transform, term_sum::AbstractTermSum; thread::Bool=true)
    source_terms = terms(term_sum)
    source_coefficients = coefficients(term_sum)

    @assert length(source_terms) == length(source_coefficients)
    AK.foreachindex(source_terms; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do index
        @inbounds begin
            term, coefficient = @inline transform(source_terms[index], source_coefficients[index])
            source_terms[index] = term
            source_coefficients[index] = coefficient
        end
    end

    setsortedprefix!(term_sum, 0)
    return term_sum
end

function _map!(::ArrayStorage, transform, prop_cache::AbstractPropagationCache; thread::Bool=true)
    source_terms = terms(prop_cache)
    source_coefficients = coefficients(prop_cache)

    @assert length(source_terms) == length(source_coefficients)
    AK.foreachindex(source_terms; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do index
        @inbounds begin
            term, coefficient = @inline transform(source_terms[index], source_coefficients[index])
            source_terms[index] = term
            source_coefficients[index] = coefficient
        end
    end

    setsortedprefix!(mainsum(prop_cache), 0)
    return prop_cache
end


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
        @inbounds active_coefficients[index] = @inline transform(active_coefficients[index])
    end
    return thing
end

function _mapcoeffs!(::StorageType, transform, term_sum::AbstractTermSum; thread::Bool=true)
    for (term, coefficient) in term_sum
        set!(term_sum, term, transform(coefficient))
    end
    return term_sum
end

function _mapcoeffs!(::StorageType, transform, prop_cache::AbstractPropagationCache; thread::Bool=true)
    mapcoeffs!(transform, mainsum(prop_cache); thread)
    return prop_cache
end


"""
    mapcoeffsbypair!(transform, term_sum::AbstractTermSum; thread=true)
    mapcoeffsbypair!(transform, prop_cache::AbstractPropagationCache; thread=true)

Replace every active coefficient by `transform(term, coefficient)`, leaving terms unchanged.
Like `mapcoeffs!`, this updates coefficients in place and does not require cache scratch storage.
"""
mapcoeffsbypair!(transform, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    _mapcoeffsbypair!(StorageType(thing), transform, thing; thread)

_mapcoeffsbypair!(::ArrayStorage, transform, term_sum::AbstractTermSum; thread::Bool=true) =
    _mapactivecoeffsbypair!(transform, term_sum; thread)

_mapcoeffsbypair!(::ArrayStorage, transform, prop_cache::AbstractPropagationCache; thread::Bool=true) =
    _mapactivecoeffsbypair!(transform, prop_cache; thread)

function _mapactivecoeffsbypair!(transform::F, thing; thread::Bool=true) where {F}
    active_terms = terms(thing)
    active_coefficients = coefficients(thing)

    @assert length(active_terms) == length(active_coefficients)
    AK.foreachindex(active_coefficients; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do index
        @inbounds active_coefficients[index] = @inline transform(active_terms[index], active_coefficients[index])
    end
    return thing
end

function _mapcoeffsbypair!(::DictStorage, transform::F, term_sum::AbstractTermSum; thread::Bool=true) where {F}
    dict = storage(term_sum)
    if _hasdictinternals(dict)
        _mapcoeffsbypair_internals!(transform, dict)
        return term_sum
    end

    for (term, coefficient) in term_sum
        set!(term_sum, term, transform(term, coefficient))
    end
    return term_sum
end

function _mapcoeffsbypair!(::StorageType, transform, term_sum::AbstractTermSum; thread::Bool=true)
    for (term, coefficient) in term_sum
        set!(term_sum, term, transform(term, coefficient))
    end
    return term_sum
end

function _mapcoeffsbypair!(::StorageType, transform, prop_cache::AbstractPropagationCache; thread::Bool=true)
    mapcoeffsbypair!(transform, mainsum(prop_cache); thread)
    return prop_cache
end


function _mapcoeffs!(::MultiSumStorage, transform, msum::AbstractTermSum; thread::Bool=true)
    map_zone!(zone_id) = mapcoeffs!(transform, zones(msum)[zone_id]; thread=false)
    _eachzone(map_zone!, msum, thread)
    return msum
end

function _mapcoeffs!(::MultiSumStorage, transform, prop_cache::AbstractPropagationCache; thread::Bool=true)
    map_zone!(zone_id) = mapcoeffs!(transform, zonecaches(prop_cache)[zone_id]; thread=false)
    _eachzone(map_zone!, prop_cache, thread)
    return _syncsums!(prop_cache)
end

function _mapcoeffsbypair!(::MultiSumStorage, transform, msum::AbstractTermSum; thread::Bool=true)
    map_zone!(zone_id) = mapcoeffsbypair!(transform, zones(msum)[zone_id]; thread=false)
    _eachzone(map_zone!, msum, thread)
    return msum
end

function _mapcoeffsbypair!(::MultiSumStorage, transform, prop_cache::AbstractPropagationCache; thread::Bool=true)
    map_zone!(zone_id) = mapcoeffsbypair!(transform, zonecaches(prop_cache)[zone_id]; thread=false)
    _eachzone(map_zone!, prop_cache, thread)
    return _syncsums!(prop_cache)
end
