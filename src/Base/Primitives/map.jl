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
Returning `nothing` drops the term. Storage may update coefficients in place or write kept pairs
through its scratch sum; sorted built-in storage stays sorted.
"""
mapcoeffsbypair!(transform, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    _mapcoeffsbypair!(StorageType(thing), transform, thing; thread)

_mapcoeffsbypair!(::DictStorage, transform::F, term_sum::AbstractTermSum; thread::Bool=true) where {F} =
    (_mapcoeffsbypairdict!(transform, storage(term_sum)); term_sum)

_mapcoeffsbypair!(::DictStorage, transform::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F} =
    (mapcoeffsbypair!(transform, mainsum(prop_cache); thread); prop_cache)

# Coefficients are updated while iterating, as `map!` on the values of a dictionary does, and the
# dropped terms are deleted afterwards, as `filter!` on a dictionary does.
function _mapcoeffsbypairdict!(transform::F, dict::AbstractDict) where {F}
    dropped = Vector{keytype(dict)}()

    for (term, coefficient) in dict
        mapped = @inline transform(term, coefficient)

        if mapped === nothing
            push!(dropped, term)
        elseif mapped !== coefficient
            dict[term] = mapped
        end
    end

    for term in dropped
        delete!(dict, term)
    end

    return dict
end

function _mapcoeffsbypair!(::ArrayStorage, transform::F, term_sum::AbstractTermSum; thread::Bool=true) where {F}
    prop_cache = PropagationCache(term_sum)
    mapcoeffsbypair!(transform, prop_cache; thread)
    return extractsum!(prop_cache, term_sum)
end

# The kept terms are compacted in the order they had, so the sorted prefix survives as the number of
# kept terms it held.
function _mapcoeffsbypair!(::ArrayStorage, transform::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    isempty(prop_cache) && return prop_cache

    if _iscpuarray(terms(mainsum(prop_cache)))
        return _mapcoeffsbypaircpu!(transform, prop_cache; thread)
    end

    return _mapcoeffsbypairflagged!(transform, prop_cache; thread)
end

# One task compacts in place, since it writes at or behind the term it just read. Several tasks
# first count what each of them keeps, then write their parts into the auxiliary arrays.
function _mapcoeffsbypaircpu!(transform::F, prop_cache; thread::Bool=true) where {F}
    n = activesize(prop_cache)
    n_sorted = sortedprefix(mainsum(prop_cache))
    task_partitioner, n_tasks = _preparetasks(n, thread)
    main_terms, main_coefficients, aux_terms, aux_coefficients = _mainauxarrays(prop_cache)

    if n_tasks == 1
        n_kept, n_sorted_kept = _mapcoeffsbypairwrite!(transform, main_terms, main_coefficients, 1, main_terms, main_coefficients, 1, n, n_sorted, Val(true))
        setactivesize!(prop_cache, n_kept)
        setsortedprefix!(mainsum(prop_cache), n_sorted_kept)
        return prop_cache
    end

    kept_counts = Vector{Int}(undef, n_tasks)
    sorted_kept_counts = Vector{Int}(undef, n_tasks)

    function count_kept!(task_id)
        chunk = task_partitioner[task_id]
        kept_counts[task_id], sorted_kept_counts[task_id] =
            _mapcoeffsbypairwrite!(transform, aux_terms, aux_coefficients, 1, main_terms, main_coefficients, chunk.start, chunk.stop, n_sorted, Val(false))
    end
    _eachtask(count_kept!, n_tasks)

    offsets = _offsetsfromcounts(kept_counts)

    function write_kept!(task_id)
        chunk = task_partitioner[task_id]
        _mapcoeffsbypairwrite!(transform, aux_terms, aux_coefficients, offsets[task_id], main_terms, main_coefficients, chunk.start, chunk.stop, n_sorted, Val(true))
    end
    _eachtask(write_kept!, n_tasks)

    return _commitwrite!(prop_cache, offsets[end] - 1, sum(sorted_kept_counts))
end

# Walks terms[lo:hi] under `transform`, writing every kept term from `write_start` on into the output arrays,
# or only counting on a dry run (`DoWrite` false). Returns the number of kept terms and how many of
# them came from the first `n_sorted`.
@inline function _mapcoeffsbypairwrite!(transform::F, output_terms, output_coefficients, write_start,
    terms, coefficients, lo, hi, n_sorted, ::Val{DoWrite}) where {F,DoWrite}

    write_pos = write_start
    n_sorted_kept = 0

    @inbounds for ii in lo:hi
        mapped = @inline transform(terms[ii], coefficients[ii])
        mapped === nothing && continue

        write_pos = _writeandadvance!(output_terms, output_coefficients, write_pos, terms[ii], mapped, Val(DoWrite))
        ii <= n_sorted && (n_sorted_kept += 1)
    end

    return write_pos - write_start, n_sorted_kept
end

# Every pass is an array kernel, so the arrays may live anywhere: the coefficients are mapped in
# place, and the flags then compact the kept terms.
function _mapcoeffsbypairflagged!(transform::F, prop_cache; thread::Bool=true) where {F}
    active_flags = activeflags(prop_cache)
    active_terms = activeterms(prop_cache)
    active_coefficients = activecoeffs(prop_cache)

    AK.foreachindex(active_flags; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do ii
        @inbounds begin
            mapped = @inline transform(active_terms[ii], active_coefficients[ii])
            active_flags[ii] = mapped !== nothing
            mapped === nothing || (active_coefficients[ii] = mapped)
        end
    end

    return filterviaflags!(prop_cache; thread)
end

function _mapcoeffsbypair!(::StorageType, transform::F, term_sum::AbstractTermSum; thread::Bool=true) where {F}
    prop_cache = PropagationCache(term_sum)
    mapcoeffsbypair!(transform, prop_cache; thread)
    return extractsum!(prop_cache, term_sum)
end

function _mapcoeffsbypair!(::StorageType, transform::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    function map_or_drop(term, coefficient)
        mapped = transform(term, coefficient)
        return mapped === nothing ? () : ((term, mapped),)
    end

    return flatmap!(map_or_drop, prop_cache; thread)
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

function _mapcoeffsbypair!(::MultiSumStorage, transform::F, msum::AbstractTermSum; thread::Bool=true) where {F}
    map_zone!(zone_id) = mapcoeffsbypair!(transform, zones(msum)[zone_id]; thread=false)
    _eachzone(map_zone!, msum, thread)
    return msum
end

function _mapcoeffsbypair!(::MultiSumStorage, transform::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    map_zone!(zone_id) = mapcoeffsbypair!(transform, zonecaches(prop_cache)[zone_id]; thread=false)
    _eachzone(map_zone!, prop_cache, thread)
    return _syncsums!(prop_cache)
end
