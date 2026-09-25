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
Several tasks on an array call `keep` once to count the pairs and once to write them, so it must return the same for the same pair each time.
"""
Base.filter!(keep, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    _filter!(StorageType(thing), keep, thing; thread)

function _filter!(::DictStorage, keep, term_sum::AbstractTermSum; thread::Bool=true)
    _filterdict!(keep, storage(term_sum))
    return term_sum
end

function _filter!(::DictStorage, keep, prop_cache::AbstractPropagationCache; thread::Bool=true)
    _filterdict!(keep, storage(mainsum(prop_cache)))
    return prop_cache
end

# filters through the slots where the internals of the dictionary are known
function _filterdict!(keep::F, dict) where {F}
    if _hasdictinternals(dict)
        _filter_internals!(keep, dict)
    else
        Base.filter!(entry -> keep(entry.first, entry.second), dict)
    end
    return dict
end

# a term sum filters through a propagation cache of its own
function _filter!(::StorageType, keep, term_sum::AbstractTermSum; thread::Bool=true)
    prop_cache = PropagationCache(term_sum)
    Base.filter!(keep, prop_cache; thread)
    return extractsum!(prop_cache, term_sum)
end

function _filter!(::StorageType, keep, prop_cache::AbstractPropagationCache; thread::Bool=true)
    keep_or_drop(term, coefficient) = keep(term, coefficient) ? ((term, coefficient),) : ()
    return flatmap!(keep_or_drop, prop_cache; thread)
end


### Array storage

# The kept pairs are compacted in the order they had, so the sorted prefix survives as the number of
# kept terms it held.
function _filter!(::ArrayStorage, keep::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    if isempty(prop_cache)
        return prop_cache
    end

    if _iscpuarray(prop_cache)
        return _filtercpu!(keep, prop_cache; thread)
    end

    flag!(keep, prop_cache; thread)
    return filterviaflags!(prop_cache; thread)
end

# One task compacts in place, since it writes at or behind the pair it just read. Several tasks
# first count what each of them keeps, then write their parts into the auxiliary arrays.
function _filtercpu!(keep::F, prop_cache; thread::Bool=true) where {F}
    n = activesize(prop_cache)
    n_sorted = sortedprefix(mainsum(prop_cache))
    task_partitioner, n_tasks = _preparetasks(n, thread)
    main_terms, main_coefficients, aux_terms, aux_coefficients = _mainauxarrays(prop_cache)

    if n_tasks == 1
        n_kept, n_sorted_kept = _filterwrite!(keep, main_terms, main_coefficients, 1, main_terms, main_coefficients, 1, n, n_sorted, Val(true))
        setactivesize!(prop_cache, n_kept)
        setsortedprefix!(mainsum(prop_cache), n_sorted_kept)
        return prop_cache
    end

    kept_counts = Vector{Int}(undef, n_tasks)
    sorted_kept_counts = Vector{Int}(undef, n_tasks)

    function count_kept!(task_id)
        chunk = task_partitioner[task_id]
        kept_counts[task_id], sorted_kept_counts[task_id] =
            _filterwrite!(keep, aux_terms, aux_coefficients, 1, main_terms, main_coefficients, chunk.start, chunk.stop, n_sorted, Val(false))
    end
    _eachtask(count_kept!, n_tasks)

    offsets = _offsetsfromcounts(kept_counts)

    # a task keeps at most as many pairs as its chunk holds, so it never writes past the chunk's end
    written = Vector{Int}(undef, n_tasks)
    function write_kept!(task_id)
        chunk = task_partitioner[task_id]
        written[task_id], _ = _filterwrite!(keep, aux_terms, aux_coefficients, offsets[task_id], main_terms, main_coefficients, chunk.start, chunk.stop, n_sorted, Val(true))
    end
    _eachtask(write_kept!, n_tasks)

    if written != kept_counts
        _throwreplaymismatch()
    end
    return _commitwrite!(prop_cache, offsets[end] - 1, sum(sorted_kept_counts))
end

# Walks terms[lo:hi] under `keep`, writing every kept pair from `write_start` on into the output arrays,
# or only counting on a dry run (`DoWrite` false). Returns the number of kept pairs and how many of
# them came from the first `n_sorted`.
@inline function _filterwrite!(keep::F, output_terms, output_coefficients, write_start,
    terms, coefficients, lo, hi, n_sorted, ::Val{DoWrite}) where {F,DoWrite}

    write_pos = write_start
    n_sorted_kept = 0

    @inbounds for ii in lo:hi
        if @inline(keep(terms[ii], coefficients[ii]))
            write_pos = _writeandadvance!(output_terms, output_coefficients, write_pos, terms[ii], coefficients[ii], Val(DoWrite))
            if ii <= n_sorted
                n_sorted_kept += 1
            end
        end
    end

    return write_pos - write_start, n_sorted_kept
end


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


function _filter!(::MultiSumStorage, keep, prop_cache::AbstractPropagationCache; thread::Bool=true)
    filter_zone!(zone_id) = filter!(keep, zonecaches(prop_cache)[zone_id]; thread=false)
    _eachzone(filter_zone!, prop_cache, thread)
    return _syncsums!(prop_cache)
end
