### MERGE

"""
    merge(term_sum::AbstractTermSum; thread=true)
    merge(prop_cache::AbstractPropagationCache; thread=true)

`merge!` on a copy.
"""
Base.merge(thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) = merge!(deepcopy(thing); thread)

"""
    merge!(term_sum::AbstractTermSum; thread=true)
    merge!(prop_cache::AbstractPropagationCache; thread=true)

Merge the terms the last gate left in the auxiliary sum, or appended past the sorted prefix, into the main sum, combining equal terms with `mergefunc`.
A term sum combines its own equal terms the same way, through a propagation cache of its own where it needs one.
"""
Base.merge!(thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) = mergeandtruncate!(nothing, thing; thread)

"""
    mergeandtruncate!(truncfunc, term_sum::AbstractTermSum; thread=true)
    mergeandtruncate!(truncfunc, prop_cache::AbstractPropagationCache; thread=true)

Merge all equal active terms, then drop every `(term, coefficient)` pair for which `truncfunc(term, coefficient)` returns `true`.
A `truncfunc` of `nothing` drops nothing, which is `merge!`; every storage merges the two the same way.
`truncfunc` sees the merged coefficient, so contributions can still cancel before a term is judged,
and it is applied even when nothing needs merging, since a gate may have rescaled the terms without creating any.
An array on the CPU truncates as it writes the merged terms, every other storage merges and then truncates.
Several tasks on an array call `truncfunc` once to count the terms and once to write them, so it must return the same for the same pair each time.
"""
mergeandtruncate!(truncfunc::F, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) where {F} =
    _mergeandtruncate!(StorageType(thing), truncfunc, thing; thread)


### Term sums

# a dictionary is merged by construction
_mergeandtruncate!(::DictStorage, truncfunc::F, term_sum::AbstractTermSum; thread::Bool=true) where {F} =
    _truncate!(truncfunc, term_sum; thread)

# any other term sum merges through a propagation cache of its own
function _mergeandtruncate!(storage::StorageType, truncfunc::F, term_sum::AbstractTermSum; thread::Bool=true) where {F}
    prop_cache = PropagationCache(term_sum)
    _mergeandtruncate!(storage, truncfunc, prop_cache; thread)
    return extractsum!(prop_cache, term_sum)
end

# equal terms share a zone, so the zones merge on their own
function _mergeandtruncate!(::MultiSumStorage, truncfunc::F, msum::AbstractTermSum; thread::Bool=true) where {F}
    merge_zone!(zone_id) = mergeandtruncate!(truncfunc, zones(msum)[zone_id]; thread=false)
    _eachzone(merge_zone!, msum, thread)
    return msum
end


### Propagation caches

function _mergeandtruncate!(::DictStorage, truncfunc::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    term_sum1 = mainsum(prop_cache)
    term_sum2 = auxsum(prop_cache)

    # merge the smaller into the larger
    if length(term_sum1) < length(term_sum2)
        term_sum2, term_sum1 = term_sum1, term_sum2
    end

    if _hasdictinternals(storage(term_sum1)) && _hasdictinternals(storage(term_sum2))
        _mergewith_internals!(storage(term_sum1), storage(term_sum2))
    else
        mergewith!(mergefunc, storage(term_sum1), storage(term_sum2))
    end
    empty!(term_sum2)

    setmainsum!(prop_cache, term_sum1)
    setauxsum!(prop_cache, term_sum2)

    return _truncate!(truncfunc, prop_cache; thread)
end

# The generic flatmap path writes through `add!`, whose contract already combines equal terms.
# There may still be a caller-owned auxiliary sum, so fold it into the main sum and clear it.
function _mergeandtruncate!(::StorageType, truncfunc::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    add!(mainsum(prop_cache), auxsum(prop_cache))
    empty!(auxsum(prop_cache))
    return _truncate!(truncfunc, prop_cache; thread)
end

# An array merges the cheapest way its state allows: not at all when the sorted head is all there
# is, by a sort of the tail alone when the head covers most of the array, and by a sort of
# everything otherwise. On the CPU the merged terms are truncated as they are written.
function _mergeandtruncate!(::ArrayStorage, truncfunc::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    if isempty(prop_cache)
        return prop_cache
    end

    # nothing to merge, but the terms may have been rescaled since they were last truncated
    n_sorted, n_total = _sortedandactive(prop_cache)
    if n_sorted == n_total
        return _truncate!(truncfunc, prop_cache; thread)
    end

    return _mergeandtruncatebysorting!(truncfunc, prop_cache, n_sorted, n_total; thread)
end

# The tail merge and the one-walk reduction of the sorted array are scalar code, so an array
# elsewhere than the CPU sorts everything, reduces the runs of equal terms in flagging passes, and
# truncates after.
function _mergeandtruncatebysorting!(truncfunc::F, prop_cache::AbstractPropagationCache, n_sorted::Int, n_total::Int; thread::Bool=true) where {F}
    on_cpu = _iscpuarray(prop_cache)

    if on_cpu && _tailmergepays(n_sorted, n_total)
        return _sortedtailmergeandtruncate!(truncfunc, prop_cache; thread)
    end

    sortterms!(prop_cache; thread)
    if on_cpu
        return _deduplicateandtruncatecpu!(truncfunc, prop_cache; thread)
    end

    _deduplicateflagged!(prop_cache; thread)
    setsortedprefix!(mainsum(prop_cache), activesize(prop_cache))
    return _truncate!(truncfunc, prop_cache; thread)
end

# the sorted head and the active size of an array, which no write leaves the head reaching past
function _sortedandactive(prop_cache::AbstractPropagationCache)
    n_sorted = sortedprefix(mainsum(prop_cache))
    n_total = activesize(prop_cache)

    if !(0 <= n_sorted <= n_total)
        # something went wrong. Set to zero and do a full merge.
        setsortedprefix!(mainsum(prop_cache), 0)
        n_sorted = 0
    end

    return n_sorted, n_total
end

# whether the sorted head covers enough of the array for a sort of the tail alone to beat a sort of
# everything
_tailmergepays(n_sorted::Int, n_total::Int) = n_sorted / n_total > _TAILMERGE_SORTEDPREFIX_FRACTION

# the outboxes are the auxiliary sums of a multi sum, and a gate may have left terms in them
function _mergeandtruncate!(::MultiSumStorage, truncfunc::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    merge_zone!(owner) = mergeandtruncate!(truncfunc, _deliverto!(prop_cache, owner); thread=false)
    _eachzone(merge_zone!, prop_cache, thread)
    return _syncsums!(prop_cache)
end
