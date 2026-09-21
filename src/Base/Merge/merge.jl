### MERGE

# Default merge function for coefficients: simple addition
# Can be overloaded for different coefficient types.
mergefunc(coeff1, coeff2) = coeff1 + coeff2

Base.merge(obj) = merge!(deepcopy(obj))

function Base.merge!(term_sum::TS) where TS<:AbstractTermSum
    return _merge!(StorageType(term_sum), term_sum)
end

function _merge!(::DictStorage, term_sum::AbstractTermSum)
    # Dicts are always already merged
    return term_sum
end

function _merge!(::ArrayStorage, term_sum::AbstractTermSum)
    prop_cache = PropagationCache(term_sum)

    merge!(prop_cache)

    # extracts the original input term sum
    return extractsum!(prop_cache, term_sum)
end

"""
    merge!(prop_cache::AbstractPropagationCache; thread=true)

Merge the terms the last gate left in the auxiliary sum, or appended past the sorted prefix, into the main sum, combining equal terms with `mergefunc`.
"""
function Base.merge!(prop_cache::AbstractPropagationCache; thread::Bool=true, kwargs...)
    return _merge!(StorageType(prop_cache), prop_cache; thread)
end

function _merge!(::DictStorage, prop_cache::AbstractPropagationCache; thread::Bool=true)
    term_sum1 = mainsum(prop_cache)
    term_sum2 = auxsum(prop_cache)

    # merge the smaller into the larger 
    if length(term_sum1) < length(term_sum2)
        term_sum2, term_sum1 = term_sum1, term_sum2
    end

    # mergefunc can be overloaded for different coefficient types
    mergewith!(mergefunc, storage(term_sum1), storage(term_sum2))
    empty!(term_sum2)

    setmainsum!(prop_cache, term_sum1)
    setauxsum!(prop_cache, term_sum2)

    return prop_cache
end

# The generic flatmap path writes through `add!`, whose contract already combines equal terms.
# There may still be a caller-owned auxiliary sum, so fold it into the main sum and clear it.
function _merge!(::StorageType, prop_cache::AbstractPropagationCache; thread::Bool=true)
    add!(mainsum(prop_cache), auxsum(prop_cache))
    empty!(auxsum(prop_cache))
    return prop_cache
end

# An array merges the cheapest way its state allows: not at all when the sorted head is all there
# is, by a sort of the tail alone when the head covers most of the array, and by a sort of
# everything otherwise. `_mergeandtruncate!` makes the same choices, truncating as the merged terms
# are written.
function _merge!(::ArrayStorage, prop_cache::AbstractPropagationCache; thread::Bool=true)
    if isempty(prop_cache)
        return prop_cache
    end

    n_sorted, n_total = _sortedandactive(prop_cache)
    if n_sorted == n_total
        return prop_cache
    end

    # the sorts run through AcceleratedKernels, which starts tasks of its own
    merge_by_sorting!() = _mergebysorting!(prop_cache, n_sorted, n_total; thread)
    return _with_threads_freed_for(merge_by_sorting!, thread)
end

# The tail merge and the one-walk reduction of the sorted array are scalar code, so an array
# elsewhere than the CPU sorts everything and reduces the runs of equal terms in flagging passes.
function _mergebysorting!(prop_cache::AbstractPropagationCache, n_sorted::Int, n_total::Int; thread::Bool=true)
    on_cpu = _iscpuarray(prop_cache)

    if on_cpu && _tailmergepays(n_sorted, n_total)
        return sortedtailmerge!(prop_cache; thread)
    end

    sortterms!(prop_cache; thread)
    if on_cpu
        return _deduplicatecpu!(prop_cache; thread)
    end

    _deduplicate!(prop_cache; thread)
    setsortedprefix!(mainsum(prop_cache), activesize(prop_cache))
    return prop_cache
end

# the sorted head and the active size of an array, which no write leaves the head reaching past
function _sortedandactive(prop_cache::AbstractPropagationCache)
    n_sorted = sortedprefix(mainsum(prop_cache))
    n_total = activesize(prop_cache)

    if n_sorted > n_total
        # something went wrong. Set to zero and do a full merge.
        setsortedprefix!(mainsum(prop_cache), 0)
        n_sorted = 0
    end

    return n_sorted, n_total
end

# whether the sorted head covers enough of the array for a sort of the tail alone to beat a sort of
# everything
_tailmergepays(n_sorted::Int, n_total::Int) = n_sorted / n_total > _TAILMERGE_SORTEDPREFIX_FRACTION

_merge!(::MultiSumStorage, msum::AbstractTermSum) = (foreach(merge!, zones(msum)); msum)

# the outboxes are the auxiliary sums of a multi sum, and a gate may have left terms in them
function _merge!(::MultiSumStorage, prop_cache::AbstractPropagationCache; thread::Bool=true)
    merge_zone!(owner) = merge!(_deliverto!(prop_cache, owner); thread=false)
    _eachzone(merge_zone!, prop_cache, thread)
    return _syncsums!(prop_cache)
end


function _deduplicate!(prop_cache::AbstractPropagationCache; thread::Bool=true)

    _flaggroupbegin!(prop_cache; thread)

    flagstoindices!(prop_cache; thread)

    _mergegroups!(prop_cache; thread)

    return prop_cache
end

# flags if term at i is different from term at i-1
function _flaggroupbegin!(prop_cache::AbstractPropagationCache; thread::Bool=true)
    term_view = activeterms(prop_cache)
    flags_view = activeflags(prop_cache)

    AK.foreachindex(term_view; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do ii
        if ii == 1
            flags_view[ii] = true
        else
            flags_view[ii] = term_view[ii] != term_view[ii-1]
        end
    end
    return prop_cache
end

# Given flagged group beginnings, merge the groups.
function _mergegroups!(prop_cache::AbstractPropagationCache; thread::Bool=true)

    term_view = activeterms(prop_cache)
    coeffs = activecoeffs(prop_cache)
    aux_terms = activeauxterms(prop_cache)
    aux_coeffs = activeauxcoeffs(prop_cache)
    flags = activeflags(prop_cache)
    indices = activeindices(prop_cache)
    active_size = activesize(prop_cache)

    AK.foreachindex(term_view; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do ii
        # if this is the start of a new group
        if flags[ii]
            # end index is the before the next flag or the end of the array
            end_idx = ii
            while end_idx < active_size && !flags[end_idx+1]
                end_idx += 1
            end

            # Sum the values in the range.
            CT = typeof(coeffs[ii])
            merged_coeff = zero(CT)
            for jj in ii:end_idx
                # mergefunc can be overloaded for different coefficient types
                merged_coeff = mergefunc(merged_coeff, coeffs[jj])
            end

            aux_terms[indices[ii]] = term_view[ii]
            aux_coeffs[indices[ii]] = merged_coeff
        end
    end

    # swap terms and aux_terms
    swapsums!(prop_cache)

    setactivesize!(prop_cache, lastactiveindex(prop_cache))

    return prop_cache
end
