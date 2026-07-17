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

# Merge auxsum into mainsum
function Base.merge!(prop_cache::AbstractPropagationCache; kwargs...)

    prop_cache = _merge!(StorageType(prop_cache), prop_cache; kwargs...)

    return prop_cache
end

function _merge!(::DictStorage, prop_cache::AbstractPropagationCache; kwargs...)
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

function _merge!(::ArrayStorage, prop_cache::AbstractPropagationCache; thread::Bool=true, truncfunc=nothing, kwargs...)

    if isempty(prop_cache)
        return prop_cache
    end

    n_sorted = sortedprefix(mainsum(prop_cache))
    n_total = activesize(prop_cache)

    if n_sorted > n_total
        # something went wrong. Set to zero and do a full merge.
        setsortedprefix!(mainsum(prop_cache), 0)
        n_sorted = sortedprefix(mainsum(prop_cache))
    end

    if n_sorted / n_total > _TAILMERGE_SORTEDPREFIX_FRACTION && _iscpuarray(terms(mainsum(prop_cache)))
        # the sorted head covers most of the array: sort just the unsorted tail and merge it in
        # (CPU-only scalar code, hence the backing-array check -- GPU backends fall through to
        # the fully AK-portable path below instead)
        sortedtailmerge!(prop_cache; thread, truncfunc)
        return prop_cache
    end

    # fallback: sort everything
    # TODO: allow sorting kwargs?
    sortbyterm!(prop_cache; thread)

    _deduplicate!(prop_cache; thread, truncfunc)

    setsortedprefix!(mainsum(prop_cache), activesize(prop_cache))

    return prop_cache

end


function _deduplicate!(prop_cache::AbstractPropagationCache; thread::Bool=true, truncfunc=nothing)

    _flaggroupbegin!(prop_cache; thread)

    flagstoindices!(prop_cache; thread)

    _mergegroups!(prop_cache; thread, truncfunc)

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
function _mergegroups!(prop_cache::AbstractPropagationCache; thread::Bool=true, kwargs...)

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


