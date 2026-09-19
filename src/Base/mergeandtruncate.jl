"""
    mergeandtruncate!(truncfunc, term_sum::AbstractTermSum; thread=true)
    mergeandtruncate!(truncfunc, prop_cache::AbstractPropagationCache; thread=true)

Merge all equal active terms, then drop every `(term, coefficient)` pair for which `truncfunc(term, coefficient)` returns `true`.
`truncfunc` sees the merged coefficient, so contributions can still cancel before a term is judged,
and it is applied even when nothing needs merging, since a gate may have rescaled the terms without creating any.
An array on the CPU truncates as it writes the merged terms, every other storage merges and then truncates.
Several tasks on an array call `truncfunc` once to count the terms and once to write them, so it must return the same for the same pair each time.
"""
function mergeandtruncate!(truncfunc::F, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) where {F}
    return _mergeandtruncate!(StorageType(thing), truncfunc, thing; thread)
end

# `mergeandtruncate!`, or `merge!` when there is nothing to truncate by; the fused tail merges fall back on it
function _mergeandtruncateby!(truncfunc::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    if truncfunc === nothing
        return merge!(prop_cache; thread)
    end
    return mergeandtruncate!(truncfunc, prop_cache; thread)
end


### Generic storage

# a term sum merges through a propagation cache of its own
function _mergeandtruncate!(::StorageType, truncfunc::F, term_sum::AbstractTermSum; thread::Bool=true) where {F}
    prop_cache = PropagationCache(term_sum)
    mergeandtruncate!(truncfunc, prop_cache; thread)
    return extractsum!(prop_cache, term_sum)
end

function _mergeandtruncate!(::StorageType, truncfunc::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    merge!(prop_cache; thread)
    truncate!(truncfunc, prop_cache; thread)
    return prop_cache
end


### Dictionary storage

# a dictionary is merged by construction
function _mergeandtruncate!(::DictStorage, truncfunc::F, term_sum::AbstractTermSum; thread::Bool=true) where {F}
    truncate!(truncfunc, term_sum; thread)
    return term_sum
end


### Array storage

function _mergeandtruncate!(::ArrayStorage, truncfunc::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    if isempty(prop_cache)
        return prop_cache
    end

    n_total = activesize(prop_cache)
    n_sorted = sortedprefix(mainsum(prop_cache))

    if n_sorted > n_total
        # something went wrong. Set to zero and do a full merge.
        setsortedprefix!(mainsum(prop_cache), 0)
        n_sorted = 0
    end

    # nothing to merge, but the terms may have been rescaled since they were last truncated
    if n_sorted == n_total
        return truncate!(truncfunc, prop_cache; thread)
    end

    if n_sorted / n_total > _TAILMERGE_SORTEDPREFIX_FRACTION && _iscpuarray(terms(mainsum(prop_cache)))
        # the sorted head covers most of the array: sort just the unsorted tail and merge it in,
        # truncating as the merged terms are written
        return _sortedtailmerge!(truncfunc, prop_cache; thread)
    end

    # fallback: sort everything, then reduce and truncate each run of equal terms in one walk on
    # the CPU, or in a merge pass and a truncation pass anywhere else
    sortterms!(prop_cache; thread)

    if _iscpuarray(terms(mainsum(prop_cache)))
        _deduplicateandtruncatecpu!(truncfunc, prop_cache; thread)
    else
        _deduplicate!(prop_cache; thread)
        setsortedprefix!(mainsum(prop_cache), activesize(prop_cache))
        truncate!(truncfunc, prop_cache; thread)
    end

    return prop_cache
end

# Reduces complete runs of equal terms. One task compacts in place, since it writes at or behind the
# run it just read. Several tasks first count what each of them keeps, then write their parts into
# the auxiliary arrays.
function _deduplicateandtruncatecpu!(truncfunc::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    n = activesize(prop_cache)

    main_terms, main_coefficients, aux_terms, aux_coefficients = _mainauxarrays(prop_cache)
    task_partitioner, n_tasks = _preparetasks(n, thread)

    if n_tasks == 1
        n_kept = _deduplicatetruncatewrite!(truncfunc,
            main_terms, main_coefficients, 1,
            main_terms, main_coefficients, 1, n, Val(true))
        setactivesize!(prop_cache, n_kept)
        setsortedprefix!(mainsum(prop_cache), n_kept)
        return prop_cache
    end

    input_bounds = _completegroupbounds(view(main_terms, 1:n), task_partitioner, n_tasks)
    kept_counts = Vector{Int}(undef, n_tasks)

    function count_kept_groups!(task_id)
        lo = input_bounds[task_id]
        hi = input_bounds[task_id+1] - 1
        kept_counts[task_id] = _deduplicatetruncatewrite!(truncfunc,
            aux_terms, aux_coefficients, 1,
            main_terms, main_coefficients, lo, hi, Val(false))
    end
    _eachtask(count_kept_groups!, n_tasks)

    write_offsets = _offsetsfromcounts(kept_counts)
    written_counts = Vector{Int}(undef, n_tasks)

    function write_kept_groups!(task_id)
        lo = input_bounds[task_id]
        hi = input_bounds[task_id+1] - 1
        written_counts[task_id] = _deduplicatetruncatewrite!(truncfunc,
            aux_terms, aux_coefficients, write_offsets[task_id],
            main_terms, main_coefficients, lo, hi, Val(true))
    end
    _eachtask(write_kept_groups!, n_tasks)

    written_counts == kept_counts || _throwreplaymismatch()

    n_kept = write_offsets[end] - 1
    return _commitwrite!(prop_cache, n_kept, n_kept)
end

# Moves every task boundary past the run of equal terms it falls into, so that no run is reduced by
# two tasks. A run wider than a task leaves that task an empty range, which is harmless.
function _completegroupbounds(sorted_terms, task_partitioner, n_tasks::Int)
    bounds = Vector{Int}(undef, n_tasks + 1)
    bounds[1] = 1
    bounds[end] = length(sorted_terms) + 1

    @inbounds for task_id in 1:n_tasks-1
        nominal_stop = task_partitioner[task_id].stop
        boundary_term = sorted_terms[nominal_stop]
        bounds[task_id+1] = searchsortedlast(sorted_terms, boundary_term) + 1
    end

    return bounds
end

# Walks the runs of equal terms in terms[lo:hi], which is sorted and holds every run in full, writing
# each merged pair `truncfunc` keeps from `write_start` on, or only counting on a dry run (`DoWrite`
# false). Returns the number of pairs kept.
@inline function _deduplicatetruncatewrite!(truncfunc::F,
    output_terms, output_coefficients, write_start,
    input_terms, input_coefficients, lo, hi, ::Val{DoWrite}) where {F,DoWrite}

    write_pos = write_start
    read_pos = lo

    @inbounds while read_pos <= hi
        term = input_terms[read_pos]
        merged_coefficient = input_coefficients[read_pos]
        read_pos += 1

        while read_pos <= hi && input_terms[read_pos] == term
            merged_coefficient = mergefunc(merged_coefficient, input_coefficients[read_pos])
            read_pos += 1
        end

        if !(@inline truncfunc(term, merged_coefficient))
            write_pos = _writeandadvance!(output_terms, output_coefficients, write_pos,
                term, merged_coefficient, Val(DoWrite))
        end
    end

    return write_pos - write_start
end


### Multi-sum storage

# equal terms share a zone, so the zones merge on their own
function _mergeandtruncate!(::MultiSumStorage, truncfunc::F, msum::AbstractTermSum; thread::Bool=true) where {F}
    merge_zone!(zone_id) = mergeandtruncate!(truncfunc, zones(msum)[zone_id]; thread=false)
    _eachzone(merge_zone!, msum, thread)
    return msum
end

function _mergeandtruncate!(::MultiSumStorage, truncfunc::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    merge_zone!(owner) = mergeandtruncate!(truncfunc, _deliverto!(prop_cache, owner); thread=false)
    _eachzone(merge_zone!, prop_cache, thread)
    return _syncsums!(prop_cache)
end
