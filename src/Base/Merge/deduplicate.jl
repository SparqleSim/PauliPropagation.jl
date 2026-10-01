###
##
# Reducing every run of equal terms of a sorted array to one term: in one walk on the CPU, which
# truncates the merged terms as it writes them when asked to, or in flagging passes anywhere else.
##
###

### One walk on the CPU

# Reduces the runs of equal terms of the sorted active array, dropping the pairs `truncfunc` rejects
# as they are written when there is one. One task compacts in place, since it writes at or behind
# the run it just read. Several tasks first count what each of them keeps, then write their parts
# into the auxiliary arrays.
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

    if written_counts != kept_counts
        _throwreplaymismatch()
    end

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

        write_pos = _writekept!(output_terms, output_coefficients, write_pos, term, merged_coefficient, truncfunc, Val(DoWrite))
    end

    return write_pos - write_start
end


### Flagging passes

# every pass is an array kernel, so the arrays may live anywhere
function _deduplicateflagged!(prop_cache::AbstractPropagationCache; thread::Bool=true)
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

            # mergefunc can be overloaded for different coefficient types
            merged_coeff = coeffs[ii]
            for jj in ii+1:end_idx
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
