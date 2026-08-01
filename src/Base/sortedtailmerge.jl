###
##
# Sort-tail-and-merge: cheaper than a full re-sort when active terms are already
# [sorted, duplicate-free head][unsorted tail]. The tail may contain duplicates against the head or
# itself; collisions are merged via mergefunc. Gate-agnostic; callers just need the head sorted.
##
###

# _merge!() only dispatches to sortedtailmerge! when:
# sortedprefix(term_sum) / length(term_sum) > _TAILMERGE_SORTEDPREFIX_FRACTION
# below it, a full re-sort is cheaper.
const _TAILMERGE_SORTEDPREFIX_FRACTION = 0.4

"""
    sortedtailmerge!(prop_cache::AbstractPropagationCache; thread::Bool=true, truncfunc=nothing)

Merges the sorted head against the unsorted tail (see file header) and updates
`activesize`/`sortedprefix`. Set `thread=false` to force sequential execution.

`truncfunc(term, merged_coeff)`, if given, is applied to every merged coefficient and drops the term
if it returns `true`, which truncates the sum without a separate pass over it. Judging a term on the
coefficient it merged into, rather than on each contribution separately, is what keeps a
contribution too small to stand alone from being lost when it would have cancelled a larger one.
"""
function sortedtailmerge!(prop_cache::AbstractPropagationCache; thread::Bool=true, truncfunc=nothing)
    n_old = sortedprefix(mainsum(prop_cache))
    n_new = activesize(prop_cache)
    n_tail = n_new - n_old
    if n_tail == 0
        setsortedprefix!(mainsum(prop_cache), n_old)
        return prop_cache
    end

    main_terms, main_coeffs, aux_terms, aux_coeffs = _mainauxarrays(prop_cache)

    # sort the tail
    # interestingly sort! of StructArray is not faster, potentially due to heavier writes
    unsorted_tail_terms = view(main_terms, n_old+1:n_new)
    unsorted_tail_coeffs = view(main_coeffs, n_old+1:n_new)
    tail_perm = view(indices(prop_cache), 1:n_tail)
    AK.sortperm!(tail_perm, unsorted_tail_terms; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)

    tail_terms, tail_coeffs = _tailscratch(aux_terms, aux_coeffs, n_new, n_tail, main_terms, main_coeffs)
    permuteviaindices!(tail_terms, tail_coeffs, unsorted_tail_terms, unsorted_tail_coeffs, tail_perm; thread)

    return _mergesortedhead!(prop_cache, aux_terms, aux_coeffs, main_terms, main_coeffs, n_old,
        tail_terms, tail_coeffs, n_tail, truncfunc, thread)
end

"""
    _tailscratch(aux_terms, aux_coeffs, n_new, n_tail, main_terms, main_coeffs)

Scratch space for a sorted copy of the tail. The merge only ever writes into `aux[1:n_new]`, so any
capacity beyond that is free; otherwise allocate.
"""
function _tailscratch(aux_terms, aux_coeffs, n_new::Int, n_tail::Int, main_terms, main_coeffs)
    if length(aux_terms) - n_new >= n_tail
        return view(aux_terms, n_new+1:n_new+n_tail), view(aux_coeffs, n_new+1:n_new+n_tail)
    end
    return similar(main_terms, n_tail), similar(main_coeffs, n_tail)
end

"""
    _mergesortedhead!(prop_cache, aux_terms, aux_coeffs, main_terms, main_coeffs, n_old,
                      tail_terms, tail_coeffs, n_tail, truncfunc, thread)

Merge the sorted head `main[1:n_old]` against an already sorted tail into `aux`, and commit. Shared
by every tail merge; they differ only in how they sorted the tail.
"""
function _mergesortedhead!(prop_cache, aux_terms, aux_coeffs, main_terms, main_coeffs, n_old::Int,
    tail_terms, tail_coeffs, n_tail::Int, truncfunc, thread::Bool)

    task_partitioner, n_tasks = _preparetasks(n_old, thread)

    if n_tasks == 1
        merged_count = _tailmerge_write!(aux_terms, aux_coeffs, 1,
            main_terms, main_coeffs, 1, n_old, tail_terms, tail_coeffs, 1, n_tail, truncfunc, Val(true))
    else
        # slice and partition the two-pointer merge across threads
        tail_bounds_per_task = Vector{Int}(undef, n_tasks + 1)
        tail_bounds_per_task[1] = 1
        tail_bounds_per_task[n_tasks+1] = n_tail + 1
        @inbounds for task_id in 1:(n_tasks-1)
            head_chunk_boundary_term = main_terms[task_partitioner[task_id].stop]
            tail_bounds_per_task[task_id+1] = searchsortedlast(tail_terms, head_chunk_boundary_term) + 1
        end

        # dry run: each task counts its own merged output size (unknown ahead of time due to collisions)
        merged_counts_per_task = Vector{Int}(undef, n_tasks)
        AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
            head_range = task_partitioner[task_id]
            merged_counts_per_task[task_id] = _tailmerge_write!(aux_terms, aux_coeffs, 1,
                main_terms, main_coeffs, head_range.start, head_range.stop,
                tail_terms, tail_coeffs, tail_bounds_per_task[task_id], tail_bounds_per_task[task_id+1] - 1, truncfunc, Val(false))
        end

        # prefix sum over the per-task counts gives each task its exact final write offset
        write_offsets_per_task = _offsetsfromcounts(merged_counts_per_task)
        merged_count = write_offsets_per_task[end] - 1

        # real pass: each task redoes the same merge, now writing directly into its final position
        AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
            head_range = task_partitioner[task_id]
            _tailmerge_write!(aux_terms, aux_coeffs, write_offsets_per_task[task_id],
                main_terms, main_coeffs, head_range.start, head_range.stop,
                tail_terms, tail_coeffs, tail_bounds_per_task[task_id], tail_bounds_per_task[task_id+1] - 1, truncfunc, Val(true))
        end
    end

    _commitwrite!(prop_cache, merged_count, merged_count)

    return prop_cache
end

# Folds the run of terms in tail_terms[tail_j:tail_hi] equal to tail_term into `seed` via
# mergefunc, in order. Returns (merged_coeff, run_length, next_tail_j).
@inline function _foldtailrun(tail_terms, tail_coeffs, tail_j, tail_hi, tail_term, seed)
    merged_coeff = seed
    run_length = 0
    @inbounds while tail_j <= tail_hi && tail_terms[tail_j] == tail_term
        merged_coeff = mergefunc(merged_coeff, tail_coeffs[tail_j])
        tail_j += 1
        run_length += 1
    end
    return merged_coeff, run_length, tail_j
end

# Two-pointer merge of a sorted head against a sorted tail that may contain duplicate runs (against
# the head or itself); collisions combined via mergefunc. Writes from out_start when DoWrite,
# otherwise only counts. `truncfunc`, if given, is applied to every merged coefficient, so that a
# term is judged on the coefficient it ends up with rather than on the parts it was built from.
# Returns the output element count.
@inline function _tailmerge_write!(out_terms, out_coeffs, out_start,
    head_terms, head_coeffs, head_lo, head_hi,
    tail_terms, tail_coeffs, tail_lo, tail_hi,
    truncfunc::F, ::Val{DoWrite}) where {F,DoWrite}

    head_i = head_lo
    tail_j = tail_lo
    write_pos = out_start
    @inbounds while head_i <= head_hi && tail_j <= tail_hi
        head_term = head_terms[head_i]
        tail_term = tail_terms[tail_j]
        if head_term == tail_term
            # collision: merge the head term with the *entire run* of equal tail terms
            merged_coeff, _, tail_j = _foldtailrun(tail_terms, tail_coeffs, tail_j, tail_hi, tail_term, head_coeffs[head_i])
            write_pos = _writekept!(out_terms, out_coeffs, write_pos, head_term, merged_coeff, truncfunc, Val(DoWrite))
            head_i += 1
        elseif head_term < tail_term
            write_pos = _writekept!(out_terms, out_coeffs, write_pos, head_term, head_coeffs[head_i], truncfunc, Val(DoWrite))
            head_i += 1
        else
            # tail term has no match in the head (yet): merge its own run of duplicates first
            merged_coeff, _, tail_j = _foldtailrun(tail_terms, tail_coeffs, tail_j + 1, tail_hi, tail_term, tail_coeffs[tail_j])
            write_pos = _writekept!(out_terms, out_coeffs, write_pos, tail_term, merged_coeff, truncfunc, Val(DoWrite))
        end
    end
    @inbounds while head_i <= head_hi
        write_pos = _writekept!(out_terms, out_coeffs, write_pos, head_terms[head_i], head_coeffs[head_i], truncfunc, Val(DoWrite))
        head_i += 1
    end
    @inbounds while tail_j <= tail_hi
        tail_term = tail_terms[tail_j]
        merged_coeff, _, tail_j = _foldtailrun(tail_terms, tail_coeffs, tail_j + 1, tail_hi, tail_term, tail_coeffs[tail_j])
        write_pos = _writekept!(out_terms, out_coeffs, write_pos, tail_term, merged_coeff, truncfunc, Val(DoWrite))
    end

    return write_pos - out_start
end

# `_writeandadvance!` past `truncfunc`; with no truncfunc the test is compiled away
@inline function _writekept!(out_terms, out_coeffs, write_pos, term, coeff, truncfunc::F, ::Val{DoWrite}) where {F,DoWrite}
    if truncfunc !== nothing && truncfunc(term, coeff)
        return write_pos
    end
    return _writeandadvance!(out_terms, out_coeffs, write_pos, term, coeff, Val(DoWrite))
end
