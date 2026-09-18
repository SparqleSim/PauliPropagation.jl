"""
    flatmap(f, term_sum::AbstractTermSum; thread=true)
    flatmap(f, prop_cache::AbstractPropagationCache; thread=true)

Replace every active term and coefficient pair by the pairs `f(term, coefficient)` returns, on a copy.
See `flatmap!`.
"""
flatmap(f::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F} =
    flatmap!(f, deepcopy(prop_cache); thread)

flatmap(f::F, term_sum::AbstractTermSum; thread::Bool=true) where {F} =
    extractsum!(flatmap!(f, PropagationCache(deepcopy(term_sum)); thread))

"""
    flatmap!(f, prop_cache::AbstractPropagationCache; thread=true)

Replace every active term and coefficient pair by the pairs `f(term, coefficient)` returns,
a tuple of `(term, coefficient)` pairs or another iterable of them, such as what `apply` returns for a gate.
The pairs are not merged: a dictionary merges them as it adds them, an array holds duplicates until `merge!`,
and a multi sum delivers each pair to the zone that owns it and leaves the merging to `merge!` as well.
Several tasks on an array call `f` once to count the pairs and once to write them, so `f` must return the same pairs each time.
Only a cache is transformed in place, because the pairs need room of their own; a term sum takes `flatmap`.
"""
flatmap!(f::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F} =
    _flatmap!(StorageType(prop_cache), f, prop_cache; thread)

# Written against the interface of a term sum alone, so any storage with `add!`, `empty!` and
# `swapsums!` takes this path.
function _flatmap!(::StorageType, f::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    output_sum = auxsum(prop_cache)
    isempty(output_sum) || empty!(output_sum)

    for (term, coefficient) in prop_cache
        for (new_term, new_coefficient) in f(term, coefficient)
            add!(output_sum, new_term, new_coefficient)
        end
    end

    empty!(mainsum(prop_cache))
    return swapsums!(prop_cache)
end


### Array storage

# The pairs are written into the auxiliary arrays in the order of the terms that made them.
function _flatmap!(::ArrayStorage, f::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    n_new = if _iscpuarray(terms(mainsum(prop_cache)))
        _flatmapcpu!(f, prop_cache; thread)
    else
        _flatmapflagged!(f, prop_cache; thread)
    end

    return _commitwrite!(prop_cache, n_new, 0)
end

# One task writes each pair as it makes it, growing the arrays as they fill. Several tasks first
# count what each of them makes, so that every task knows where to write.
function _flatmapcpu!(f::F, prop_cache; thread::Bool=true) where {F}
    n_old = activesize(prop_cache)
    task_partitioner, n_tasks = _preparetasks(n_old, thread)

    if n_tasks == 1
        return _flatmapserially!(f, prop_cache, n_old)
    end

    return _flatmapintasks!(f, prop_cache, task_partitioner, n_tasks)
end

function _flatmapserially!(f::F, prop_cache, n_old::Int) where {F}
    main_terms, main_coefficients, aux_terms, aux_coefficients = _mainauxarrays(prop_cache)
    write_pos = 1

    for ii in 1:n_old
        for (term, coefficient) in @inline f((@inbounds main_terms[ii]), (@inbounds main_coefficients[ii]))
            if write_pos > length(aux_terms)
                _growto!(prop_cache, write_pos)
                main_terms, main_coefficients, aux_terms, aux_coefficients = _mainauxarrays(prop_cache)
            end
            write_pos = _writeandadvance!(aux_terms, aux_coefficients, write_pos, term, coefficient, Val(true))
        end
    end

    return write_pos - 1
end

function _flatmapintasks!(f::F, prop_cache, task_partitioner, n_tasks::Int) where {F}
    counts = Vector{Int}(undef, n_tasks)

    function count_pairs!(task_id)
        main_terms, main_coefficients, aux_terms, aux_coefficients = _mainauxarrays(prop_cache)
        chunk = task_partitioner[task_id]
        counts[task_id] = _flatmapwrite!(f, aux_terms, aux_coefficients, 1, main_terms, main_coefficients, chunk.start, chunk.stop, Val(false))
    end
    _eachtask(count_pairs!, n_tasks)

    offsets = _offsetsfromcounts(counts)
    n_new = offsets[end] - 1
    _growto!(prop_cache, n_new)

    function write_pairs!(task_id)
        main_terms, main_coefficients, aux_terms, aux_coefficients = _mainauxarrays(prop_cache)
        chunk = task_partitioner[task_id]
        _flatmapwrite!(f, aux_terms, aux_coefficients, offsets[task_id], main_terms, main_coefficients, chunk.start, chunk.stop, Val(true))
    end
    _eachtask(write_pairs!, n_tasks)

    return n_new
end

# Walks terms[lo:hi] under `f`, writing the pairs every term makes from `write_start` on into the
# output arrays, or only counting them on a dry run (`DoWrite` false). Returns the number of pairs.
@inline function _flatmapwrite!(f::F, output_terms, output_coefficients, write_start,
    terms, coefficients, lo, hi, ::Val{DoWrite}) where {F,DoWrite}

    write_pos = write_start

    @inbounds for ii in lo:hi
        for (term, coefficient) in @inline f(terms[ii], coefficients[ii])
            write_pos = _writeandadvance!(output_terms, output_coefficients, write_pos, term, coefficient, Val(DoWrite))
        end
    end

    return write_pos - write_start
end

# Every pass is an array kernel, so the arrays may live anywhere: count the pairs every term makes,
# turn the counts into the positions the pairs end at, then walk once more to write.
function _flatmapflagged!(f::F, prop_cache; thread::Bool=true) where {F}
    main_terms, main_coefficients, _, _ = _mainauxarrays(prop_cache)
    counts = activeindices(prop_cache)

    AK.foreachindex(counts; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do ii
        @inbounds counts[ii] = _countpairs(f, main_terms[ii], main_coefficients[ii])
    end
    AK.accumulate!(+, counts; init=0, max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)

    n_new = lastactiveindex(prop_cache)
    _growto!(prop_cache, n_new)

    main_terms, main_coefficients, aux_terms, aux_coefficients = _mainauxarrays(prop_cache)
    write_ends = activeindices(prop_cache)

    AK.foreachindex(write_ends; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do ii
        @inbounds begin
            write_pos = ii == 1 ? 1 : write_ends[ii-1] + 1
            for (term, coefficient) in @inline f(main_terms[ii], main_coefficients[ii])
                write_pos = _writeandadvance!(aux_terms, aux_coefficients, write_pos, term, coefficient, Val(true))
            end
        end
    end

    return n_new
end

@inline function _countpairs(f::F, term, coefficient) where {F}
    n = 0
    for _ in @inline f(term, coefficient)
        n += 1
    end
    return n
end


### Multi sum storage

# Every zone parks the pairs its terms make in its outbox, sorted by the zones that own them, and
# then takes delivery of the pairs the other zones made for it.
function _flatmap!(::MultiSumStorage, f::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    function scatter_zone!(zone_id)
        outbox = outboxes(prop_cache)[zone_id]
        zonecache = zonecaches(prop_cache)[zone_id]

        for (term, coefficient) in zonecache
            for (new_term, new_coefficient) in f(term, coefficient)
                push!(outbox, new_term, new_coefficient)
            end
        end

        empty!(zonecache)
        return
    end
    _eachzone(scatter_zone!, prop_cache, thread)

    deliver_to_zone!(owner) = foreach(
        outbox -> _deliver!(zonecaches(prop_cache)[owner], zones(outbox)[owner]),
        outboxes(prop_cache),
    )
    _eachzone(deliver_to_zone!, prop_cache, thread)

    return _syncsums!(prop_cache)
end
