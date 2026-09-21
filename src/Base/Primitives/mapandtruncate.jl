"""
    mapandtruncate!(mapfunc, truncfunc, prop_cache::AbstractPropagationCache; thread=true)

Map every active coefficient with `mapfunc(term, coefficient)` and discard the mapped pair when
`truncfunc(term, new_coefficient)` returns `true`. This is a propagation-cache primitive: unlike
`mapcoeffsbypair!`, it may use the cache's auxiliary storage to compact the active terms.
`mapandtruncate!` itself constructs the `Kept(new_coefficient)` and `Truncated()` outcomes, so its
callers only supply the mapping and truncation functions.

It is intended for a gate that deterministically rescales terms and can decide, from the new
coefficient, that a term is truncated. Array storage performs the mapping and compaction in one walk;
other storage types use their corresponding direct or cache-aware implementation.

On a multithreaded CPU array, both functions are called once to count retained terms and again to
write them. They must therefore be deterministic, replayable, and safe to call concurrently.
"""
mapandtruncate!(mapfunc::F, truncfunc::G, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F,G} =
    _mapandtruncate!(StorageType(prop_cache), mapfunc, truncfunc, prop_cache; thread)

"""
    Truncated()

The internal outcome of `mapandtruncate!` for an input term whose mapped coefficient is truncated.
"""
struct Truncated end

@inline function _mapandtruncateoutcome(mapfunc::F, truncfunc::G, term, coefficient) where {F,G}
    new_coefficient = @inline mapfunc(term, coefficient)
    truncated = @inline truncfunc(term, new_coefficient)
    return truncated ? Truncated() : Kept(new_coefficient)
end

# A cache with no specialized storage implementation can still express the operation through the
# general expansion primitive. `flatmap!` also combines duplicate terms for dictionary-like sums.
function _mapandtruncate!(::StorageType, mapfunc::F, truncfunc::G, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F,G}
    function map_or_drop(term, coefficient)
        outcome = _mapandtruncateoutcome(mapfunc, truncfunc, term, coefficient)
        return outcome isa Truncated ? () : ((term, outcome.coefficient),)
    end

    return flatmap!(map_or_drop, prop_cache; thread)
end


### Dictionary storage

# Updating values while iterating is supported by `Dict`; defer deletions until the walk is over.
function _mapandtruncate!(::DictStorage, mapfunc::F, truncfunc::G, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F,G}
    dict = storage(mainsum(prop_cache))
    dropped = Vector{keytype(dict)}()

    for (term, coefficient) in dict
        outcome = _mapandtruncateoutcome(mapfunc, truncfunc, term, coefficient)

        if outcome isa Truncated
            push!(dropped, term)
        else
            new_coefficient = outcome.coefficient
            new_coefficient !== coefficient && (dict[term] = new_coefficient)
        end
    end

    for term in dropped
        delete!(dict, term)
    end

    return prop_cache
end


### Array storage

# The retained pairs are compacted in the order they had, so the sorted prefix survives as the
# number of retained terms it held.
function _mapandtruncate!(::ArrayStorage, mapfunc::F, truncfunc::G, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F,G}
    isempty(prop_cache) && return prop_cache

    if _iscpuarray(prop_cache)
        return _mapandtruncatecpu!(mapfunc, truncfunc, prop_cache; thread)
    end

    return _mapandtruncateflagged!(mapfunc, truncfunc, prop_cache; thread)
end

# One task compacts in place, since it writes at or behind the pair it just read. Several tasks
# first count what each of them retains, then write their parts into the auxiliary arrays.
function _mapandtruncatecpu!(mapfunc::F, truncfunc::G, prop_cache; thread::Bool=true) where {F,G}
    n = activesize(prop_cache)
    n_sorted = sortedprefix(mainsum(prop_cache))
    task_partitioner, n_tasks = _preparetasks(n, thread)
    main_terms, main_coefficients, aux_terms, aux_coefficients = _mainauxarrays(prop_cache)

    if n_tasks == 1
        n_kept, n_sorted_kept = _mapandtruncatewrite!(mapfunc, truncfunc, main_terms, main_coefficients, 1,
            main_terms, main_coefficients, 1, n, n_sorted, Val(true))
        setactivesize!(prop_cache, n_kept)
        setsortedprefix!(mainsum(prop_cache), n_sorted_kept)
        return prop_cache
    end

    kept_counts = Vector{Int}(undef, n_tasks)
    sorted_kept_counts = Vector{Int}(undef, n_tasks)

    function count_kept!(task_id)
        chunk = task_partitioner[task_id]
        kept_counts[task_id], sorted_kept_counts[task_id] =
            _mapandtruncatewrite!(mapfunc, truncfunc, aux_terms, aux_coefficients, 1, main_terms,
                main_coefficients, chunk.start, chunk.stop, n_sorted, Val(false))
    end
    _eachtask(count_kept!, n_tasks)

    offsets = _offsetsfromcounts(kept_counts)

    # a task keeps at most as many pairs as its chunk holds, so it never writes past the chunk's end
    written = Vector{Int}(undef, n_tasks)
    function write_kept!(task_id)
        chunk = task_partitioner[task_id]
        written[task_id], _ = _mapandtruncatewrite!(mapfunc, truncfunc, aux_terms, aux_coefficients, offsets[task_id], main_terms,
            main_coefficients, chunk.start, chunk.stop, n_sorted, Val(true))
    end
    _eachtask(write_kept!, n_tasks)

    if written != kept_counts
        _throwreplaymismatch()
    end
    return _commitwrite!(prop_cache, offsets[end] - 1, sum(sorted_kept_counts))
end

# Walk terms[lo:hi] under `mapfunc` and `truncfunc`, writing every retained term from `write_start`
# on into the output arrays, or only counting on a dry run (`DoWrite` false). Returns the number of
# retained terms and how many of them came from the first `n_sorted`.
@inline function _mapandtruncatewrite!(mapfunc::F, truncfunc::G, output_terms, output_coefficients, write_start,
    terms, coefficients, lo, hi, n_sorted, ::Val{DoWrite}) where {F,G,DoWrite}

    write_pos = write_start
    n_sorted_kept = 0

    @inbounds for ii in lo:hi
        outcome = _mapandtruncateoutcome(mapfunc, truncfunc, terms[ii], coefficients[ii])
        outcome isa Truncated && continue

        write_pos = _writeandadvance!(output_terms, output_coefficients, write_pos, terms[ii], outcome.coefficient, Val(DoWrite))
        ii <= n_sorted && (n_sorted_kept += 1)
    end

    return write_pos - write_start, n_sorted_kept
end

# Every pass is an array kernel, so the arrays may live anywhere: the coefficients are mapped in
# place, and the flags then compact the retained terms.
function _mapandtruncateflagged!(mapfunc::F, truncfunc::G, prop_cache; thread::Bool=true) where {F,G}
    active_flags = activeflags(prop_cache)
    active_terms = activeterms(prop_cache)
    active_coefficients = activecoeffs(prop_cache)

    AK.foreachindex(active_flags; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do ii
        @inbounds begin
            outcome = _mapandtruncateoutcome(mapfunc, truncfunc, active_terms[ii], active_coefficients[ii])
            active_flags[ii] = !(outcome isa Truncated)
            if outcome isa Kept
                active_coefficients[ii] = outcome.coefficient
            end
        end
    end

    return filterviaflags!(prop_cache; thread)
end


### Multi sum storage

function _mapandtruncate!(::MultiSumStorage, mapfunc::F, truncfunc::G, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F,G}
    mapandtruncate_zone!(zone_id) = mapandtruncate!(mapfunc, truncfunc, zonecaches(prop_cache)[zone_id]; thread=false)
    _eachzone(mapandtruncate_zone!, prop_cache, thread)
    return _syncsums!(prop_cache)
end
