"""
    xorbranch(rule, term_sum::AbstractTermSum, mask; thread=true, truncfunc=nothing)
    xorbranch(rule, prop_cache::AbstractPropagationCache, mask; thread=true, truncfunc=nothing)

Branch every active term by `rule` on a copy, see `xorbranch!`.
"""
xorbranch(rule, thing::Union{AbstractTermSum,AbstractPropagationCache}, mask; kwargs...) =
    xorbranch!(rule, deepcopy(thing), mask; kwargs...)

"""
    xorbranch!(rule, term_sum::AbstractTermSum, mask; thread=true, truncfunc=nothing)
    xorbranch!(rule, prop_cache::AbstractPropagationCache, mask; thread=true, truncfunc=nothing)

Branch every active term by `rule` and merge the terms this creates.
`rule(term, coefficient)` returns `nothing` to leave the term alone, a coefficient to keep it with,
or a pair `(kept_coefficient, new_coefficient)` to keep it with the first and create the term `term ⊻ mask` with the second.
Every new term is the same `⊻ mask` away from its parent, which lets an array sum sort the new terms in without comparing them,
and a multi sum send each zone's new terms to a single other zone.
`truncfunc(term, coefficient)`, if given, drops terms once merging has settled their coefficients.
"""
xorbranch!(rule::F, thing::Union{AbstractTermSum,AbstractPropagationCache}, mask; thread::Bool=true, truncfunc=nothing) where {F} =
    _xorbranch!(StorageType(thing), rule, thing, mask; thread, truncfunc)

"""
    ruleat(rule, terms, coefficients, ii)

`rule(terms[ii], coefficients[ii])`, as the array kernels of `xorbranch!` call a rule.
A rule that can decide from a part of the term overloads this to read only that part,
and the coefficient only once it needs it.
The kernels keep `terms` alive for the whole walk, so a rule may read it through a pointer.
"""
@inline ruleat(rule::F, terms, coefficients, ii::Int) where {F} = @inline rule((@inbounds terms[ii]), (@inbounds coefficients[ii]))

# a term sum branches through a propagation cache of its own
function _xorbranch!(::StorageType, rule::F, term_sum::AbstractTermSum, mask; thread::Bool=true, truncfunc=nothing) where {F}
    prop_cache = PropagationCache(term_sum)
    xorbranch!(rule, prop_cache, mask; thread, truncfunc)
    return extractsum!(prop_cache, term_sum)
end

function _xorbranch!(::StorageType, rule::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true, truncfunc=nothing) where {F}
    function branch(term, coefficient)
        branched = rule(term, coefficient)
        branched === nothing && return ((term, coefficient),)

        if branched isa Tuple
            kept_coefficient, new_coefficient = branched
            return ((term, kept_coefficient), (term ⊻ mask, new_coefficient))
        end

        return ((term, branched),)
    end

    # `flatmap!` writes through `add!`, so its generic path has already combined equal terms.
    flatmap!(branch, prop_cache; thread)
    truncfunc === nothing || truncate!(truncfunc, prop_cache; thread)
    return prop_cache
end


### Dictionary storage

function _xorbranch!(::DictStorage, rule::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true, truncfunc=nothing) where {F}
    new_sum = auxsum(prop_cache)
    isempty(new_sum) || empty!(new_sum)

    _branchdict!(rule, mainsum(prop_cache), new_sum, mask)

    merge!(prop_cache; thread)
    truncfunc === nothing || truncate!(truncfunc, prop_cache; thread)
    return prop_cache
end

# Rescales the terms of `term_sum` in place and sets the terms they create in `new_sum`, which is
# empty. Two terms never create the same new term, so each is set rather than added.
function _branchdict!(rule::F, term_sum, new_sum, mask) where {F}
    for (term, coefficient) in term_sum
        branched = @inline rule(term, coefficient)
        branched === nothing && continue

        if branched isa Tuple
            kept_coefficient, new_coefficient = branched
            set!(term_sum, term, kept_coefficient)
            set!(new_sum, term ⊻ mask, new_coefficient)
        else
            set!(term_sum, term, branched)
        end
    end

    return term_sum
end


### Array storage

# The new terms are appended past the active terms in the order of their parents, and then sorted in.
function _xorbranch!(::ArrayStorage, rule::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true, truncfunc=nothing) where {F}
    n_old = activesize(prop_cache)
    n_old == 0 && return prop_cache

    sorted_before = sortedprefix(mainsum(prop_cache)) == n_old

    if _iscpuarray(terms(mainsum(prop_cache)))
        _branchcpu!(rule, prop_cache, mask; thread)
    else
        _branchflagged!(rule, prop_cache, mask; thread)
    end

    return xorsortedtailmerge!(prop_cache, mask, sorted_before; thread, truncfunc)
end

# One task writes each new term as it makes it, growing the arrays as they fill. Several tasks first
# count what each of them makes, so that every task knows where to write.
function _branchcpu!(rule::F, prop_cache, mask; thread::Bool=true) where {F}
    n_old = activesize(prop_cache)
    task_partitioner, n_tasks = _preparetasks(n_old, thread)

    n_new = if n_tasks == 1
        _branchserially!(rule, prop_cache, n_old, mask)
    else
        _branchintasks!(rule, prop_cache, n_old, task_partitioner, n_tasks, mask)
    end

    setactivesize!(prop_cache, n_old + n_new)
    return prop_cache
end

# A term makes at most one new term, so a walk never outruns the free slots it started with.
function _branchserially!(rule::F, prop_cache, n_old::Int, mask) where {F}
    write_pos = n_old + 1
    lo = 1

    while lo <= n_old
        _growto!(prop_cache, write_pos)
        main_terms, main_coefficients, _, _ = _mainauxarrays(prop_cache)

        hi = min(n_old, lo + capacity(prop_cache) - write_pos)
        write_pos += _branchwrite!(rule, main_terms, main_coefficients, write_pos, main_terms, main_coefficients, lo, hi, mask, Val(true))
        lo = hi + 1
    end

    return write_pos - n_old - 1
end

function _branchintasks!(rule::F, prop_cache, n_old::Int, task_partitioner, n_tasks::Int, mask) where {F}
    counts = Vector{Int}(undef, n_tasks)

    function count_new_terms!(task_id)
        main_terms, main_coefficients, _, _ = _mainauxarrays(prop_cache)
        chunk = task_partitioner[task_id]
        counts[task_id] = _branchwrite!(rule, main_terms, main_coefficients, 1, main_terms, main_coefficients, chunk.start, chunk.stop, mask, Val(false))
    end
    _eachtask(count_new_terms!, n_tasks)

    offsets = _offsetsfromcounts(counts)
    n_new = offsets[end] - 1
    _growto!(prop_cache, n_old + n_new)

    function write_new_terms!(task_id)
        main_terms, main_coefficients, _, _ = _mainauxarrays(prop_cache)
        chunk = task_partitioner[task_id]
        _branchwrite!(rule, main_terms, main_coefficients, n_old + offsets[task_id], main_terms, main_coefficients, chunk.start, chunk.stop, mask, Val(true))
    end
    _eachtask(write_new_terms!, n_tasks)

    return n_new
end

# Walks terms[lo:hi] under `rule`. A term keeps its coefficient in place, and the term it creates is
# written from `write_start` on into the output arrays, which may be the very arrays being walked.
# A dry run (`DoWrite` false) only counts. Returns the number of new terms.
@inline function _branchwrite!(rule::F, output_terms, output_coefficients, write_start,
    terms, coefficients, lo, hi, mask, ::Val{DoWrite}) where {F,DoWrite}

    write_pos = write_start

    GC.@preserve terms @inbounds for ii in lo:hi
        branched = ruleat(rule, terms, coefficients, ii)
        branched === nothing && continue

        if branched isa Tuple
            kept_coefficient, new_coefficient = branched
            DoWrite && (coefficients[ii] = kept_coefficient)
            write_pos = _writeandadvance!(output_terms, output_coefficients, write_pos, terms[ii] ⊻ mask, new_coefficient, Val(DoWrite))
        elseif DoWrite
            coefficients[ii] = branched
        end
    end

    return write_pos - write_start
end

# Every pass is an array kernel, so the arrays may live anywhere: flag the terms that create a new
# term, turn the flags into write positions, then walk once more to rescale and write.
function _branchflagged!(rule::F, prop_cache, mask; thread::Bool=true) where {F}
    n_old = activesize(prop_cache)

    createsnewterm(term, coefficient) = @inline(rule(term, coefficient)) isa Tuple
    flag!(createsnewterm, prop_cache; thread)
    flagstoindices!(prop_cache; thread)

    n_new = lastactiveindex(prop_cache)
    _growto!(prop_cache, n_old + n_new)

    main_terms, main_coefficients, _, _ = _mainauxarrays(prop_cache)
    write_positions = activeindices(prop_cache)

    AK.foreachindex(write_positions; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do ii
        @inbounds begin
            branched = @inline rule(main_terms[ii], main_coefficients[ii])
            if branched isa Tuple
                kept_coefficient, new_coefficient = branched
                main_coefficients[ii] = kept_coefficient
                main_terms[n_old+write_positions[ii]] = main_terms[ii] ⊻ mask
                main_coefficients[n_old+write_positions[ii]] = new_coefficient
            elseif branched !== nothing
                main_coefficients[ii] = branched
            end
        end
    end

    setactivesize!(prop_cache, n_old + n_new)
    return prop_cache
end


# Because the zone assignment is linear in the term, `⊻ mask` permutes the zones: every zone writes
# the terms it creates into a single box, and takes delivery from a single zone.
function _xorbranch!(::MultiSumStorage, rule::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true, truncfunc=nothing) where {F}
    zone_storage = zonestorage(prop_cache)
    sorted_zones = _sortedzones(zone_storage, prop_cache)

    branch_zone!(zone_id) = _branchzone!(zone_storage, rule, zonecaches(prop_cache)[zone_id], _branchbox(prop_cache, zone_id), mask)
    _eachzone(branch_zone!, prop_cache, thread)

    # The box a zone collects is its tail already, in the parent order of the zone that made it, so
    # an array zone sorts it in from where it is instead of taking delivery first.
    function collect_branch!(owner)
        source = _xortarget(zonemap(prop_cache), owner, mask)
        _mergebox!(zone_storage, zonecaches(prop_cache)[owner], _branchbox(prop_cache, source), mask, sorted_zones, source; truncfunc)
    end
    _eachzone(collect_branch!, prop_cache, thread)

    return _syncsums!(prop_cache)
end

# A dictionary zone keeps its terms where they are and sets the new ones in its box.
_branchzone!(::StorageType, rule::F, zonecache, box, mask) where {F} = _branchdict!(rule, mainsum(zonecache), box, mask)

# An array zone writes the new terms straight into its box, in the order of their parents.
function _branchzone!(::ArrayStorage, rule::F, zonecache, box, mask) where {F}
    n_old = activesize(zonecache)

    # A term makes at most one new term, so the zone's size bounds what the box has to hold.
    length(box) < n_old && resize!(box, n_old)
    n_new = _branchwrite!(rule, terms(box), coefficients(box), 1,
        terms(mainsum(zonecache)), coefficients(mainsum(zonecache)), 1, n_old, mask, Val(true))
    resize!(box, n_new)

    return zonecache
end

# A zone that is sorted throughout hands its terms to a single other zone in ascending order, so the
# tail that zone takes delivery of is `mask ⊻ ascending` and sorts by XOR passes instead of by
# comparison. Merging here leaves `merge!` nothing to do afterwards.
_sortedzones(::StorageType, prop_cache::AbstractPropagationCache) = nothing

_sortedzones(::ArrayStorage, prop_cache::AbstractPropagationCache) =
    [sortedprefix(mainsum(zonecache)) == activesize(zonecache) for zonecache in zonecaches(prop_cache)]

# A dictionary zone merges as it takes delivery, where an array zone sorts the box in and merges it.
function _mergebox!(::StorageType, zonecache, box, mask, sorted_zones, source::Int; truncfunc=nothing)
    _deliver!(zonecache, box)
    truncfunc === nothing || truncate!(truncfunc, zonecache; thread=false)
    return zonecache
end

_mergebox!(::ArrayStorage, zonecache, box, mask, sorted_zones, source::Int; truncfunc=nothing) =
    xorsortedboxmerge!(zonecache, box, mask, (@inbounds sorted_zones[source]); thread=false, truncfunc)

# `⊻ mask` maps zone `source` onto this zone, and this zone back onto `source`.
@inline _xortarget(zone_map::ZoneMap, source::Int, mask) =
    ((source - 1) ⊻ _zonebits(mask, zone_map.masks)) + 1

# A fixed-mask branch has one destination zone, so one box holds all its new terms.
@inline _branchbox(prop_cache::AbstractPropagationCache, zone_id::Int) =
    @inbounds first(zones(outboxes(prop_cache)[zone_id]))
