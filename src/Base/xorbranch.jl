"""
    xorbranch(rule, prop_cache::AbstractPropagationCache, mask; thread=true)

Branch every active term by `rule` on a copy, see `xorbranch!`.
"""
xorbranch(rule, prop_cache::AbstractPropagationCache, mask; kwargs...) =
    xorbranch!(rule, deepcopy(prop_cache), mask; kwargs...)

"""
    xorbranch!(rule, prop_cache::AbstractPropagationCache, mask; thread=true)

Branch every active term by `rule`.
`rule(term, coefficient)` returns `Unchanged()` to leave the term alone, `Kept(coefficient)` to keep it with a new coefficient,
or `Branch(kept, created)` to keep it with `kept` and create the term `term ⊻ mask` with `created`.
The terms it creates are left for `xormerge!` or `xormergeandtruncate!` to merge:
a dictionary holds them in the auxiliary sum, an array appends them past its sorted prefix in the order of their parents,
a multi sum leaves them in the outboxes, each zone's in the one box of the zone that owns them, and any other storage adds them through `flatmap!`.
Every new term is the same `⊻ mask` away from its parent, which is what lets those merges sort the new terms in without comparing them.
`merge!` and `mergeandtruncate!` merge them too, only by comparison.
Several tasks on an array call `rule` once to count the terms and once to write them, so it must return the same for the same pair each time.
"""
xorbranch!(rule::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true) where {F} =
    _xorbranch!(StorageType(prop_cache), rule, prop_cache, mask; thread)

"""
    Unchanged()

The outcome of a rule for `xorbranch!` that leaves a term as it is.
A rule that returns it need not read the coefficient.
"""
struct Unchanged end

"""
    Kept(coefficient)

The outcome of a rule for `xorbranch!` that keeps the term with `coefficient`.
"""
struct Kept{C}
    coefficient::C
end

"""
    Branch(kept, created)

The outcome of a rule for `xorbranch!` that keeps the term with the coefficient `kept`
and creates the term `term ⊻ mask` with the coefficient `created`.
"""
struct Branch{C}
    kept::C
    created::C
end

"""
    ruleat(rule, terms, coefficients, ii)

`rule(terms[ii], coefficients[ii])`, as the array kernels of `xorbranch!` call a rule.
A rule that can decide from a part of the term overloads this to read only that part,
and the coefficient only once it needs it.
The kernels keep `terms` alive for the whole walk, so a rule may read it through a pointer.
"""
@inline ruleat(rule::F, terms, coefficients, ii::Int) where {F} = @inline rule((@inbounds terms[ii]), (@inbounds coefficients[ii]))

function _xorbranch!(::StorageType, rule::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true) where {F}
    function branch(term, coefficient)
        branched = rule(term, coefficient)
        if branched isa Unchanged
            return ((term, coefficient),)
        elseif branched isa Kept
            return ((term, branched.coefficient),)
        elseif branched isa Branch
            return ((term, branched.kept), (term ⊻ mask, branched.created))
        else
            _throwunknownoutcome(branched)
        end
    end

    return flatmap!(branch, prop_cache; thread)
end


### Dictionary storage

function _xorbranch!(::DictStorage, rule::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true) where {F}
    new_sum = auxsum(prop_cache)
    if !isempty(new_sum)
        empty!(new_sum)
    end

    _branchdict!(rule, mainsum(prop_cache), new_sum, mask)
    return prop_cache
end

# Rescales the terms of `term_sum` in place and sets the terms they create in `new_sum`, which is
# empty. Two terms never create the same new term, so each is set rather than added.
function _branchdict!(rule::F, term_sum, new_sum, mask) where {F}
    for (term, coefficient) in term_sum
        branched = @inline rule(term, coefficient)

        if branched isa Kept
            set!(term_sum, term, branched.coefficient)
        elseif branched isa Branch
            set!(term_sum, term, branched.kept)
            set!(new_sum, term ⊻ mask, branched.created)
        elseif !(branched isa Unchanged)
            _throwunknownoutcome(branched)
        end
    end

    return term_sum
end


### Array storage

# The new terms are appended past the active terms in the order of their parents.
function _xorbranch!(::ArrayStorage, rule::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true) where {F}
    if isempty(prop_cache)
        return prop_cache
    end

    if _iscpuarray(terms(mainsum(prop_cache)))
        return _branchcpu!(rule, prop_cache, mask; thread)
    end
    return _branchflagged!(rule, prop_cache, mask; thread)
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
        write_pos += _branchwrite!(rule, main_terms, main_coefficients, write_pos, capacity(prop_cache),
            main_terms, main_coefficients, lo, hi, mask, Val(true))
        lo = hi + 1
    end

    return write_pos - n_old - 1
end

function _branchintasks!(rule::F, prop_cache, n_old::Int, task_partitioner, n_tasks::Int, mask) where {F}
    counts = Vector{Int}(undef, n_tasks)

    function count_new_terms!(task_id)
        main_terms, main_coefficients, _, _ = _mainauxarrays(prop_cache)
        chunk = task_partitioner[task_id]
        counts[task_id] = _branchwrite!(rule, main_terms, main_coefficients, 1, 0, main_terms, main_coefficients, chunk.start, chunk.stop, mask, Val(false))
    end
    _eachtask(count_new_terms!, n_tasks)

    offsets = _offsetsfromcounts(counts)
    n_new = offsets[end] - 1
    _growto!(prop_cache, n_old + n_new)

    # each task writes within the room its count reserved and reports what it made
    written = Vector{Int}(undef, n_tasks)
    function write_new_terms!(task_id)
        main_terms, main_coefficients, _, _ = _mainauxarrays(prop_cache)
        chunk = task_partitioner[task_id]
        written[task_id] = _branchwrite!(rule, main_terms, main_coefficients, n_old + offsets[task_id], n_old + offsets[task_id+1] - 1,
            main_terms, main_coefficients, chunk.start, chunk.stop, mask, Val(true))
    end
    _eachtask(write_new_terms!, n_tasks)

    if written != counts
        _throwreplaymismatch()
    end
    return n_new
end

# Walks terms[lo:hi] under `rule`. A term keeps its coefficient in place, and the term it creates is
# written from `write_start` up to `write_stop` into the output arrays, which may be the very arrays
# being walked. A dry run (`DoWrite` false) only counts. Returns the number of new terms, which
# includes those past `write_stop` that were not written.
@inline function _branchwrite!(rule::F, output_terms, output_coefficients, write_start, write_stop,
    terms, coefficients, lo, hi, mask, ::Val{DoWrite}) where {F,DoWrite}

    write_pos = write_start

    GC.@preserve terms @inbounds for ii in lo:hi
        branched = ruleat(rule, terms, coefficients, ii)

        if branched isa Kept
            if DoWrite
                coefficients[ii] = branched.coefficient
            end
        elseif branched isa Branch
            if DoWrite
                coefficients[ii] = branched.kept
            end
            write_pos = _writeandadvance!(output_terms, output_coefficients, write_pos, write_stop, terms[ii] ⊻ mask, branched.created, Val(DoWrite))
        elseif !(branched isa Unchanged)
            _throwunknownoutcome(branched)
        end
    end

    return write_pos - write_start
end

# Every pass is an array kernel, so the arrays may live anywhere: flag the terms that create a new
# term, turn the flags into write positions, then walk once more to rescale and write.
function _branchflagged!(rule::F, prop_cache, mask; thread::Bool=true) where {F}
    n_old = activesize(prop_cache)

    createsnewterm(term, coefficient) = @inline(rule(term, coefficient)) isa Branch
    flag!(createsnewterm, prop_cache; thread)
    flagstoindices!(prop_cache; thread)

    n_new = lastactiveindex(prop_cache)
    _growto!(prop_cache, n_old + n_new)

    main_terms, main_coefficients, _, _ = _mainauxarrays(prop_cache)
    write_positions = activeindices(prop_cache)
    counted = activeflags(prop_cache)

    # a term only writes where the first pass counted a new term, and its flag then says whether
    # the rule answered the same again
    AK.foreachindex(write_positions; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do ii
        @inbounds begin
            branched = @inline rule(main_terms[ii], main_coefficients[ii])
            if branched isa Kept
                main_coefficients[ii] = branched.coefficient
            elseif branched isa Branch
                main_coefficients[ii] = branched.kept
                if counted[ii]
                    main_terms[n_old+write_positions[ii]] = main_terms[ii] ⊻ mask
                    main_coefficients[n_old+write_positions[ii]] = branched.created
                end
            elseif !(branched isa Unchanged)
                _throwunknownoutcome(branched)
            end
            counted[ii] = counted[ii] != (branched isa Branch)
        end
    end

    if AK.any(identity, counted; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
        _throwreplaymismatch()
    end

    setactivesize!(prop_cache, n_old + n_new)
    return prop_cache
end


# Because the zone assignment is linear in the term, `⊻ mask` permutes the zones: every zone writes
# the terms it creates into the box of the single zone that owns them.
function _xorbranch!(::MultiSumStorage, rule::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true) where {F}
    zone_storage = zonestorage(prop_cache)

    branch_zone!(zone_id) = _branchzone!(zone_storage, rule, zonecaches(prop_cache)[zone_id], _branchbox(prop_cache, zone_id, mask), mask)
    _eachzone(branch_zone!, prop_cache, thread)

    return prop_cache
end

# A dictionary zone keeps its terms where they are and sets the new ones in its box.
_branchzone!(::StorageType, rule::F, zonecache, box, mask) where {F} = _branchdict!(rule, mainsum(zonecache), box, mask)

# An array zone writes the new terms straight into its box, in the order of their parents.
function _branchzone!(::ArrayStorage, rule::F, zonecache, box, mask) where {F}
    n_old = activesize(zonecache)

    # A term makes at most one new term, so the zone's size bounds what the box has to hold.
    if length(box) < n_old
        resize!(box, n_old)
    end
    n_new = _branchwrite!(rule, terms(box), coefficients(box), 1, n_old,
        terms(mainsum(zonecache)), coefficients(mainsum(zonecache)), 1, n_old, mask, Val(true))
    resize!(box, n_new)

    return zonecache
end

# `⊻ mask` maps zone `source` onto this zone, and this zone back onto `source`.
@inline _xortarget(zone_map::ZoneMap, source::Int, mask) =
    ((source - 1) ⊻ _zonebits(mask, zone_map.masks)) + 1

# the box in the outbox of `zone_id` for the zone its new terms belong to
@inline _branchbox(prop_cache::AbstractPropagationCache, zone_id::Int, mask) =
    zones(outboxes(prop_cache)[zone_id])[_xortarget(zonemap(prop_cache), zone_id, mask)]

@noinline _throwunknownoutcome(branched) =
    throw(ArgumentError("rule returned $(typeof(branched)); expected Unchanged(), Kept(coefficient), or Branch(kept, created)"))
