###
##
# Dict-backed term sum helpers that work on a `Base.Dict` through its slots.
# A walk rewrites or deletes each entry at its slot, and an addition hashes its term once,
# where the public interface would hash the term again for each lookup.
# These are internals of Julia, used only where `_hasdictinternals` confirms them.
##
###

# whether `dict` is a `Base.Dict` whose internals match what the walks below rely on
_hasdictinternals(dict) = false
_hasdictinternals(::Dict) = _DICT_INTERNALS


### Walks over the slots

# Rescales the terms of `dict` in place and sets the terms they create in `new_sum`, as `_branchdict!`.
function _branchdict_internals!(rule::F, dict, new_sum, mask) where {F}
    slots, dict_keys, dict_vals = _dicttables(dict)
    age = dict.age
    n_touched = 0

    for i in _FilledSlots(slots)
        branched = @inline rule((@inbounds dict_keys[i]), (@inbounds dict_vals[i]))
        _checkunchanged(dict, age)

        if branched isa Kept
            @inbounds dict_vals[i] = branched.coefficient
            n_touched += 1
        elseif branched isa Branch
            @inbounds dict_vals[i] = branched.kept
            set!(new_sum, (@inbounds dict_keys[i]) ⊻ mask, branched.created)
            _checkunchanged(dict, age)
            n_touched += 1
        elseif !(branched isa Unchanged)
            _throwunknownoutcome(branched)
        end
    end

    return n_touched
end

# Maps every coefficient of `dict` and deletes the entries `truncfunc` drops, as `_mapandtruncate!`.
function _mapandtruncate_internals!(mapfunc::F, truncfunc::G, dict) where {F,G}
    slots, dict_keys, dict_vals = _dicttables(dict)
    age = dict.age

    for i in _FilledSlots(slots)
        outcome = _mapandtruncateoutcome(mapfunc, truncfunc, (@inbounds dict_keys[i]), (@inbounds dict_vals[i]))
        _checkunchanged(dict, age)

        if outcome isa Truncated
            Base._delete!(dict, i)
            age = dict.age
        else
            @inbounds dict_vals[i] = outcome.coefficient
        end
    end

    return dict
end

# Deletes the entries of `dict` that `keep` rejects, as `filter!`.
function _filter_internals!(keep::F, dict) where {F}
    slots, dict_keys, dict_vals = _dicttables(dict)
    age = dict.age

    for i in _FilledSlots(slots)
        is_kept = @inline keep((@inbounds dict_keys[i]), (@inbounds dict_vals[i]))
        _checkunchanged(dict, age)

        if !is_kept
            Base._delete!(dict, i)
            age = dict.age
        end
    end

    return dict
end

# Replaces every coefficient of `dict` by `transform(term, coefficient)`, as `mapcoeffsbypair!`.
function _mapcoeffsbypair_internals!(transform::F, dict) where {F}
    slots, dict_keys, dict_vals = _dicttables(dict)
    age = dict.age

    for i in _FilledSlots(slots)
        new_coefficient = @inline transform((@inbounds dict_keys[i]), (@inbounds dict_vals[i]))
        _checkunchanged(dict, age)
        @inbounds dict_vals[i] = new_coefficient
    end

    return dict
end

# Gives every coefficient of `dict` the one `new_coeff_func` returns for its slot on the cumulative weight, as `_mapslots!`.
function _mapslots_internals!(weight_func::W, new_coeff_func::F, dict) where {W,F}
    slots, _, dict_vals = _dicttables(dict)
    age = dict.age
    slot_end = zero(real(numcoefftype(valtype(dict))))

    for i in _FilledSlots(slots)
        coeff = @inbounds dict_vals[i]
        slot_start = slot_end
        slot_end += weight_func(coeff)
        new_coeff = new_coeff_func(coeff, slot_start, slot_end)
        _checkunchanged(dict, age)
        @inbounds dict_vals[i] = new_coeff
    end

    return dict
end

# Adds every entry of `source` into `dict`, combining equal terms with `mergefunc`, as `mergewith!(mergefunc, dict, source)`.
function _mergewith_internals!(dict::Dict, source::Dict)
    # grow the table to hold both, as `mergewith!` does from Julia 1.11 on, but never shrink it
    n_entries = length(dict) + length(source)
    if 3 * n_entries > 2 * length(dict.slots)
        sizehint!(dict, n_entries)
    end

    slots, source_keys, source_vals = _dicttables(source)
    age = source.age

    for i in _FilledSlots(slots)
        _add_internals!(dict, (@inbounds source_keys[i]), (@inbounds source_vals[i]))
        _checkunchanged(source, age)
    end

    return dict
end

# The entry in the first filled slot from `i` on and the slot to go on from, as `iterate(dict, i)`.
# The tables are read at every step, so the entry is found within them even if the dictionary changed in between.
@inline function _iterate_internals(dict::Dict, i::Int=1)
    if isempty(dict)
        return nothing
    end
    slots, dict_keys, dict_vals = _dicttables(dict)
    next = iterate(_FilledSlots(slots), max(i, 1))
    if next === nothing
        return nothing
    end
    slot, next_slot = next
    return Pair((@inbounds dict_keys[slot]), (@inbounds dict_vals[slot])), next_slot
end


### Shrinking

# A dictionary keeps its table when entries are deleted, and every walk visits the whole table,
# so a filter that leaves fewer entries than a sixteenth of the slots rebuilds the table with room for twice the entries.
function _shrinkifsparse!(dict)
    if _hasdictinternals(dict) && 16 * length(dict) < length(dict.slots)
        sizehint!(dict, 2 * length(dict))
    end
    return dict
end


### Adding with one hash

# Adds `coeff` to the coefficient of `term` in `dict`, hashing `term` once.
function _add_internals!(dict::Dict{K,V}, term::K, coeff) where {K,V}
    index, shorthash = Base.ht_keyindex2_shorthash!(dict, term)
    if index > 0
        dict.vals[index] = mergefunc(dict.vals[index], coeff)
    else
        Base._setindex!(dict, convert(V, coeff), term, -index, shorthash)
    end
    return dict
end


### The model of `Base.Dict`

# The slot, key and value tables of `dict`, which the walks index by the slots.
function _dicttables(dict::Dict)
    slots, dict_keys, dict_vals = dict.slots, dict.keys, dict.vals
    if !(length(slots) == length(dict_keys) == length(dict_vals))
        throw(ArgumentError("the slot, key and value tables of the dictionary differ in length"))
    end
    return slots, dict_keys, dict_vals
end

# The indices of the slots that hold an entry, in order.
# The slots are read eight at a time, which needs the table to be a multiple of eight slots long, as every power of two from 16 on is.
struct _FilledSlots{S}
    slots::S

    function _FilledSlots(slots::S) where {S}
        if length(slots) % 8 != 0
            throw(ArgumentError("the dictionary has $(length(slots)) slots, which is not a multiple of eight"))
        end
        return new{S}(slots)
    end
end

Base.IteratorSize(::Type{<:_FilledSlots}) = Base.SizeUnknown()

@inline function Base.iterate(filled::_FilledSlots, i::Int=1)
    next_filled = _nextfilled(filled.slots, i)
    if next_filled == 0
        return nothing
    end
    return next_filled, next_filled + 1
end

# The first slot from `i` on that holds an entry, or 0 if none does, for slots a multiple of eight long.
# A slot holds an entry when its top bit is set, so the eight slots of a word hold one where the word has a top bit of a byte set.
@inline function _nextfilled(slots, i::Int)
    while i <= length(slots)
        offset = (i - 1) & 7
        word_start = i - 1 - offset
        filled = (_slotword(slots, word_start) & 0x8080808080808080) >> (UInt(8) * (offset % UInt))
        if filled != 0
            return i + (trailing_zeros(filled) >> 3)
        end
        i = word_start + 9
    end
    return 0
end

# the eight slots after `word_start` as one word, the first in its lowest byte, for `word_start + 8` within the slots
@inline _slotword(slots, word_start::Int) = GC.@preserve slots ltoh(unsafe_load(Ptr{UInt64}(pointer(slots)) + word_start))

# Every insert, delete, rehash and empty of a `Dict` advances its age, so a walk that finds the age
# unchanged after calling back into user code still indexes the tables it read.
@inline function _checkunchanged(dict::Dict, age)
    if dict.age != age
        _throwchangedwhilewalked()
    end
    return
end

@noinline _throwchangedwhilewalked() = throw(ArgumentError(
    "the dictionary changed while its terms were walked; a callback must not change the sum it is applied to"))

# Edits a dictionary through the walks' own pieces and through the public interface alike, and compares the two.
function _dictinternalsagree()
    walked = Dict{UInt64,Float64}()
    reference = Dict{UInt64,Float64}()

    # enough terms to grow the tables several times, each added twice
    for round in 1:2, k in UInt64(1):UInt64(2000)
        term = k * 0x9e3779b97f4a7c15
        _add_internals!(walked, term, Float64(k))
        reference[term] = get(reference, term, 0.0) + Float64(k)
    end
    if walked != reference
        return false
    end

    # the next filled slot from every slot on is the one Base finds
    slots, dict_keys, dict_vals = _dicttables(walked)
    for i in eachindex(slots)
        next_filled = something(findnext(j -> Base.isslotfilled(walked, j), eachindex(slots), i), 0)
        if _nextfilled(slots, i) != next_filled
            return false
        end
    end

    # a walk visits every entry once, deletes some entries behind itself and rescales others
    n_entries = length(walked)
    n_visited = 0
    for i in _FilledSlots(slots)
        n_visited += 1
        term = dict_keys[i]
        if term % 3 == 0
            Base._delete!(walked, i)
            delete!(reference, term)
        else
            dict_vals[i] = -dict_vals[i]
            reference[term] = -reference[term]
        end
    end

    source = Dict{UInt64,Float64}(UInt64(k) * 0x9e3779b97f4a7c15 => 1.0 for k in 1500:2500)
    _mergewith_internals!(walked, source)
    mergewith!(+, reference, source)

    n_filled = count(i -> Base.isslotfilled(walked, i), eachindex(walked.slots))
    if !(n_visited == n_entries && walked == reference && n_filled == length(walked) && length(walked.slots) % 8 == 0)
        return false
    end

    # an insert, a delete, a rehash and an empty each advance the age
    ages = [walked.age]
    _add_internals!(walked, zero(UInt64), 1.0)
    push!(ages, walked.age)
    Base._delete!(walked, findfirst(i -> Base.isslotfilled(walked, i), eachindex(walked.slots)))
    push!(ages, walked.age)
    sizehint!(walked, 4 * length(walked.slots))
    push!(ages, walked.age)
    empty!(walked)
    push!(ages, walked.age)
    return allunique(ages)
end

# Whether `Base.Dict` has the fields and helpers used here, and whether they behave as used.
# Checked once, when the package is precompiled for a Julia version.
function _checkdictinternals()
    for field in (:slots, :keys, :vals, :age)
        if !hasfield(Dict{UInt64,Float64}, field)
            return false
        end
    end
    for helper in (:ht_keyindex2_shorthash!, :_setindex!, :_delete!, :isslotfilled)
        if !isdefined(Base, helper)
            return false
        end
    end

    try
        return _dictinternalsagree()
    catch
        return false
    end
end

const _DICT_INTERNALS = _checkdictinternals()
