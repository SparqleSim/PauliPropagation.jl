###
##
# A term sum held in a hash table, for the sums of a few to a few thousand terms that fit the cache, such as the terms
# of one coset of a layer pass: an entry per term, found by a probe of the slots from the key of the term. A gate that
# branches by an XOR mask finds the partner of every term in place and merges as it branches, so that nothing is sorted
# in afterwards. An entry the truncation drops stays where it is, absent with a zero coefficient, and takes its term
# again when a gate creates it, so that no term ever has two entries.
#
# The key of a term is a hash of it, unless the sum has been told which bits tell its terms apart and they fit in 64
# bits: then the key is those bits of the term gathered into an integer, it alone identifies the term, and the key of
# the partner of a term under an XOR mask is the key of the term XOR the key of the mask.
##
###

struct HashedStorage <: StorageType end

"""
    HashedTermSum{TT,CT}(nsites)

A sum of terms of type `TT` with coefficients of type `CT` on `nsites` sites, held in a hash table.
Made for the sums of a few to a few thousand terms that fit the cache: `xorbranchmergeandtruncate!` on it finds the partner of every term in place and merges as it branches, and every term goes through the truncation after every branch, as on a dictionary.
The terms come out in no particular order.
"""
mutable struct HashedTermSum{TT,CT} <: AbstractTermSum
    nsites::Int
    # the entries: a term, its key, its coefficient, whether the term is in the sum, and the stamp of the branch that
    # last wrote the entry; an entry the truncation dropped keeps its term and its slot and holds a zero coefficient,
    # and the entry past the last is scratch for the probes
    terms::Vector{TT}
    keys::Vector{UInt64}
    coeffs::Vector{CT}
    present::Vector{Bool}
    stamps::Vector{Int32}
    n_entries::Int
    # the entry at the slot that the key of its term picks, or 0; the first `n_slots` slots are in use, a power of two
    # with at least `_SLOTS_PER_ENTRY` of them per entry, and the highest `64 - shift` bits of a hash pick the slot
    slots::Vector{Int32}
    n_slots::Int
    shift::Int
    stamp::Int32
    # the bits in which the terms of the sum differ, and whether they fit in 64 bits, so that the keys are those bits
    # gathered rather than hashes of the terms
    distinguishing_bits::TT
    gathered::Bool
    # the entries that a branch in passes does more to than keep their coefficients, each with its partner or 0 in
    # the high half
    events::Vector{Int64}
end

function HashedTermSum{TT,CT}(nsites::Int) where {TT,CT}
    hsum = HashedTermSum{TT,CT}(nsites, TT[], UInt64[], CT[], Bool[], Int32[], 0, Int32[], 0, 64, zero(Int32),
        ~zero(TT), false, Int64[])
    _distinguishby!(hsum, ~zero(TT))
    empty!(hsum)
    return hsum
end

# the slots per entry the table keeps at least, so that a probe ends after a few slots
const _SLOTS_PER_ENTRY = 8

# the entries from which a branch runs in passes over the sum rather than entry by entry
const _PASSES_FROM_ENTRIES = 32

storage(hsum::HashedTermSum) = hsum
_storagetype(::HashedTermSum) = HashedStorage()
nsites(hsum::HashedTermSum) = hsum.nsites

_length(::HashedStorage, hsum::HashedTermSum) = count(view(hsum.present, 1:hsum.n_entries))
_terms(::HashedStorage, hsum::HashedTermSum) = [hsum.terms[entry] for entry in 1:hsum.n_entries if hsum.present[entry]]
_coefficients(::HashedStorage, hsum::HashedTermSum) = [hsum.coeffs[entry] for entry in 1:hsum.n_entries if hsum.present[entry]]
_termtype(::HashedStorage, ::HashedTermSum{TT,CT}) where {TT,CT} = TT
_coefftype(::HashedStorage, ::HashedTermSum{TT,CT}) where {TT,CT} = CT
function _similar(::HashedStorage, hsum::HashedTermSum{TT,CT}) where {TT,CT}
    similar_sum = HashedTermSum{TT,CT}(hsum.nsites)
    _distinguishby!(similar_sum, hsum.distinguishing_bits)
    return similar_sum
end
_emptylike(::HashedStorage, hsum::HashedTermSum) = similar(hsum)
_lookupcost(::HashedStorage, hsum::HashedTermSum) = 1

# the present entries, as (term, coefficient)
function _iterate(::HashedStorage, hsum::HashedTermSum, entry::Int=1)
    while entry <= hsum.n_entries
        if hsum.present[entry]
            return (hsum.terms[entry], hsum.coeffs[entry]), entry + 1
        end
        entry += 1
    end
    return nothing
end

"""
    empty!(hsum::HashedTermSum; sizehint=0)

Empties the sum, with slots for `sizehint` terms before they have to grow.
"""
function Base.empty!(hsum::HashedTermSum; sizehint::Int=0)
    hsum.n_entries = 0
    _setslots!(hsum, max(64, nextpow(2, _SLOTS_PER_ENTRY * max(sizehint, 1))))
    _reserve!(hsum, 0)
    return hsum
end

"""
    _distinguishby!(termsum, bits)

Tells the sum that any two terms it holds, or that a gate applied to it makes, differ in some of the `bits`, so that it may key its entries by those bits of their terms alone.
A sum that keys its entries otherwise ignores this.
"""
_distinguishby!(termsum, bits) = termsum

function _distinguishby!(hsum::HashedTermSum{TT}, bits::TT) where {TT}
    hsum.distinguishing_bits = bits
    hsum.gathered = count_ones(bits) <= 64
    if hsum.n_entries > 0
        for entry in 1:hsum.n_entries
            hsum.keys[entry] = _key(hsum, hsum.terms[entry])
        end
        _rebuildslots!(hsum, hsum.n_slots)
    end
    return hsum
end

# the key of a term: its distinguishing bits gathered into an integer where they fit, or a hash of the term
@inline function _key(hsum::HashedTermSum, term)
    if hsum.gathered
        return _gatherbits(term, hsum.distinguishing_bits)
    end
    return _slothash(term)
end

# The bits of `term` at the set bits of `bits`, gathered into an integer in their order; `bits` has at most 64 set bits.
@inline function _gatherbits(term::TT, bits::TT) where {TT}
    set_limbs = _limbs(term & bits)
    bit_limbs = _limbs(bits)
    gathered = zero(UInt64)
    n_below = 0
    for limb_index in eachindex(bit_limbs)
        set_bits = set_limbs[limb_index]
        while set_bits != 0
            lowest = set_bits & (~set_bits + one(UInt64))
            set_bits &= set_bits - one(UInt64)
            gathered |= one(UInt64) << ((n_below + count_ones(bit_limbs[limb_index] & (lowest - one(UInt64)))) & 63)
        end
        n_below += count_ones(bit_limbs[limb_index])
    end
    return gathered
end

# a hash of a term whose highest bits pick its slot: the limbs folded onto 64 bits as in `_hashbits`, without its final mixing
@inline function _slothash(term)
    limbs = _limbs(term)
    folded = zero(UInt64)
    for i in eachindex(limbs)
        folded ⊻= (limbs[i] ⊻ (limbs[i] >> 32)) * _foldfactor(i)
    end
    return folded
end

# The slot that a key picks first, from its highest bits: a gathered key is multiplied by an odd constant first, which
# moves its bits up to them, while a hash holds them already.
@inline _keyedslot(key::UInt64, shift::Int) = ((key * 0x9e3779b97f4a7c15) >> (shift & 63)) % Int
@inline _hashedslot(key::UInt64, shift::Int) = (key >> (shift & 63)) % Int

# `n_slots` empty slots
function _setslots!(hsum::HashedTermSum, n_slots::Int)
    _ensurecapacity!(hsum.slots, n_slots)
    fill!(view(hsum.slots, 1:n_slots), zero(Int32))
    hsum.n_slots = n_slots
    hsum.shift = 64 - trailing_zeros(n_slots)
    return hsum
end

# `n_slots` slots holding the entries
function _rebuildslots!(hsum::HashedTermSum, n_slots::Int)
    _setslots!(hsum, n_slots)
    slots = hsum.slots
    slot_mask = hsum.n_slots - 1
    for entry in 1:hsum.n_entries
        slot = if hsum.gathered
            _keyedslot(hsum.keys[entry], hsum.shift)
        else
            _hashedslot(hsum.keys[entry], hsum.shift)
        end
        while slots[slot+1] != 0
            slot = (slot + 1) & slot_mask
        end
        slots[slot+1] = entry % Int32
    end
    return hsum
end

# Room for `n` entries and the scratch entry past them, and slots for them.
@inline function _reserve!(hsum::HashedTermSum, n::Int)
    if length(hsum.terms) <= n || hsum.n_slots < _SLOTS_PER_ENTRY * n
        _grow!(hsum, n)
    end
    return hsum
end

# the arrays grown for `n` entries, and the slots, which are rebuilt where they grow
@noinline function _grow!(hsum::HashedTermSum, n::Int)
    if length(hsum.terms) <= n
        for array in (hsum.terms, hsum.keys, hsum.coeffs, hsum.present, hsum.stamps, hsum.events)
            _ensurecapacity!(array, n + 1)
        end
    end
    if hsum.n_slots < _SLOTS_PER_ENTRY * n
        _rebuildslots!(hsum, nextpow(2, _SLOTS_PER_ENTRY * n))
    end
    return hsum
end


### Probes

# The probe of a sum finds the entry of a term from its key: where the keys are gathered bits, the key of the partner
# of a term under the mask is the key XOR the key of the mask, and a key is compared on its own; where they are
# hashes, the partner is hashed and the terms are compared.
struct _KeyedProbe
    slots::Vector{Int32}
    keys::Vector{UInt64}
    n_slots::Int
    shift::Int
    mask_key::UInt64
end

struct _TermProbe{TT}
    slots::Vector{Int32}
    terms::Vector{TT}
    n_slots::Int
    shift::Int
end

# the probe of a sum with gathered keys, for the partners under `mask`, and of a sum with hashed keys; the callers
# pick the one the sum keys by, so that each loop is compiled for its probe
_KeyedProbe(hsum::HashedTermSum{TT}, mask::TT) where {TT} =
    _KeyedProbe(hsum.slots, hsum.keys, hsum.n_slots, hsum.shift, _gatherbits(mask, hsum.distinguishing_bits))
_TermProbe(hsum::HashedTermSum) = _TermProbe(hsum.slots, hsum.terms, hsum.n_slots, hsum.shift)

# the key of the partner `partner_term` of a term with the key `key`
@inline _partnerkey(probe::_KeyedProbe, key::UInt64, partner_term) = key ⊻ probe.mask_key
@inline _partnerkey(::_TermProbe, key::UInt64, partner_term) = _slothash(partner_term)

# The entry holding the term `term` with the key `key`, present or not, or 0, and the slot where it is or would go.
# What the probe compares is written to the scratch entry `sentinel` past the others, which an empty slot reads, so
# that the probe ends on one comparison, which holds for an empty slot and the entry alike, without a branch on which
# of them it is. Every caller has reserved room for the scratch entry and keeps the slots at most
# `1 / _SLOTS_PER_ENTRY` full, so that an empty slot is found.
@inline function _find(probe::_KeyedProbe, key::UInt64, term, sentinel::Int)
    slots = probe.slots
    keys = probe.keys
    @inbounds keys[sentinel] = key
    slot = _keyedslot(key, probe.shift)
    slot_mask = probe.n_slots - 1
    while true
        entry = (@inbounds slots[slot+1]) % Int
        if (@inbounds keys[ifelse(entry == 0, sentinel, entry)]) == key
            return entry, slot
        end
        slot = (slot + 1) & slot_mask
    end
end

@inline function _find(probe::_TermProbe, key::UInt64, term, sentinel::Int)
    slots = probe.slots
    terms = probe.terms
    @inbounds terms[sentinel] = term
    slot = _hashedslot(key, probe.shift)
    slot_mask = probe.n_slots - 1
    while true
        entry = (@inbounds slots[slot+1]) % Int
        if (@inbounds terms[ifelse(entry == 0, sentinel, entry)]) == term
            return entry, slot
        end
        slot = (slot + 1) & slot_mask
    end
end

@noinline _throwindistinct(held, term) =
    throw(ArgumentError("the terms $held and $term do not differ in the distinguishing bits of the sum"))

# the entry of `term` with the key `key`, present or not, or 0, and the slot where it is or would go; a gathered key
# that matches a different term breaks the promise made to `_distinguishby!`
@inline function _entryof(hsum::HashedTermSum, term, key::UInt64=_key(hsum, term))
    sentinel = hsum.n_entries + 1
    if hsum.gathered
        probe = _KeyedProbe(hsum.slots, hsum.keys, hsum.n_slots, hsum.shift, zero(UInt64))
        entry, slot = _find(probe, key, term, sentinel)
        if entry != 0 && hsum.terms[entry] != term
            _throwindistinct(hsum.terms[entry], term)
        end
        return entry, slot
    end
    return _find(_TermProbe(hsum), key, term, sentinel)
end

# the entry of `term`, present or not, made past the others where there is none, and whether it was present
@inline function _entryof!(hsum::HashedTermSum, term)
    _reserve!(hsum, hsum.n_entries + 1)
    key = _key(hsum, term)
    entry, slot = _entryof(hsum, term, key)
    if entry != 0
        return entry, (@inbounds hsum.present[entry])
    end
    entry = hsum.n_entries + 1
    hsum.n_entries = entry
    @inbounds begin
        hsum.terms[entry] = term
        hsum.keys[entry] = key
        hsum.present[entry] = false
        hsum.stamps[entry] = zero(Int32)
        hsum.slots[slot+1] = entry % Int32
    end
    return entry, false
end


### The term-sum interface

@inline function _add!(::HashedStorage, hsum::HashedTermSum, term, coeff)
    entry, was_present = _entryof!(hsum, term)
    new_coeff = if was_present
        mergefunc((@inbounds hsum.coeffs[entry]), coeff)
    else
        coeff
    end
    @inbounds hsum.coeffs[entry] = new_coeff
    @inbounds hsum.present[entry] = true
    return hsum
end

function _set!(::HashedStorage, hsum::HashedTermSum, term, coeff)
    entry, _ = _entryof!(hsum, term)
    @inbounds hsum.coeffs[entry] = coeff
    @inbounds hsum.present[entry] = true
    return hsum
end

function _delete!(::HashedStorage, hsum::HashedTermSum{TT,CT}, term) where {TT,CT}
    entry, _ = _entryof(hsum, term)
    if entry != 0
        @inbounds hsum.present[entry] = false
        @inbounds hsum.coeffs[entry] = zero(CT)
    end
    return hsum
end

function _getcoeff(::HashedStorage, hsum::HashedTermSum, term)
    entry, _ = _entryof(hsum, term)
    if entry != 0 && @inbounds(hsum.present[entry])
        return @inbounds hsum.coeffs[entry]
    end
    return zero(coefftype(hsum))
end

_getmergedcoeff(::HashedStorage, hsum::HashedTermSum, term) = _getcoeff(HashedStorage(), hsum, term)

function _copy!(::HashedStorage, dst::HashedTermSum, src::HashedTermSum)
    empty!(dst; sizehint=length(src))
    for (term, coeff) in src
        add!(dst, term, coeff)
    end
    return dst
end

function _mapcoeffs!(::HashedStorage, transform::F, hsum::HashedTermSum; thread::Bool=true) where {F}
    @inbounds for entry in 1:hsum.n_entries
        if hsum.present[entry]
            hsum.coeffs[entry] = transform(hsum.coeffs[entry])
        end
    end
    return hsum
end

function _mapreduce(::HashedStorage, mapper::F, reducer::O, thing; init, neutral, thread::Bool) where {F,O}
    accumulated = init
    for (term, coefficient) in thing
        accumulated = reducer(accumulated, @inline mapper(term, coefficient))
    end
    return accumulated
end

_mapreducecoeffs(::HashedStorage, mapper::F, reducer::O, thing; init, neutral, thread::Bool) where {F,O} =
    _mapreduce(HashedStorage(), (term, coefficient) -> mapper(coefficient), reducer, thing; init, neutral, thread)

# a dropped entry stays where it is, absent, with a zero coefficient
function _filter!(::HashedStorage, keep::F, hsum::HashedTermSum{TT,CT}; thread::Bool=true) where {F,TT,CT}
    @inbounds for entry in 1:hsum.n_entries
        if hsum.present[entry]
            kept = @inline keep(hsum.terms[entry], hsum.coeffs[entry])
            hsum.present[entry] = kept
            hsum.coeffs[entry] = ifelse(kept, hsum.coeffs[entry], zero(CT))
        end
    end
    return hsum
end

# every term is in the sum once, so the sum is merged as it is
_mergeandtruncate!(::HashedStorage, truncfunc::F, hsum::HashedTermSum; thread::Bool=true) where {F} =
    _truncate!(truncfunc, hsum; thread)


### Propagation caches

# A propagation cache over a hashed sum, of no basis in particular: the cache into which a layer pass loads its cosets
# and applies the gates to them with `xorbranchmergeandtruncate!`. It holds the sum as it is, and the auxiliary sum only
# takes the terms of the generic passes.
mutable struct HashedPropagationCache{TT,CT} <: AbstractPropagationCache
    main_sum::HashedTermSum{TT,CT}
    aux_sum::HashedTermSum{TT,CT}
end

PropagationCache(hsum::HashedTermSum) = HashedPropagationCache(hsum, similar(hsum))

mainsum(prop_cache::HashedPropagationCache) = prop_cache.main_sum
auxsum(prop_cache::HashedPropagationCache) = prop_cache.aux_sum

function setmainsum!(prop_cache::HashedPropagationCache, new_mainsum)
    prop_cache.main_sum = new_mainsum
    return prop_cache
end

function setauxsum!(prop_cache::HashedPropagationCache, new_auxsum)
    prop_cache.aux_sum = new_auxsum
    return prop_cache
end

_iterate(::HashedStorage, prop_cache::AbstractPropagationCache, args...) = iterate(mainsum(prop_cache), args...)
_capacity(::HashedStorage, prop_cache::AbstractPropagationCache) = length(mainsum(prop_cache).terms) - 1
_resize!(::HashedStorage, prop_cache::AbstractPropagationCache, n::Int) = (_reserve!(mainsum(prop_cache), n); prop_cache)

_filter!(::HashedStorage, keep::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F} =
    (_filter!(HashedStorage(), keep, mainsum(prop_cache); thread); prop_cache)

# a branch leaves every term in the main sum once, so only terms that the generic passes left in the auxiliary sum are
# added to it before the truncation
function _mergeandtruncate!(::HashedStorage, truncfunc::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    if !isempty(auxsum(prop_cache))
        add!(mainsum(prop_cache), auxsum(prop_cache))
        empty!(auxsum(prop_cache))
    end
    _truncate!(truncfunc, prop_cache; thread)
    return prop_cache
end

# A branch runs entry by entry, or in passes over the entries where at least a quarter of them hold coefficients below
# the floor that the rule derives from the cutoff of the truncation, which create nothing that the truncation keeps.
function _xorbranchmergeandtruncate!(::HashedStorage, rule::F, truncfunc::G, prop_cache::AbstractPropagationCache, mask; thread::Bool=true) where {F,G}
    hsum = mainsum(prop_cache)
    # an entry for the partner every entry may create
    _reserve!(hsum, 2 * hsum.n_entries)
    if hsum.gathered
        _branchhashed!(rule, truncfunc, hsum, mask, _KeyedProbe(hsum, mask))
    else
        _branchhashed!(rule, truncfunc, hsum, mask, _TermProbe(hsum))
    end
    return prop_cache
end

function _branchhashed!(rule::F, truncfunc::G, hsum::HashedTermSum, mask, probe::P) where {F,G,P}
    floor = _creationfloor(rule, _coefficientcutoff(truncfunc))
    if hsum.n_entries >= _PASSES_FROM_ENTRIES && _manybelow(hsum, floor)
        _branchinpasses!(rule, truncfunc, hsum, mask, probe, floor)
    else
        _branchentrybyentry!(rule, truncfunc, hsum, mask, probe)
    end
    return hsum
end

# whether at least a quarter of the present entries hold a coefficient below `floor`
function _manybelow(hsum::HashedTermSum, floor)
    n_present = 0
    n_below = 0
    @inbounds for entry in 1:hsum.n_entries
        is_present = hsum.present[entry]
        n_present += is_present
        n_below += is_present & truncatemincoeff(hsum.coeffs[entry], floor)
    end
    return 4 * n_below >= n_present
end

@inline _truncates(truncfunc::F, term, coeff) where {F} = @inline truncfunc(term, coeff)
@inline _truncates(::Nothing, term, coeff) = false

# Branches every present entry by `rule` together with its partner `term ⊻ mask`, through `branchpair`, entry by entry:
# an entry whose partner is present is mixed with it, and the partner is stamped so that it is not visited again; an
# entry without a present partner keeps what the rule keeps of it, and the partner the rule creates, if `truncfunc`
# keeps it, takes the partner's absent entry or a new one past those of the branch. Every coefficient written is
# judged by `truncfunc`.
function _branchentrybyentry!(rule::F, truncfunc::G, hsum::HashedTermSum{TT,CT}, mask, probe::P) where {F,G,TT,CT,P}
    n_before = hsum.n_entries
    stamp = hsum.stamp + one(Int32)
    hsum.stamp = stamp
    terms = hsum.terms
    keys = hsum.keys
    coeffs = hsum.coeffs
    present = hsum.present
    stamps = hsum.stamps
    slots = hsum.slots
    n_entries = n_before

    @inbounds for entry in 1:n_before
        if !present[entry] || stamps[entry] == stamp
            continue
        end
        term = terms[entry]
        coeff = coeffs[entry]
        partner_term = term ⊻ mask
        partner_key = _partnerkey(probe, keys[entry], partner_term)
        partner, slot = _find(probe, partner_key, partner_term, n_entries + 1)
        stamps[entry] = stamp
        if partner != 0 && present[partner]
            new_coeff, new_partner_coeff, _ = branchpair(rule, term, coeff, partner_term, coeffs[partner])
            keep = !_truncates(truncfunc, term, new_coeff)
            keep_partner = !_truncates(truncfunc, partner_term, new_partner_coeff)
            coeffs[entry] = ifelse(keep, new_coeff, zero(CT))
            coeffs[partner] = ifelse(keep_partner, new_partner_coeff, zero(CT))
            present[entry] = keep
            present[partner] = keep_partner
            stamps[partner] = stamp
        else
            kept, created, creates = branchpair(rule, term, coeff, partner_term, zero(CT))
            keep = !_truncates(truncfunc, term, kept)
            coeffs[entry] = ifelse(keep, kept, zero(CT))
            present[entry] = keep
            if creates && !_truncates(truncfunc, partner_term, created)
                if partner == 0
                    n_entries += 1
                    partner = n_entries
                    terms[partner] = partner_term
                    keys[partner] = partner_key
                    slots[slot+1] = partner % Int32
                end
                coeffs[partner] = created
                present[partner] = true
                stamps[partner] = stamp
            end
        end
    end
    hsum.n_entries = n_entries
    return hsum
end

# The same branch in three passes over the entries, so that no pass branches on what the rule does to an entry. The
# first finds the partner of every entry and lists the events: the lower entry of every pair of present partners, and
# every present entry without a present partner whose coefficient reaches `floor`. The second branches the events
# through `branchpair`, where a partner that is absent or missing holds zero, and gives a partner the rule creates its
# absent entry or a new one. The third gives every other present entry what the rule keeps of it, which is all a
# coefficient below the floor can come to.
function _branchinpasses!(rule::F, truncfunc::G, hsum::HashedTermSum{TT,CT}, mask, probe::P, floor) where {F,G,TT,CT,P}
    n_before = hsum.n_entries
    stamp = hsum.stamp + one(Int32)
    hsum.stamp = stamp
    terms = hsum.terms
    keys = hsum.keys
    coeffs = hsum.coeffs
    present = hsum.present
    stamps = hsum.stamps
    events = hsum.events
    slots = hsum.slots

    # the scratch entry reads as absent, so that a missing partner does
    sentinel = n_before + 1
    @inbounds present[sentinel] = false
    n_events = 0
    @inbounds for entry in 1:n_before
        partner_term = terms[entry] ⊻ mask
        partner_key = _partnerkey(probe, keys[entry], partner_term)
        partner, _ = _find(probe, partner_key, partner_term, sentinel)
        is_present = present[entry]
        partner_present = present[ifelse(partner == 0, sentinel, partner)]
        is_lower_of_pair = is_present & partner_present & (partner > entry)
        creates = is_present & !partner_present & !truncatemincoeff(coeffs[entry], floor)
        events[n_events+1] = entry | (partner << 32)
        n_events += is_lower_of_pair | creates
    end

    n_entries = n_before
    @inbounds for event in 1:n_events
        entry = events[event] & 0xffffffff
        partner = events[event] >> 32
        term = terms[entry]
        partner_term = term ⊻ mask
        partner_coeff = ifelse(partner == 0, zero(CT), coeffs[max(partner, 1)])
        new_coeff, new_partner_coeff, acts = branchpair(rule, term, coeffs[entry], partner_term, partner_coeff)
        keep = !_truncates(truncfunc, term, new_coeff)
        coeffs[entry] = ifelse(keep, new_coeff, zero(CT))
        present[entry] = keep
        stamps[entry] = stamp
        if partner != 0
            keep_partner = (acts | present[partner]) && !_truncates(truncfunc, partner_term, new_partner_coeff)
            coeffs[partner] = ifelse(keep_partner, new_partner_coeff, zero(CT))
            present[partner] = keep_partner
            stamps[partner] = stamp
        elseif acts && !_truncates(truncfunc, partner_term, new_partner_coeff)
            partner_key = _partnerkey(probe, keys[entry], partner_term)
            _, slot = _find(probe, partner_key, partner_term, n_entries + 1)
            n_entries += 1
            terms[n_entries] = partner_term
            keys[n_entries] = partner_key
            coeffs[n_entries] = new_partner_coeff
            present[n_entries] = true
            stamps[n_entries] = stamp
            slots[slot+1] = n_entries % Int32
        end
    end

    @inbounds for entry in 1:n_before
        term = terms[entry]
        coeff = coeffs[entry]
        scales = present[entry] & (stamps[entry] != stamp)
        kept, _, _ = _keptandcreated(@inline(rule(term, coeff)), coeff)
        keep = !_truncates(truncfunc, term, kept)
        coeffs[entry] = ifelse(scales, ifelse(keep, kept, zero(CT)), coeff)
        present[entry] = ifelse(scales, keep, present[entry])
    end
    hsum.n_entries = n_entries
    return hsum
end
