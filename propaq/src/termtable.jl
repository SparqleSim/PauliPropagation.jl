###
##
# An open-addressed table from Pauli string to the position it holds in the term vector.
#
# It replaces the sort-and-merge that a vector-backed sum does after every gate: a product is
# looked up once and either added onto the term already there or appended, so terms the gate did
# not touch are never read and never moved.
#
# A slot is one word, the position in its low half and a hash tag in its high half. The tag
# settles almost every mismatch without reading the Pauli string, so a probe is one random access.
##
###

const _MAX_LOAD = 0.7

"""
    TermTable

Maps the Pauli strings of a term vector to their positions in it. Positions are `UInt32`, so a sum
can hold up to 2^32 - 1 terms. The table stores no Pauli strings of its own and is only meaningful
next to the vector it was filled from.
"""
mutable struct TermTable
    slots::Vector{UInt64}
    mask::UInt64
    nfilled::Int
end

function TermTable(capacity::Int=0)
    nslots = _slotsfor(capacity)
    return TermTable(zeros(UInt64, nslots), UInt64(nslots - 1), 0)
end

_slotsfor(capacity::Int) = max(64, nextpow(2, max(ceil(Int, capacity / _MAX_LOAD), 1)))

Base.length(table::TermTable) = table.nfilled

"""
    empty!(table::TermTable)

Forget every position, keeping the slots already allocated.
"""
function Base.empty!(table::TermTable)
    fill!(table.slots, zero(UInt64))
    table.nfilled = 0
    return table
end

"""
    termhash(pstr)

Hash of a Pauli string, mixed well enough in its high bits to index a table by and in its low bits
to use as a tag. Reads the string one word at a time rather than as a whole, which keeps it fast
for wide strings.
"""
@inline function termhash(pstr::TT) where {TT<:Unsigned}
    h = zero(UInt64)
    for word in keywords(pstr)
        h = bitrotate(h, 23) ⊻ (word * 0x9e3779b97f4a7c15)
    end
    return h * 0xd6e8feb86659fd93
end

@inline _slotof(h::UInt64, mask::UInt64) = (h >> 32) & mask
@inline _tagof(h::UInt64) = (h % UInt32) | 0x00000001

"""
    findslot(table::TermTable, terms::Vector, pstr, h::UInt64)

Probe for `pstr`, hashed to `h`. Returns `(slot, position)`, where `position` is 0 when the string
is absent and `slot` is then the free slot to claim for it with `slotentry`.
"""
@inline function findslot(table::TermTable, terms::Vector, pstr, h::UInt64)
    return _findslot(table.slots, table.mask, terms, pstr, h)
end

@inline function _findslot(slots::Vector{UInt64}, mask::UInt64, terms::Vector, pstr, h::UInt64)
    slot = _slotof(h, mask)
    tag = _tagof(h)
    @inbounds while true
        entry = slots[slot+1]
        entry == zero(UInt64) && return (Int(slot) + 1, 0)
        if (entry % UInt32) == tag && terms[entry>>32] == pstr
            return (Int(slot) + 1, Int(entry >> 32))
        end
        slot = (slot + one(UInt64)) & mask
    end
end

"""
    slotentry(h::UInt64, position::Int)

The slot contents for a Pauli string hashed to `h` and held at `position`. Callers write it into the
free slot `findslot` handed back and count the entry themselves.
"""
@inline slotentry(h::UInt64, position::Int) = (UInt64(position) << 32) | UInt64(_tagof(h))

"""
    reserve!(table::TermTable, terms::Vector, nterms::Int, capacity::Int)

Make sure `capacity` entries fit without exceeding the load factor, refilling the table from
`terms[1:nterms]` if it has to grow.
"""
function reserve!(table::TermTable, terms::Vector, nterms::Int, capacity::Int)
    nslots = _slotsfor(capacity)
    nslots <= length(table.slots) && return table

    table.slots = zeros(UInt64, nslots)
    table.mask = UInt64(nslots - 1)
    table.nfilled = 0
    return refill!(table, terms, nterms)
end

"""
    refill!(table::TermTable, terms::Vector, nterms::Int)

Fill an emptied table with the positions of `terms[1:nterms]`.
"""
function refill!(table::TermTable, terms::Vector, nterms::Int)
    slots, mask = table.slots, table.mask
    @inbounds for i in 1:nterms
        h = termhash(terms[i])
        slot = _slotof(h, mask)
        while slots[slot+1] != zero(UInt64)
            slot = (slot + one(UInt64)) & mask
        end
        slots[slot+1] = slotentry(h, i)
    end
    table.nfilled = nterms
    return table
end
