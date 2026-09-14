###
##
# A propagation cache that carries a transposed index and a term table next to the term vector.
#
# The term vector, and the over-allocation around it, are the same as in `VectorPauliPropagationCache`.
# What changes is that a gate no longer reads or rewrites the terms it leaves alone: the index picks
# out the branching terms and the table absorbs the products, so both passes cost the branching
# fraction of the sum rather than all of it.
#
# Truncated terms are zeroed rather than removed, because removing one would move every term behind
# it and invalidate both indices. `compact!` clears them out once enough have accumulated.
##
###

# terms are compacted away once this fraction of the sum is zeroed
const _COMPACT_DEAD_FRACTION = 0.25

# the term vector grows by this factor when it runs out of room
const _GROWTH_FACTOR = 1.5

"""
    IndexedPauliPropagationCache

Propagation cache holding a `VectorPauliSum` together with a `TransposedIndex` over its Pauli
strings and a `TermTable` from Pauli string to position. Built from any Pauli sum, and converted
back with `VectorPauliSum(cache)` or `PauliSum(cache)`.
"""
mutable struct IndexedPauliPropagationCache{VPS<:VectorPauliSum} <: PauliPropagation.AbstractPauliPropagationCache
    psum::VPS
    aux_psum::VPS
    table::TermTable
    index::TransposedIndex
    marks::Vector{UInt64}
    handled::Vector{UInt64}
    active_size::Int
    n_dead::Int
end

function IndexedPauliPropagationCache(vpsum::VectorPauliSum)
    vpsum = merge(vpsum)
    n = length(vpsum)

    table = TermTable(n)
    refill!(table, terms(vpsum), n)

    index = TransposedIndex(nqubits(vpsum), n)
    appendterms!(index, terms(vpsum), n)

    aux_psum = VectorPauliSum(nqubits(vpsum), similar(terms(vpsum), 0), similar(coefficients(vpsum), 0))

    nwords = cld(max(n, 1), 64)
    return IndexedPauliPropagationCache(vpsum, aux_psum, table, index,
        zeros(UInt64, nwords), zeros(UInt64, nwords), n, 0)
end

IndexedPauliPropagationCache(psum::PauliPropagation.AbstractPauliSum) = IndexedPauliPropagationCache(VectorPauliSum(psum))
IndexedPauliPropagationCache(pstr::PauliString) = IndexedPauliPropagationCache(VectorPauliSum(pstr))

PropagationBase.activesize(cache::IndexedPauliPropagationCache) = cache.active_size
PropagationBase.setactivesize!(cache::IndexedPauliPropagationCache, n::Int) = (cache.active_size = n; cache)

"""
    nliveterms(cache::IndexedPauliPropagationCache)

The number of terms with a non-zero coefficient. `activesize` also counts the truncated terms that
`compact!` has not cleared out yet.
"""
nliveterms(cache::IndexedPauliPropagationCache) = cache.active_size - cache.n_dead

function Base.resize!(cache::IndexedPauliPropagationCache, n::Int)
    resize!(cache.psum, n)
    return cache
end

function Base.show(io::IO, cache::IndexedPauliPropagationCache)
    print(io, "IndexedPauliPropagationCache with $(nliveterms(cache)) terms")
    cache.n_dead == 0 || print(io, " and $(cache.n_dead) truncated")
    print(io, " on $(nqubits(cache)) qubits")
    return
end

"""
    getcoeff(cache::IndexedPauliPropagationCache, pstr)

The coefficient of `pstr`, found through the term table rather than by searching the term vector.
"""
function PauliPropagation.getcoeff(cache::IndexedPauliPropagationCache, pstr)
    trms = terms(mainsum(cache))
    _, position = findslot(cache.table, trms, pstr, termhash(pstr))
    return position == 0 ? zero(coefftype(mainsum(cache))) : coefficients(mainsum(cache))[position]
end

### Room for more terms

# Makes room for `n` terms in the term vector, the table and the index. Any of the three may move,
# so callers re-read the arrays after this.
#
# The index is sized from the term vector's capacity rather than from `n`: growing it moves all of
# its columns, and one more word per column buys only 64 more terms, so sizing it from `n` pays that
# every time the sum passes another multiple of 64. The table only ever doubles, so `n` is fine for
# it, and keeps its load factor up.
function _reserve!(cache::IndexedPauliPropagationCache, n::Int)
    if n > length(mainsum(cache))
        resize!(cache, max(n, ceil(Int, length(mainsum(cache)) * _GROWTH_FACTOR)))
    end

    capacity = length(mainsum(cache))
    reserve!(cache.table, terms(mainsum(cache)), cache.active_size, n)
    reserve!(cache.index, capacity)
    if length(cache.marks) < cld(capacity, 64)
        resize!(cache.marks, cld(capacity, 64))
        resize!(cache.handled, cld(capacity, 64))
    end
    return cache
end

# Wipes the record of which terms a gate has already paired up, for the first `n` of them.
@inline function _clearhandled!(cache::IndexedPauliPropagationCache, n::Int)
    fill!(view(cache.handled, 1:(((n - 1) >> 6) + 1)), zero(UInt64))
    return cache
end

### Clearing out truncated terms

"""
    compact!(cache::IndexedPauliPropagationCache)

Drop the terms whose coefficients were zeroed by truncation and rebuild both indices around the
terms that remain. This is the one pass over the whole sum that the design still needs, so it runs
only once truncation has cost enough memory to be worth it.
"""
function compact!(cache::IndexedPauliPropagationCache)
    cache.n_dead == 0 && return cache

    main_terms, main_coeffs = storage(mainsum(cache))
    n_live = nliveterms(cache)

    resize!(auxsum(cache), n_live)
    aux_terms, aux_coeffs = storage(auxsum(cache))

    pos = 0
    @inbounds for i in 1:cache.active_size
        coeff = main_coeffs[i]
        iszero(coeff) && continue
        pos += 1
        aux_terms[pos] = main_terms[i]
        aux_coeffs[pos] = coeff
    end

    swapsums!(cache)
    cache.active_size = pos
    cache.n_dead = 0

    empty!(cache.table)
    reserve!(cache.table, terms(mainsum(cache)), 0, pos)
    refill!(cache.table, terms(mainsum(cache)), pos)

    empty!(cache.index)
    appendterms!(cache.index, terms(mainsum(cache)), pos)

    return cache
end

# Compacts once truncation has zeroed enough of the sum to pay for the rebuild.
@inline function _maybecompact!(cache::IndexedPauliPropagationCache)
    cache.n_dead > _COMPACT_DEAD_FRACTION * cache.active_size && compact!(cache)
    return cache
end

### Back-conversions

function PauliPropagation.VectorPauliSum(cache::IndexedPauliPropagationCache)
    compact!(cache)
    vpsum = deepcopy(mainsum(cache))
    resize!(vpsum, cache.active_size)
    return vpsum
end

function PauliPropagation.PauliSum(cache::IndexedPauliPropagationCache)
    compact!(cache)
    return PauliSum(nqubits(cache), Dict(zip(view(terms(mainsum(cache)), 1:cache.active_size),
        view(coefficients(mainsum(cache)), 1:cache.active_size))))
end

function PropagationBase.extractsum!(cache::IndexedPauliPropagationCache)
    compact!(cache)
    resize!(cache, cache.active_size)
    setsortedprefix!(mainsum(cache), 0)
    return mainsum(cache)
end
