###
##
# A transposed view of the Pauli strings in a sum: one bitmap column per symplectic bit position,
# with one bit per term.
#
# A term anticommutes with a gate generator exactly when an odd number of the columns named by
# `columnsof(gate_mask)` have its bit set, so XORing those columns marks every branching term in one
# sweep of `ncolumns * nterms / 64` words. That is one or two columns per gate qubit, against a read
# of every Pauli string in the sum.
#
# The index is append-only: a Pauli rotation never rewrites a term it keeps, so the columns stay
# valid and only the terms appended since the last call have to be added.
##
###

"""
    TransposedIndex

Bitmap columns over the terms of a Pauli sum, one per symplectic bit position (see
`symplecticword`). Column `c` is `nwords` consecutive words, and bit `i` of it says whether term `i`
sets bit `c`. Terms are indexed by `appendterms!` and read by `foreachbranching`.
"""
mutable struct TransposedIndex
    ncolumns::Int
    nwords::Int     # words per column, so 64 * nwords terms fit
    nterms::Int     # terms indexed so far
    words::Vector{UInt64}
end

function TransposedIndex(nqubits::Int, capacity::Int=0)
    ncolumns = 2 * nqubits
    nwords = cld(max(capacity, 1), 64)
    return TransposedIndex(ncolumns, nwords, 0, zeros(UInt64, ncolumns * nwords))
end

Base.length(index::TransposedIndex) = index.nterms

"""
    reserve!(index::TransposedIndex, capacity::Int)

Make room for `capacity` terms, moving the columns apart if they have to grow. Costs a copy of the
index, so callers grow it geometrically.
"""
function reserve!(index::TransposedIndex, capacity::Int)
    nwords = cld(max(capacity, 1), 64)
    nwords <= index.nwords && return index

    old_words, old_nwords = index.words, index.nwords
    words = zeros(UInt64, index.ncolumns * nwords)
    # highest column first would also do, but the columns do not overlap in the new array anyway
    for c in 1:index.ncolumns
        copyto!(words, (c - 1) * nwords + 1, old_words, (c - 1) * old_nwords + 1, old_nwords)
    end

    index.words = words
    index.nwords = nwords
    return index
end

"""
    empty!(index::TransposedIndex)

Drop every indexed term, keeping the space the columns already have.
"""
function Base.empty!(index::TransposedIndex)
    fill!(index.words, zero(UInt64))
    index.nterms = 0
    return index
end

"""
    appendterms!(index::TransposedIndex, terms::Vector, upto::Int)

Index `terms` up to position `upto`, starting from wherever the last call stopped. Each term costs
one bit per non-identity Pauli it carries, plus one more for every Y or Z.
"""
function appendterms!(index::TransposedIndex, terms::Vector{TT}, upto::Int) where {TT<:Unsigned}
    upto <= index.nterms && return index
    reserve!(index, upto)

    if iswordwidth(TT)
        _appendwords!(index, terms, upto)
    else
        _appendvalues!(index, terms, upto)
    end

    index.nterms = upto
    return index
end

# Word-width Pauli strings are read straight out of the term vector, so nothing shifts a whole
# string, which is what gets slow past a few hundred qubits.
function _appendwords!(index::TransposedIndex, terms::Vector{TT}, upto::Int) where {TT<:Unsigned}
    words, nwords = index.words, index.nwords
    nw = nkeywords(TT)

    GC.@preserve terms begin
        keys = Ptr{UInt64}(pointer(terms))
        row_stride = Base.aligned_sizeof(TT) >> 3

        @inbounds for i in (index.nterms+1):upto
            word = ((i - 1) >> 6) + 1
            bit = one(UInt64) << ((i - 1) & 63)
            row = (i - 1) * row_stride

            for k in 0:(nw-1)
                sym = symplecticword(unsafe_load(keys, row + k + 1))
                column = 64 * k
                while !iszero(sym)
                    words[(column+trailing_zeros(sym))*nwords+word] |= bit
                    sym &= sym - one(UInt64)
                end
            end
        end
    end

    return index
end

function _appendvalues!(index::TransposedIndex, terms::Vector{TT}, upto::Int) where {TT<:Unsigned}
    words, nwords = index.words, index.nwords

    @inbounds for i in (index.nterms+1):upto
        word = ((i - 1) >> 6) + 1
        bit = one(UInt64) << ((i - 1) & 63)

        for (k, key) in enumerate(keywords(terms[i]))
            sym = symplecticword(key)
            column = 64 * (k - 1)
            while !iszero(sym)
                words[(column+trailing_zeros(sym))*nwords+word] |= bit
                sym &= sym - one(UInt64)
            end
        end
    end

    return index
end

"""
    foreachbranching(f, index::TransposedIndex, columns, upto::Int)

Call `f(i)` for every term `i <= upto` whose bits over `columns` have odd parity, in ascending
order. With `columns` from `columnsof(gate_mask)` those are exactly the terms that anticommute with
the gate.
"""
function foreachbranching(f::F, index::TransposedIndex, columns::Vector{Int}, upto::Int) where {F}
    n = length(columns)
    n == 1 && return _foreachbranching(f, index, (columns[1],), upto)
    n == 2 && return _foreachbranching(f, index, (columns[1], columns[2]), upto)
    n == 3 && return _foreachbranching(f, index, (columns[1], columns[2], columns[3]), upto)
    n == 4 && return _foreachbranching(f, index, (columns[1], columns[2], columns[3], columns[4]), upto)
    return _foreachbranching(f, index, columns, upto)
end

# The columns are XORed a word at a time and walked straight away, so the mark bitmap never exists.
@inline function _foreachbranching(f::F, index::TransposedIndex, columns, upto::Int) where {F}
    upto == 0 && return nothing
    words, nwords = index.words, index.nwords
    last_word = ((upto - 1) >> 6) + 1
    # bits past `upto` in the last word belong to terms that are not there yet
    last_mask = (upto & 63) == 0 ? ~zero(UInt64) : (one(UInt64) << (upto & 63)) - one(UInt64)

    @inbounds for w in 1:last_word
        marks = zero(UInt64)
        for c in columns
            marks ⊻= words[(c-1)*nwords+w]
        end
        w == last_word && (marks &= last_mask)

        base = (w - 1) << 6
        while !iszero(marks)
            f(base + trailing_zeros(marks) + 1)
            marks &= marks - one(UInt64)
        end
    end

    return nothing
end
