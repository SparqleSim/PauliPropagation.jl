###
##
# The symplectic view of a Pauli string, one 64-bit word at a time.
#
# A Pauli sits in two bits, `01` for X, `10` for Y, `11` for Z. Rewriting that pair as the
# symplectic pair (x, z) -- x set for X and Y, z set for Y and Z -- leaves x on the even bit and z
# on the odd bit, and makes anticommutation the parity of `sym(a) & pairswap(sym(b))`. Both bits of
# a qubit live in the same word, so every function here works word by word and never shifts a whole
# Pauli string, which gets slow past a few hundred qubits.
##
###

# ...0101 and ...1010: the even and the odd bit of every Pauli pair
const _EVEN_BITS = 0x5555555555555555
const _ODD_BITS = 0xaaaaaaaaaaaaaaaa

"""
    symplecticword(word::UInt64)

The symplectic form of one word of a Pauli string: bit `2q` says the Pauli at `q` is X or Y, bit
`2q+1` says it is Y or Z.
"""
@inline symplecticword(word::UInt64) = ((word ⊻ (word >> 1)) & _EVEN_BITS) | (word & _ODD_BITS)

"""
    pairswapword(word::UInt64)

One word of a Pauli string with the two bits of every qubit exchanged.
"""
@inline pairswapword(word::UInt64) = ((word & _EVEN_BITS) << 1) | ((word >> 1) & _EVEN_BITS)

"""
    keywords(pstr)

The 64-bit words of the Pauli string `pstr`, low word first, as a tuple. Pauli strings narrower
than a word are widened to one; `getinttype` also hands out widths that are not a whole number of
words, which are read by shifting instead of reinterpreting.
"""
@inline keywords(pstr::TT) where {TT<:Unsigned} = _keywords(pstr, Val(nkeywords(TT)))

@inline _keywords(pstr::TT, ::Val{1}) where {TT<:Unsigned} = (pstr % UInt64,)

@inline function _keywords(pstr::TT, ::Val{N}) where {TT<:Unsigned,N}
    iswordwidth(TT) && return reinterpret(NTuple{N,UInt64}, pstr)
    return ntuple(k -> (pstr >> (64 * (k - 1))) % UInt64, Val(N))
end

"""
    nkeywords(::Type{TT})

The number of 64-bit words `keywords` returns for Pauli strings of type `TT`.
"""
@inline nkeywords(::Type{TT}) where {TT<:Unsigned} = max(cld(sizeof(TT), 8), 1)

"""
    iswordwidth(::Type{TT})

Whether Pauli strings of type `TT` are a whole number of 64-bit words, and so can be read straight
out of a term vector word by word.
"""
@inline iswordwidth(::Type{TT}) where {TT<:Unsigned} = sizeof(TT) >= 8 && sizeof(TT) % 8 == 0

"""
    columnsof(gate_mask)

The bit-index columns whose parity decides whether a Pauli string anticommutes with `gate_mask`,
as 1-based column numbers. A generator of X or Z on a qubit contributes one column, Y contributes
two, so a two-qubit ZZ rotation reads two columns of the whole pool.
"""
function columnsof(gate_mask::TT) where {TT<:Unsigned}
    cols = Int[]
    for (k, word) in enumerate(keywords(gate_mask))
        detector = pairswapword(symplecticword(word))
        while !iszero(detector)
            push!(cols, 64 * (k - 1) + trailing_zeros(detector) + 1)
            detector &= detector - one(UInt64)
        end
    end
    return cols
end
