###
##
# The Pauli operations of bitoperations.jl on an `NTupleInteger`, one limb at a time.
# A qubit's two bits never straddle a limb, so commutation, sign, weight and a single Pauli decompose over the limbs with no carries, where the generic versions shift and mask the whole value several times.
##
###

# ...0101, the low bit of every Pauli pair in one limb
const _ALTLIMB = 0x5555555555555555

alternatingmask(::NTupleInteger{N}) where {N} = NTupleInteger{N}(ntuple(_ -> _ALTLIMB, Val(N)))

# the parity of the anticommuting pairs adds under xor, so the flag limbs are folded first
@generated function _bitcommutes(pstr1::NTupleInteger{N}, pstr2::NTupleInteger{N}) where {N}
    return quote
        Base.@_inline_meta
        flags = zero(UInt64)
        Base.Cartesian.@nexprs $N k -> begin
            a = pstr1.limbs[k]
            b = pstr2.limbs[k]
            flags ⊻= ((a & (b >> 1)) ⊻ ((a >> 1) & b)) & _ALTLIMB
        end
        return iseven(count_ones(flags))
    end
end

# the exponent of `_calculatesignexponent`, with the two counts taken over all limbs
@generated function _calculatesignexponent(pauli1::NTupleInteger{N}, pauli2::NTupleInteger{N}) where {N}
    return quote
        Base.@_inline_meta
        n_anticommuting = 0
        n_negative = 0
        Base.Cartesian.@nexprs $N k -> begin
            pauli1_1 = (pauli1.limbs[k] >> 1) & _ALTLIMB
            pauli1_2 = pauli1.limbs[k] & _ALTLIMB
            pauli2_1 = (pauli2.limbs[k] >> 1) & _ALTLIMB
            pauli2_2 = pauli2.limbs[k] & _ALTLIMB
            not_commuting = (pauli1_1 | pauli1_2) & (pauli2_1 | pauli2_2) & ((pauli1_1 ⊻ pauli2_1) | (pauli1_2 ⊻ pauli2_2))
            negative_sign = not_commuting & ((pauli1_1 ⊻ pauli2_2) | (~pauli1_2 & ~pauli2_1))
            n_anticommuting += count_ones(not_commuting)
            n_negative += count_ones(negative_sign)
        end
        return (2 * n_negative + n_anticommuting) & 3
    end
end

# the number of Pauli pairs `perlimb` flags, summed over the limbs
@generated function _countlimbs(perlimb::F, pstr::NTupleInteger{N}) where {F,N}
    return quote
        Base.@_inline_meta
        n = 0
        Base.Cartesian.@nexprs $N k -> (n += count_ones(perlimb(pstr.limbs[k])))
        return n
    end
end

# the low bit of a pair holds 1 for X and Z, the high bit for Y and Z
@inline _weightlimb(w::UInt64) = (w | (w >> 1)) & _ALTLIMB
@inline _xylimb(w::UInt64) = (w ⊻ (w >> 1)) & _ALTLIMB
@inline _yzlimb(w::UInt64) = (w >> 1) & _ALTLIMB
@inline _xlimb(w::UInt64) = (w & ~(w >> 1)) & _ALTLIMB
@inline _ylimb(w::UInt64) = ((w >> 1) & ~w) & _ALTLIMB
@inline _zlimb(w::UInt64) = (w & (w >> 1)) & _ALTLIMB

_countbitweight(pstr::NTupleInteger) = _countlimbs(_weightlimb, pstr)
_countbitxy(pstr::NTupleInteger) = _countlimbs(_xylimb, pstr)
_countbityz(pstr::NTupleInteger) = _countlimbs(_yzlimb, pstr)
_countbitx(pstr::NTupleInteger) = _countlimbs(_xlimb, pstr)
_countbity(pstr::NTupleInteger) = _countlimbs(_ylimb, pstr)
_countbitz(pstr::NTupleInteger) = _countlimbs(_zlimb, pstr)

# A Pauli of a chunked string comes back as a `UInt64`, and so do up to 32 Paulis at once, since
# nothing reads them from the whole value. The limb index is checked against the tuple.
@inline function _getpaulibits(pstr::NTupleInteger, index::Integer)
    bit = _bitshiftfromsiteindex(index)
    limb = pstr.limbs[(bit>>6)+1]
    return (limb >> (bit & 63)) & 3
end

@inline function _getpaulibits(pstr::NTupleInteger{N}, index1::Integer, index2::Integer) where {N}
    n_sites = index2 - index1 + 1
    if n_sites > 32
        throw(ArgumentError("At most 32 Paulis of a chunked Pauli string can be read at once. Got $n_sites."))
    end
    bit = _bitshiftfromsiteindex(index1)
    k = (bit >> 6) + 1
    offset = bit & 63
    low = pstr.limbs[k]
    high = k < N ? pstr.limbs[k+1] : zero(UInt64)
    window = (low >> offset) | (high << (64 - offset))
    return window & ((one(UInt64) << (2 * n_sites)) - one(UInt64))
end

@inline function _setpaulibits(pstr::NTupleInteger{N}, target_pauli::PauliType, index::Integer) where {N}
    bit = _bitshiftfromsiteindex(index)
    k = (bit >> 6) + 1
    offset = bit & 63
    limb = (pstr.limbs[k] & ~(UInt64(3) << offset)) | (((target_pauli % UInt64) & 3) << offset)
    return NTupleInteger{N}(ntuple(j -> ifelse(j == k, limb, pstr.limbs[j]), Val(N)))
end

# the low `2 * n_sites` bits set, limb by limb: full below the cut, empty above it, partial across it
function _paulimask(::Type{NTupleInteger{N}}, n_sites) where {N}
    nbits = 2 * n_sites
    return NTupleInteger{N}(ntuple(k -> typemax(UInt64) >> clamp(64 * k - nbits, 0, 64), Val(N)))
end
