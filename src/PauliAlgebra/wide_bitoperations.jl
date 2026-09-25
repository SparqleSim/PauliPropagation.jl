###
##
# The Pauli operations of bitoperations.jl on an `NTupleInteger`, one limb at a time.
# A qubit's two bits never straddle a limb, so commutation, sign, weight and a single Pauli decompose over the limbs with no carries, where the generic versions shift and mask the whole value several times.
##
###

# ...0101, the low bit of every Pauli pair in one limb
const _ALTLIMB = 0x5555555555555555

alternatingmask(::NTupleInteger{N}) where {N} = NTupleInteger{N}(ntuple(_ -> _ALTLIMB, Val(N)))

# Two Paulis anticommute when the low bit of one meets the high bit of the other exactly once, so the parity over all
# pairs is that of the first string with the two bits of every pair swapped, ANDed with the second. The swap reads the
# first string alone, so a walk that keeps the gate's string first computes it once.
@generated function _bitcommutes(pstr1::NTupleInteger{N}, pstr2::NTupleInteger{N}) where {N}
    return quote
        Base.@_inline_meta
        flags = zero(UInt64)
        Base.Cartesian.@nexprs $N k -> (flags ⊻= _swappairbits(pstr1.limbs[k]) & pstr2.limbs[k])
        return iseven(count_ones(flags))
    end
end

@inline _swappairbits(w::UInt64) = ((w >> 1) & _ALTLIMB) | ((w & _ALTLIMB) << 1)

# The exponent of `_calculatesignexponent`. With x = low ⊻ high and z = high for every pair, a pair anticommutes where
# x1 z2 ⊻ z1 x2 is set, and it contributes -i instead of i where x3 ⊻ z3 ⊻ x1 z2 is set as well, with x3 ⊻ z3 = low1 ⊻ low2
# for the product. So the anticommuting pairs are counted, and those contributing -i only by their parity.
@generated function _calculatesignexponent(pauli1::NTupleInteger{N}, pauli2::NTupleInteger{N}) where {N}
    return quote
        Base.@_inline_meta
        n_anticommuting = 0
        negative = zero(UInt64)
        Base.Cartesian.@nexprs $N k -> begin
            p1 = pauli1.limbs[k]
            p2 = pauli2.limbs[k]
            z1 = p1 >> 1
            z2 = p2 >> 1
            x1z2 = (p1 ⊻ z1) & z2
            anticommuting = (x1z2 ⊻ (z1 & (p2 ⊻ z2))) & _ALTLIMB
            n_anticommuting += count_ones(anticommuting)
            negative ⊻= (p1 ⊻ p2 ⊻ x1z2) & anticommuting
        end
        return (n_anticommuting + 2 * count_ones(negative)) & 3
    end
end

# The rotation rules call these for every term. Unrolled over many limbs, the generic methods outgrow the inlining
# budget, which would leave a call per term that also repeats the gate's side of the work.
@inline commutes(pstr1::NTupleInteger{N}, pstr2::NTupleInteger{N}) where {N} = _bitcommutes(pstr1, pstr2)

# the sign as the generic method takes it, real(im * im^exponent)
@inline function paulirotationproduct(gate_mask::NTupleInteger{N}, pstr::NTupleInteger{N}) where {N}
    return _bitpaulimultiply(gate_mask, pstr), (_calculatesignexponent(gate_mask, pstr) & 2) - 1
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

# A Pauli of a chunked string comes back as a `UInt64`, since nothing reads it from the whole value.
# The limb index is checked against the tuple.
@inline function _getpaulibits(pstr::NTupleInteger, index::Integer)
    bit = _bitshiftfromsiteindex(index)
    limb = pstr.limbs[(bit>>6)+1]
    return (limb >> (bit & 63)) & 3
end

# up to 32 Paulis are read from the two limbs they span, more by shifting the whole value
@inline function _getpaulibits(pstr::NTupleInteger{N}, index1::Integer, index2::Integer) where {N}
    n_sites = index2 - index1 + 1
    if n_sites > 32
        return (pstr >> _bitshiftfromsiteindex(index1)) & _paulimask(NTupleInteger{N}, n_sites)
    end
    bit = _bitshiftfromsiteindex(index1)
    k = (bit >> 6) + 1
    offset = bit & 63
    low = pstr.limbs[k]
    high = k < N ? pstr.limbs[k+1] : zero(UInt64)
    window = (low >> offset) | (high << (64 - offset))
    return NTupleInteger{N}(window & ((one(UInt64) << (2 * n_sites)) - one(UInt64)))
end

# Up to 32 Paulis are gathered in one word, which widens once, instead of shifting the whole value in for every Pauli.
# Every Clifford gate reads its Paulis this way.
function getpauli(pstr::NTupleInteger{N}, qinds::Union{AbstractVector,Tuple}) where {N}
    if length(qinds) > 32
        return invoke(getpauli, Tuple{PauliStringType,Any}, pstr, qinds)
    end
    _check_qind_range(maxqubits(pstr), qinds)
    gathered = zero(UInt64)
    for (i, qind) in enumerate(qinds)
        gathered |= _getpaulibits(pstr, qind) << (2 * (i - 1))
    end
    return NTupleInteger{N}(gathered)
end

# Every limb takes the same update where it holds the index, so no limb is read at a runtime position.
# `setpauli` checks the index against the qubits first.
@inline function _setpaulibits(pstr::NTupleInteger{N}, target_pauli::PauliType, index::Integer) where {N}
    bit = _bitshiftfromsiteindex(index)
    k = (bit >> 6) + 1
    offset = bit & 63
    keep = ~(UInt64(3) << offset)
    pauli = ((target_pauli % UInt64) & 3) << offset
    return NTupleInteger{N}(ntuple(j -> ifelse(j == k, (pstr.limbs[j] & keep) | pauli, pstr.limbs[j]), Val(N)))
end

# the low `2 * n_sites` bits set, limb by limb: full below the cut, empty above it, partial across it
function _paulimask(::Type{NTupleInteger{N}}, n_sites) where {N}
    nbits = 2 * n_sites
    return NTupleInteger{N}(ntuple(k -> typemax(UInt64) >> clamp(64 * k - nbits, 0, 64), Val(N)))
end
