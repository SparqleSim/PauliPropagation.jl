"""
    getinttype(nqubits::Integer)

Return the smallest integer type that can hold `nqubits` qubits.

For `nqubits <= 32` (up to 64 bits), returns the same native type as before.
For `nqubits > 32`, returns `NTupleUInt{N, UInt64}` — an `isbits` tuple-backed
type that works on GPU where `BitIntegers.jl` types cannot.
"""
function getinttype(nqubits::Integer)
    # we need 2 bits per qubit
    nbits = 2 * nqubits

    # up to 64 bits: use native or BitIntegers types as before
    if nbits <= 64
        for trial_bits in nbits:2:64
            trial_bits % 8 != 0 && continue
            trial_bits == 8  && return UInt8
            trial_bits == 16 && return UInt16
            trial_bits == 32 && return UInt32
            trial_bits == 64 && return UInt64
            trial_inttype_expr = Symbol("UInt", trial_bits)
            isdefined(PauliPropagation, trial_inttype_expr) && return eval(trial_inttype_expr)
            try
                @eval @define_integers $trial_bits
                return eval(trial_inttype_expr)
            catch ErrorException
                continue
            end
        end
        return UInt64
    end

    # above 64 bits: use NTupleUInt which is GPU-compatible
    return getchunkedinttype(nqubits)
end


# Fast hash for BitIntegers wide unsigned types — matches value semantics.
Base.hash(v::BitIntegers.AbstractBitUnsigned, h::UInt) = Base.hash_integer(v, h)

# Count non-identity Paulis: a site is non-identity iff either of its 2 bits is 1.
function _countbitweight(pstr::PauliStringType)
    mask = alternatingmask(pstr)
    m1 = pstr & mask
    m2 = pstr & (mask << 1)
    res = m1 | (m2 >> 1)
    return count_ones(res)
end

# Count X (01) or Y (10) Paulis — each has exactly one bit set.
function _countbitxy(pstr::PauliStringType)
    mask = alternatingmask(pstr)
    op = pstr ⊻ (pstr >> 1)
    op = op & mask
    return count_ones(op)
end

# Count Y (10) or Z (11) Paulis — both have the high bit set.
function _countbityz(pstr::PauliStringType)
    mask = alternatingmask(pstr)
    op = pstr & (mask << 1)
    return count_ones(op)
end

function _countbitx(pstr::PauliStringType)
    mask_x = alternatingmask(pstr)
    mask_y = mask_x << 1
    xs = (pstr & mask_x) & ((~pstr & mask_y) >> 1)
    return count_ones(xs)
end

function _countbity(pstr::PauliStringType)
    mask_x = alternatingmask(pstr)
    mask_y = mask_x << 1
    op = ((pstr & mask_y) >> 1) & (~pstr & mask_x)
    return count_ones(op)
end

function _countbitz(pstr::PauliStringType)
    mask_x = alternatingmask(pstr)
    mask_y = mask_x << 1
    op = ((pstr & mask_y) >> 1) & (pstr & mask_x)
    return count_ones(op)
end

# Commutativity check: strings commute iff the number of site-wise anti-commuting
# Pauli pairs is even.
function _bitcommutes(pstr1::PauliStringType, pstr2::PauliStringType)
    mask0 = alternatingmask(pstr1)
    mask1 = mask0 << 1
    aBits0 = mask0 & pstr1
    aBits1 = mask1 & pstr1
    bBits0 = mask0 & pstr2
    bBits1 = mask1 & pstr2
    aBits1 = aBits1 >> 1
    bBits1 = bBits1 >> 1
    flags = (aBits0 & bBits1) ⊻ (aBits1 & bBits0)
    return (count_ones(flags) % 2) == 0
end

_bitpaulimultiply(pstr1::PauliStringType, pstr2::PauliStringType) = pstr1 ⊻ pstr2

_paulishiftright(pstr::PauliStringType) = pstr >> 2

_bitshiftfromsiteindex(siteindex::Integer) = 2 * (siteindex - 1)

# Generic path (Integer / BitIntegers types): delegates to Bits.mask.
_paulimask(::Type{T}, n_sites) where T = mask(T, 2 * n_sites)

_pauliwindowmask(::Type{T}, index1::Integer, index2::Integer) where T =
    _paulimask(T, index2 - index1 + 1) << _bitshiftfromsiteindex(index1)

function _getpaulibits(pstr::PauliStringType, index::Integer)
    return _getpaulibits(pstr, index, index)
end

function _getpaulibits(pstr::PauliStringType, index1::Integer, index2::Integer)
    T = typeof(pstr)
    bitindex = _bitshiftfromsiteindex(index1)
    shifted_pstr = (pstr >> bitindex)
    return shifted_pstr & _paulimask(T, index2 - index1 + 1)
end

function _setpaulibits(pstr::PauliStringType, target_pauli::PauliType, index::Integer)
    return _setpaulibits(pstr, target_pauli, index, index)
end

function _setpaulibits(pstr::PauliStringType, target_pstr::PauliStringType, index1::Integer, index2::Integer)
    T = typeof(pstr)
    bitindex = _bitshiftfromsiteindex(index1)
    window_mask = _pauliwindowmask(T, index1, index2)
    return (pstr & ~window_mask) | (T(target_pstr) << bitindex)
end

# alternatingmask: compile-time constant for both native integers and NTupleUInt.
# The NTupleUInt specialization is defined in NTuplePauliString.jl (via @generated).
# This @generated covers native integers (Integer / BitIntegers) and keeps
# the existing behaviour for those types.
@generated function alternatingmask(pstr::T) where {T<:PauliStringType}
    n_bits = min(bitsize(T), 2_048)  # cap at 1024 qubits
    mask_val = zero(T)
    for ii in 0:(n_bits - 1)
        if ii % 2 == 0
            mask_val = mask_val | (T(1) << ii)
        end
    end
    return mask_val
end