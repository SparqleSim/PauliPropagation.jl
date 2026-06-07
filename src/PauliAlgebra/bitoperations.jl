"""
    getinttype(nqubits::Integer; use_multiuint::Bool=true, word::Type=UInt64)

Return the smallest integer type that can hold `nqubits` qubits.

For `nqubits ≤ 32` (i.e. ≤ 64 bits), returns the same native type as
before (e.g. `getinttype(17) === UInt40`): **bit-for-bit identical to
prior behaviour**.

For `nqubits > 32` (> 64 bits), returns `MultiUInt{N, word}` where `N`
is the smallest number of `word`-sized words that holds `2*nqubits` bits.
`MultiUInt` is `isbitstype`, lives in GPU registers, and supports the
full bitwise contract this codebase relies on — fixing the CUDA extension's
>32-qubit limitation (issue #145).

**Word width (`word` kwarg):**
- `word=UInt64` (default): best throughput on CPU and server GPUs.
- `word=UInt32`: native register width on consumer NVIDIA GPUs (RTX series).
  Use when building `VectorPauliSum` arrays destined for `CuArray` on GPUs
  that execute 32-bit operations natively.

**Fallback (`use_multiuint=false`):** reverts to the BitIntegers.jl path
(dynamic `@define_integers` above 64 bits). Useful for benchmarking and
reproducing pre-issue-#145 results; does **not** work on GPU.

```julia
getinttype(32)                      # UInt64          (unchanged)
getinttype(33)                      # MultiUInt{2, UInt64}   (was UInt66 — now GPU-safe)
getinttype(256)                     # MultiUInt{8, UInt64}   (partial-reward target)
getinttype(256; word=UInt32)        # MultiUInt{16, UInt32}  (32-bit GPU registers)
getinttype(33; use_multiuint=false) # UInt66 (legacy BitIntegers path)
```
"""
function getinttype(nqubits::Integer; use_multiuint::Bool=true, word::Type=UInt64)
    # We need 2 bits per qubit.
    nbits = 2 * nqubits

    # ≤ 64 bits: preserve the exact historic return type — including non-power-of-two
    # widths like UInt40 — because existing tests depend on the contract
    # `getinttype(17) === UInt40`.
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
        return UInt64  # defensive fallback; unreachable in practice
    end

    # > 64 bits: MultiUInt by default (GPU-compatible, issue #145).
    if use_multiuint
        @assert word ∈ (UInt32, UInt64) "word must be UInt32 or UInt64, got $word"
        nwords = cld(nbits, 8 * sizeof(word))
        return MultiUInt{nwords, word}
    end

    # Legacy BitIntegers path — benchmarking / reproducing pre-#145 results only.
    for trial_bits in nbits:2:8_300_000
        trial_bits % 8 != 0 && continue
        trial_inttype_expr = Symbol("UInt", trial_bits)
        isdefined(PauliPropagation, trial_inttype_expr) && return eval(trial_inttype_expr)
        try
            @eval @define_integers $trial_bits
            return eval(trial_inttype_expr)
        catch ErrorException
            continue
        end
    end

    @warn "Failed to define integer types for $nqubits qubits. Falling back to BigInt."
    return BigInt
end


# Fast hash for BitIntegers wide unsigned types — matches value semantics.
Base.hash(v::BitIntegers.AbstractBitUnsigned, h::UInt) = Base.hash_integer(v, h)


# ---------------------------------------------------------------------------
# The functions below operate on PauliStringType (= Union{Integer, MultiUInt}).
# Because MultiUInt <: Unsigned <: Integer, a single generic dispatch covers
# both paths — no separate overloads needed.
# ---------------------------------------------------------------------------

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

# alternatingmask: compile-time constant for both native integers and MultiUInt.
# The MultiUInt specialization is defined in multiuint.jl (via @generated).
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
