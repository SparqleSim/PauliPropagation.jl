### multiuint.jl
##
# A GPU-friendly long unsigned integer built from a fixed-size NTuple of
# native unsigned words. Plug-in replacement for the dynamically-defined
# BitIntegers.jl types in the >64-bit code path of `getinttype`.
#
# Motivation (issue #145):
#   - The CUDA extension breaks above 32 qubits because Pauli strings need
#     >64 bits and BitIntegers' wide types blow up GPU register pressure.
#   - `MultiUInt{N,T}` is `isbits`, fits in N registers of width T, and
#     supports exactly the bitwise contract that PauliPropagation uses.
#
# Design:
#   - Pure value type stored as an NTuple. Pointwise ops `&, |, ⊻, ~`
#     use `ntuple(...)` so the Julia compiler unrolls them at compile time.
#   - Shifts handle inter-word carry correctly for arbitrary amounts.
#   - `alternatingmask` is a compile-time @generated constant (same trick
#     as NTuplePauliString), so the mask is never recomputed at runtime.
#   - Supports both UInt32 and UInt64 words — use UInt32 for GPU register
#     file compatibility on 32-bit-native hardware (e.g., consumer NVIDIA GPUs).
#   - `count_ones` is an explicit loop — no `sum(...; init=...)` which may
#     not specialise in CUDA kernel contexts.
#   - `isless` compares high-word-first (big-endian integer semantics).
#   - `hash` folds word-by-word matching `Base.hash_integer` so that
#     `MultiUInt(0)` hashes identically to `0::Int` — Dict lookups work.
#   - `+`, `-` use ripple-carry tuples — zero allocations, GPU-safe.
#
# Layout: `parts[1]` is the LOW word, `parts[end]` is the HIGH word.
# Left-shift moves bits from `parts[1]` toward `parts[end]`.
###

import Bits: bitsize, mask

"""
    MultiUInt{N, T<:Union{UInt32,UInt64}} <: Unsigned

GPU-friendly long unsigned integer built from `N` words of type `T`.
Total bit width is `N * 8 * sizeof(T)`.

Supports the complete bitwise contract used by PauliPropagation:
`&`, `|`, `⊻`, `~`, `<<`, `>>`, `count_ones`, equality, ordering,
`zero`, `one`, `hash`, `Bits.bitsize`, `Bits.mask`, `+`, `-`.

`parts[1]` is the least-significant word; `parts[end]` is the most
significant.

**Word choice:**
- `T = UInt64`: best throughput on CPU and most server GPUs.
- `T = UInt32`: native register width on consumer NVIDIA GPUs (RTX series);
  use when running 32-bit-native CUDA kernels.

# Examples

```julia
# 256-qubit Pauli string (512 bits) in four UInt64 words
a = MultiUInt{8, UInt64}(0xABCD)

# Same capacity in 16 UInt32 words — better for consumer GPUs
b = MultiUInt{16, UInt32}(0xABCD)

getinttype(256)           # === MultiUInt{8, UInt64}
getinttype(256; word=UInt32)  # === MultiUInt{16, UInt32}
```
"""
struct MultiUInt{N, T<:Union{UInt32,UInt64}} <: Unsigned
    parts::NTuple{N, T}
end

# ---- Construction -----------------------------------------------------

"""
    MultiUInt{N, T}(x::Integer)

Zero-extend `x` into a `MultiUInt`. Throws `DomainError` for negative values.
"""
function MultiUInt{N, T}(x::Integer) where {N, T<:Union{UInt32,UInt64}}
    x >= 0 || throw(DomainError(x, "MultiUInt is unsigned — got negative value $x"))
    word_bits = 8 * sizeof(T)
    # typemax(T) without causing a shift-by-wordsize UB:
    mask_low = (T(1) << (word_bits - 1) - T(1)) << 1 | T(1)   # == typemax(T)
    parts = ntuple(N) do i
        shift = (i - 1) * word_bits
        T((x >> shift) & mask_low)
    end
    return MultiUInt{N, T}(parts)
end

# Self-conversion (identity).
MultiUInt{N, T}(x::MultiUInt{N, T}) where {N, T<:Union{UInt32,UInt64}} = x

# ---- Basic predicates -------------------------------------------------

Base.zero(::Type{MultiUInt{N, T}}) where {N, T} =
    MultiUInt{N, T}(ntuple(_ -> zero(T), Val(N)))
Base.zero(x::MultiUInt) = zero(typeof(x))

Base.one(::Type{MultiUInt{N, T}}) where {N, T} =
    MultiUInt{N, T}(ntuple(i -> i == 1 ? one(T) : zero(T), Val(N)))
Base.one(x::MultiUInt) = one(typeof(x))

Base.iszero(a::MultiUInt) = all(iszero, a.parts)

Base.typemax(::Type{MultiUInt{N, T}}) where {N, T} =
    MultiUInt{N, T}(ntuple(_ -> typemax(T), Val(N)))

# ---- Bitwise: pointwise -----------------------------------------------

Base.:&(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T} =
    MultiUInt{N, T}(ntuple(i -> a.parts[i] & b.parts[i], Val(N)))

Base.:|(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T} =
    MultiUInt{N, T}(ntuple(i -> a.parts[i] | b.parts[i], Val(N)))

Base.xor(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T} =
    MultiUInt{N, T}(ntuple(i -> a.parts[i] ⊻ b.parts[i], Val(N)))

Base.:~(a::MultiUInt{N, T}) where {N, T} =
    MultiUInt{N, T}(ntuple(i -> ~a.parts[i], Val(N)))

# ---- Shifts -----------------------------------------------------------
# Left shift: low->high word migration with inter-word carry.

function _shift_left(a::MultiUInt{N, T}, s::Int) where {N, T}
    s < 0  && return _shift_right(a, -s)
    word_bits = 8 * sizeof(T)
    total_bits = N * word_bits
    s >= total_bits && return zero(MultiUInt{N, T})
    q, r = divrem(s, word_bits)
    new_parts = ntuple(Val(N)) do i
        lo_src = i - q
        hi_src = lo_src - 1
        lo = (1 <= lo_src <= N) ? a.parts[lo_src] : zero(T)
        carry = if r == 0 || hi_src < 1 || hi_src > N
            zero(T)
        else
            a.parts[hi_src] >> (word_bits - r)
        end
        (lo << r) | carry
    end
    return MultiUInt{N, T}(new_parts)
end

function _shift_right(a::MultiUInt{N, T}, s::Int) where {N, T}
    s < 0  && return _shift_left(a, -s)
    word_bits = 8 * sizeof(T)
    total_bits = N * word_bits
    s >= total_bits && return zero(MultiUInt{N, T})
    q, r = divrem(s, word_bits)
    new_parts = ntuple(Val(N)) do i
        lo_src = i + q
        hi_src = lo_src + 1
        lo = (1 <= lo_src <= N) ? a.parts[lo_src] : zero(T)
        carry = if r == 0 || hi_src < 1 || hi_src > N
            zero(T)
        else
            a.parts[hi_src] << (word_bits - r)
        end
        (lo >> r) | carry
    end
    return MultiUInt{N, T}(new_parts)
end

# Explicit Int and Integer overloads disambiguate from Base.<<(::Integer, ::Int).
Base.:<<(a::MultiUInt, s::Int)     = _shift_left(a, s)
Base.:<<(a::MultiUInt, s::Integer) = _shift_left(a, Int(s))
Base.:>>(a::MultiUInt, s::Int)     = _shift_right(a, s)
Base.:>>(a::MultiUInt, s::Integer) = _shift_right(a, Int(s))
# Unsigned right shift is logical for our type — same as >>.
Base.:>>>(a::MultiUInt, s::Int)     = _shift_right(a, s)
Base.:>>>(a::MultiUInt, s::Integer) = _shift_right(a, Int(s))

# ---- Counting / ordering / hash ---------------------------------------

# Explicit loop — inlines cleanly in GPU kernels and avoids the `sum(f; init)` edge case.
function Base.count_ones(a::MultiUInt{N, T}) where {N, T}
    s = 0
    @inbounds for i in 1:N
        s += count_ones(a.parts[i])
    end
    return s
end
Base.count_zeros(a::MultiUInt{N, T}) where {N, T} = N * 8 * sizeof(T) - count_ones(a)

# Big-endian: compare high words first so MultiUInt values sort like large integers.
function Base.isless(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T}
    @inbounds for i in N:-1:1
        ai, bi = a.parts[i], b.parts[i]
        ai != bi && return ai < bi
    end
    return false
end

Base.:(==)(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T} = a.parts == b.parts

# Comparison with native integers (needed for `m == 0`, dict key lookups, etc.)
function Base.:(==)(a::MultiUInt{N, T}, b::Integer) where {N, T}
    # Low word must match, all higher words must be zero.
    a.parts[1] == T(b & typemax(T)) || return false
    @inbounds for i in 2:N
        a.parts[i] != zero(T) && return false
    end
    return true
end
# NOTE: no Base.:(==)(b::Integer, a::MultiUInt) method here —
# MultiUInt <: Unsigned <: Integer, so that would be ambiguous with
# ==(a::MultiUInt, b::MultiUInt). The above two-arg method already covers
# all mixed MultiUInt/Integer comparisons via symmetry from Julia's fallback.

Base.:<(a::MultiUInt, b::MultiUInt)  = isless(a, b)
Base.:>(a::MultiUInt, b::MultiUInt)  = isless(b, a)
Base.:<=(a::MultiUInt, b::MultiUInt) = !isless(b, a)
Base.:>=(a::MultiUInt, b::MultiUInt) = !isless(a, b)

# Word-by-word hash matching `Base.hash_integer` so that `MultiUInt(0)` and
# `0::Int` hash to the same value — `Dict{MultiUInt, ...}[0]` works correctly.
function Base.hash(a::MultiUInt{N, T}, h::UInt) where {N, T}
    @inbounds h ⊻= Base.hash_uint((a.parts[1] % UInt) ⊻ h)
    @inbounds for i in 2:N
        # Stop folding once all remaining words are zero (matches hash_integer semantics).
        rest_zero = true
        for j in i:N
            if a.parts[j] != zero(T)
                rest_zero = false
                break
            end
        end
        rest_zero && break
        h ⊻= Base.hash_uint((a.parts[i] % UInt) ⊻ h)
    end
    return h
end

# Promote against Integer literals so `m == 0`, `iszero(m)` via default fallback,
# and dict lookups with Int keys all just work.
Base.promote_rule(::Type{MultiUInt{N, T}}, ::Type{<:Integer}) where {N, T} =
    MultiUInt{N, T}

# ---- Arithmetic -------------------------------------------------------
# Ripple-carry add as a pure recursive tuple build — zero allocations, GPU-safe.
_add_with_carry(::Tuple{}, ::Tuple{}, ::T) where {T} = ()
function _add_with_carry(a::Tuple, b::Tuple, carry::T) where {T}
    s1 = a[1] + b[1]
    c1 = T(s1 < a[1])
    s2 = s1 + carry
    c2 = T(s2 < s1)
    return (s2, _add_with_carry(Base.tail(a), Base.tail(b), c1 + c2)...)
end

Base.:+(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T} =
    MultiUInt{N, T}(_add_with_carry(a.parts, b.parts, zero(T)))

Base.:-(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T} =
    a + (~b + one(MultiUInt{N, T}))

# Mixed-integer convenience for the common `+1` / `-1` idiom used in Bits.mask.
Base.:+(a::MultiUInt{N, T}, b::Integer) where {N, T} = a + MultiUInt{N, T}(b)
Base.:+(a::Integer, b::MultiUInt{N, T}) where {N, T} = MultiUInt{N, T}(a) + b
Base.:-(a::MultiUInt{N, T}, b::Integer) where {N, T} = a - MultiUInt{N, T}(b)

# ---- Conversion to native int (for array indexing) -------------------

function Base.convert(::Type{S}, a::MultiUInt{N, T}) where {S<:Signed, N, T}
    @inbounds for i in 2:N
        a.parts[i] == zero(T) || throw(InexactError(:convert, S, a))
    end
    v = a.parts[1]
    v <= typemax(S) || throw(InexactError(:convert, S, a))
    return S(v)
end
(::Type{S})(a::MultiUInt) where {S<:Signed} = convert(S, a)

# `lookup_map[lookup_int + 1]` pattern in CliffordGate apply.
Base.to_index(a::MultiUInt) = convert(Int, a)

# Truncate / widen to a native unsigned type — used by `T(target_pstr) << bitindex`.
function (::Type{U})(a::MultiUInt{N, T}) where {U<:Unsigned, N, T}
    if sizeof(U) <= sizeof(T)
        return U(a.parts[1] & U(typemax(U)))
    end
    word_bits = 8 * sizeof(T)
    nwords = cld(sizeof(U), sizeof(T))
    @assert nwords <= N "Cannot convert MultiUInt{$N,$T} to $U: needs $nwords words but only $N available"
    out = zero(U)
    @inbounds for i in 1:nwords
        out |= U(a.parts[i]) << ((i - 1) * word_bits)
    end
    return out
end

function Base.convert(::Type{MultiUInt{N, T}}, x::Integer) where {N, T}
    return MultiUInt{N, T}(x)
end

# ---- Bits.bitsize + Bits.mask ----------------------------------------
# Required by `_paulimask` in bitoperations.jl.

bitsize(::Type{MultiUInt{N, T}}) where {N, T} = N * 8 * sizeof(T)
bitsize(x::MultiUInt) = bitsize(typeof(x))

"""
    mask(::Type{MultiUInt{N,T}}, n)

Return a `MultiUInt{N,T}` with the low `n` bits set.
Required by `_paulimask` in bitoperations.jl — this keeps the existing
`_paulimask(T, n_sites) = mask(T, 2*n_sites)` generic call working for
`MultiUInt` types without any additional overloads in bitoperations.jl.
"""
function mask(::Type{MultiUInt{N, T}}, n::Integer) where {N, T}
    n = Int(n)
    n <= 0 && return zero(MultiUInt{N, T})
    total_bits = N * 8 * sizeof(T)
    n >= total_bits && return ~zero(MultiUInt{N, T})
    word_bits = 8 * sizeof(T)
    q, r = divrem(n, word_bits)
    parts = ntuple(Val(N)) do i
        if i <= q
            ~zero(T)
        elseif i == q + 1
            (one(T) << r) - one(T)
        else
            zero(T)
        end
    end
    return MultiUInt{N, T}(parts)
end

# ---- alternatingmask (compile-time constant) -------------------------
# Overload the @generated alternatingmask from bitoperations.jl so that
# the mask is a compile-time constant for MultiUInt — identical trick to
# NTuplePauliString's alternatingmask(@generated function).
#
# Pattern: even-indexed bits (0, 2, 4, …) are 1; odd-indexed are 0.
# Produces the mask ...1010101010101 used throughout the Pauli bit-ops.

@generated function alternatingmask(::Type{MultiUInt{N, T}}) where {N, T}
    wb = 8 * sizeof(T)
    # Build the repeating per-word pattern once.
    word_mask = zero(T)
    for bit in 0:(wb - 1)
        if bit % 2 == 0
            word_mask |= T(1) << bit
        end
    end
    words = ntuple(_ -> word_mask, N)
    return :(MultiUInt{$N, $T}($words))
end

alternatingmask(x::MultiUInt{N, T}) where {N, T} = alternatingmask(MultiUInt{N, T})

# ---- Display ----------------------------------------------------------

function Base.show(io::IO, a::MultiUInt{N, T}) where {N, T}
    print(io, "MultiUInt{", N, ",", T, "}(")
    @inbounds for i in N:-1:1
        i < N && print(io, "_")
        print(io, repr(a.parts[i]))
    end
    print(io, ")")
end