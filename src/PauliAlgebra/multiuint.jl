### multiuint.jl
##
# A GPU-friendly long unsigned integer built from a fixed-size NTuple of
# native unsigned words. Plug-in replacement for the dynamically-defined
# BitIntegers.jl types in the >64-bit code path of `getinttype`.
#
# Motivation (issue #145):
#   - The CUDA extension breaks above 32 qubits, because Pauli strings
#     need >64 bits to encode, and BitIntegers' wide types blow up GPU
#     register pressure / are not always `isbits` in a way the GPU
#     compiler tolerates.
#   - `MultiUInt{N,T}` is `isbits`, fits in N registers of width T, and
#     supports exactly the bitwise contract that PauliPropagation uses.
#
# Design:
#   - Pure value type, stored as an NTuple. Pointwise ops `&, |, ⊻, ~`
#     are written as `ntuple(...)` so the Julia compiler unrolls them.
#   - Shifts handle inter-word carry. `<<` and `>>` work for arbitrary
#     non-negative shift amounts; large shifts (>= total bits) return 0.
#   - `count_ones` sums across words.
#   - `isless` compares high-word-first, matching big-endian semantics
#     when read as a single large integer.
#   - `hash` folds across words so `MultiUInt{N,T}(0)` always hashes
#     the same regardless of which subset of words is zero.
#
# Layout note: `_limbs[1]` is the LOW word, `_limbs[end]` is the HIGH
# word. This matches `<< s` shifting `_limbs[1]` into `_limbs[2]`.
###

using Bits: bitsize, mask

"""
    MultiUInt{N, T} <: Unsigned

GPU-friendly long unsigned integer built from `N` words of type `T<:Unsigned`.
The total bit width is `N * 8 * sizeof(T)` (e.g. `MultiUInt{4, UInt64}` is
a 256-bit unsigned integer occupying four 64-bit lanes).

Supports the bitwise contract used by PauliPropagation: `&`, `|`, `⊻`,
`~`, `<<`, `>>`, `count_ones`, equality, ordering, `zero`, `one`, `hash`,
and `Bits.bitsize`. Arithmetic (`+`, `*`, etc.) is intentionally NOT
defined — Pauli encoding only uses bitwise operations, and adding general
arithmetic would invite accidental misuse on this representation.

`_limbs[1]` is the least-significant word; `_limbs[end]` is the most
significant. Constructed from any small integer by zero-extension.

# Examples

```jldoctest
julia> using PauliPropagation: MultiUInt

julia> a = MultiUInt{2, UInt64}(0xABCD)
MultiUInt{2,UInt64}(0xabcd)

julia> a << 64 |> bitstring |> length
128
```
"""
struct MultiUInt{N, T<:Unsigned} <: Unsigned
    _limbs::NTuple{N, T}
end

# ---- Construction -----------------------------------------------------

# From a small integer literal, zero-extended into the low word.
function MultiUInt{N, T}(x::Integer) where {N, T<:Unsigned}
    x >= 0 || throw(DomainError(x, "MultiUInt is unsigned, got negative $x"))
    word_bits = 8 * sizeof(T)
    mask_low = (T(1) << (word_bits - 1) - T(1)) << 1 | T(1)   # = typemax(T), branchless of <<word_bits
    limbs = ntuple(N) do i
        shift = (i - 1) * word_bits
        T((x >> shift) & mask_low)
    end
    return MultiUInt{N, T}(limbs)
end

# Identity / convert from another MultiUInt of the same shape. Bound the
# `where` clause to `T<:Unsigned` so this is no more or less specific than
# the Integer-input constructor above — otherwise Julia treats the two as
# ambiguous when the input is itself a MultiUInt (since MultiUInt<:Integer).
MultiUInt{N, T}(x::MultiUInt{N, T}) where {N, T<:Unsigned} = x
# NB: an NTuple{N,T} ctor is auto-generated from the struct; no override.

# ---- Basic predicates -------------------------------------------------

Base.zero(::Type{MultiUInt{N, T}}) where {N, T} =
    MultiUInt{N, T}(ntuple(_ -> zero(T), Val(N)))
Base.zero(x::MultiUInt) = zero(typeof(x))

Base.one(::Type{MultiUInt{N, T}}) where {N, T} =
    MultiUInt{N, T}(ntuple(i -> i == 1 ? one(T) : zero(T), Val(N)))
Base.one(x::MultiUInt) = one(typeof(x))

Base.iszero(a::MultiUInt) = all(iszero, a._limbs)

# ---- Bitwise: pointwise -----------------------------------------------

Base.:&(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T} =
    MultiUInt{N, T}(ntuple(i -> a._limbs[i] & b._limbs[i], Val(N)))

Base.:|(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T} =
    MultiUInt{N, T}(ntuple(i -> a._limbs[i] | b._limbs[i], Val(N)))

Base.xor(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T} =
    MultiUInt{N, T}(ntuple(i -> a._limbs[i] ⊻ b._limbs[i], Val(N)))

Base.:~(a::MultiUInt{N, T}) where {N, T} =
    MultiUInt{N, T}(ntuple(i -> ~a._limbs[i], Val(N)))

# ---- Shifts -----------------------------------------------------------
# Left shift: low->high word migration. Carry from word i is the top
# `r` bits of `_limbs[i]`, which become the bottom of `_limbs[i+1]`.

function _shift_left(a::MultiUInt{N, T}, s::Int) where {N, T}
    s < 0 && return _shift_right(a, -s)
    word_bits = 8 * sizeof(T)
    total_bits = N * word_bits
    s >= total_bits && return zero(MultiUInt{N, T})
    q, r = divrem(s, word_bits)
    new_limbs = ntuple(Val(N)) do i
        lo_src = i - q                           # word that supplies this lane's low part
        hi_src = lo_src - 1                      # word that supplies the carry into this lane
        lo = (1 <= lo_src <= N) ? a._limbs[lo_src] : zero(T)
        # r == 0 must not produce a >> word_bits which is UB on T.
        carry = if r == 0 || hi_src < 1 || hi_src > N
            zero(T)
        else
            a._limbs[hi_src] >> (word_bits - r)
        end
        (lo << r) | carry
    end
    return MultiUInt{N, T}(new_limbs)
end

# Right shift: high->low word migration.

function _shift_right(a::MultiUInt{N, T}, s::Int) where {N, T}
    s < 0 && return _shift_left(a, -s)
    word_bits = 8 * sizeof(T)
    total_bits = N * word_bits
    s >= total_bits && return zero(MultiUInt{N, T})
    q, r = divrem(s, word_bits)
    new_limbs = ntuple(Val(N)) do i
        lo_src = i + q
        hi_src = lo_src + 1
        lo = (1 <= lo_src <= N) ? a._limbs[lo_src] : zero(T)
        carry = if r == 0 || hi_src < 1 || hi_src > N
            zero(T)
        else
            a._limbs[hi_src] << (word_bits - r)
        end
        (lo >> r) | carry
    end
    return MultiUInt{N, T}(new_limbs)
end

# Public shift entry points. We need methods explicitly typed on both Int
# and Integer to disambiguate from Base.<<(::Integer, ::Int) — Julia sees
# competing specialisations on the first vs second argument.
Base.:<<(a::MultiUInt, s::Int) = _shift_left(a, s)
Base.:<<(a::MultiUInt, s::Integer) = _shift_left(a, Int(s))
Base.:>>(a::MultiUInt, s::Int) = _shift_right(a, s)
Base.:>>(a::MultiUInt, s::Integer) = _shift_right(a, Int(s))
# Unsigned-right-shift is logical for our type — same as >>.
Base.:>>>(a::MultiUInt, s::Int) = _shift_right(a, s)
Base.:>>>(a::MultiUInt, s::Integer) = _shift_right(a, Int(s))

# ---- Counting / ordering / hash ---------------------------------------

# Explicit loop instead of `sum(f, tuple; init=...)` — guarantees no
# allocation and inlines cleanly inside GPU kernels.
function Base.count_ones(a::MultiUInt{N, T}) where {N, T}
    s = 0
    @inbounds for i in 1:N
        s += count_ones(a._limbs[i])
    end
    return s
end
Base.count_zeros(a::MultiUInt{N, T}) where {N, T} =
    N * 8 * sizeof(T) - count_ones(a)

# Big-endian compare: compare high words first.
function Base.isless(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T}
    @inbounds for i in N:-1:1
        ai, bi = a._limbs[i], b._limbs[i]
        ai != bi && return ai < bi
    end
    return false
end

Base.:(==)(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T} = a._limbs == b._limbs

# Equal-valued integers must hash identically (incl. `MultiUInt(0)` vs `0::Int`
# vs `UInt64(0)`). Reproduce `Base.hash_integer`'s behaviour directly: hash the
# low word, then each successive word UNTIL all remaining words are zero.
function Base.hash(a::MultiUInt{N, T}, h::UInt) where {N, T}
    @inbounds h ⊻= Base.hash_uint((a._limbs[1] % UInt) ⊻ h)
    @inbounds for i in 2:N
        rest_zero = true
        for j in i:N
            if a._limbs[j] != zero(T)
                rest_zero = false
                break
            end
        end
        rest_zero && break
        h ⊻= Base.hash_uint((a._limbs[i] % UInt) ⊻ h)
    end
    return h
end

# Promote against small integer literals so that `m == 0`, `iszero(m)` via
# the default fallback, and dict lookups with Int keys all work.
Base.promote_rule(::Type{MultiUInt{N, T}}, ::Type{<:Integer}) where {N, T} =
    MultiUInt{N, T}

# ---- Arithmetic ------------------------------------------------------
# Limited to the operations PauliPropagation actually uses (`+` / `-`).
# Implemented with ripple-carry across words; not vectorised, but cheap
# enough since N is small (at most ~8 for 256 qubits).

# Ripple-carry add as a pure recursive tuple build — zero allocations, fully
# unrolled by the compiler. GPU-safe.
_add_with_carry(::Tuple{}, ::Tuple{}, ::T) where {T} = ()
function _add_with_carry(a::Tuple, b::Tuple, carry::T) where {T}
    s1 = a[1] + b[1]
    c1 = T(s1 < a[1])
    s2 = s1 + carry
    c2 = T(s2 < s1)
    new_carry = c1 + c2
    return (s2, _add_with_carry(Base.tail(a), Base.tail(b), new_carry)...)
end

function Base.:+(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T}
    return MultiUInt{N, T}(_add_with_carry(a._limbs, b._limbs, zero(T)))
end

function Base.:-(a::MultiUInt{N, T}, b::MultiUInt{N, T}) where {N, T}
    # two's complement: a - b == a + (~b + 1)
    return a + (~b + one(MultiUInt{N, T}))
end

# Mixed-int convenience for the common `+1` / `-1` idiom.
Base.:+(a::MultiUInt{N, T}, b::Integer) where {N, T} = a + MultiUInt{N, T}(b)
Base.:+(a::Integer, b::MultiUInt{N, T}) where {N, T} = MultiUInt{N, T}(a) + b
Base.:-(a::MultiUInt{N, T}, b::Integer) where {N, T} = a - MultiUInt{N, T}(b)

# ---- Conversion to native int (for array indexing) -------------------
# Vector{T}[muint] does `convert(Int, muint)`. Only succeeds for values
# that fit in Int (i.e. all high words zero and low word ≤ typemax(Int)).

function Base.convert(::Type{S}, a::MultiUInt{N, T}) where {S<:Signed, N, T}
    @inbounds for i in 2:N
        a._limbs[i] == zero(T) || throw(InexactError(:convert, S, a))
    end
    v = a._limbs[1]
    v <= typemax(S) || throw(InexactError(:convert, S, a))
    return S(v)
end

(::Type{S})(a::MultiUInt) where {S<:Signed} = convert(S, a)

# Base.to_index for the indexing path used in `lookup_map[lookup_int + 1]`.
Base.to_index(a::MultiUInt) = convert(Int, a)

# ---- Conversions ------------------------------------------------------

# To/from native unsigned types — only safe when the value fits.
function Base.convert(::Type{MultiUInt{N, T}}, x::Integer) where {N, T}
    return MultiUInt{N, T}(x)
end

# Truncate to a native type: returns _limbs[1] for sizeof(U) <= sizeof(T),
# else widens by reading more words. Used by PauliPropagation when it
# does T(small_value) cycles inside _setpaulibits.
function (::Type{U})(a::MultiUInt{N, T}) where {U<:Unsigned, N, T}
    if sizeof(U) <= sizeof(T)
        return U(a._limbs[1] & U(typemax(U)))
    end
    # Widen: pack low words into U.
    word_bits = 8 * sizeof(T)
    nwords = cld(sizeof(U), sizeof(T))
    @assert nwords <= N "Cannot convert MultiUInt{$N,$T} to $U: needs $nwords words"
    out = zero(U)
    @inbounds for i in 1:nwords
        out |= U(a._limbs[i]) << ((i - 1) * word_bits)
    end
    return out
end

# Allow Base.UInt(::MultiUInt) etc. for ergonomics.
Base.UInt(a::MultiUInt) = UInt(a)::UInt

# ---- Bits.bitsize -----------------------------------------------------
# PauliPropagation calls `bitsize(x)` to size masks for `alternatingmask`.
# Define for the type rather than reusing the BitIntegers fallback.

Bits.bitsize(::Type{MultiUInt{N, T}}) where {N, T} = N * 8 * sizeof(T)
Bits.bitsize(x::MultiUInt) = Bits.bitsize(typeof(x))

# Bits.mask used in `_paulimask`: returns a MultiUInt with the low `n`
# bits set.
function Bits.mask(::Type{MultiUInt{N, T}}, n::Integer) where {N, T}
    n = Int(n)
    n <= 0 && return zero(MultiUInt{N, T})
    total_bits = N * 8 * sizeof(T)
    n >= total_bits && return ~zero(MultiUInt{N, T})
    word_bits = 8 * sizeof(T)
    q, r = divrem(n, word_bits)
    limbs = ntuple(Val(N)) do i
        if i <= q
            ~zero(T)
        elseif i == q + 1
            (one(T) << r) - one(T)
        else
            zero(T)
        end
    end
    return MultiUInt{N, T}(limbs)
end

# ---- Display ----------------------------------------------------------

function Base.show(io::IO, a::MultiUInt{N, T}) where {N, T}
    print(io, "MultiUInt{", N, ",", T, "}(")
    @inbounds for i in N:-1:1
        i < N && print(io, "_")
        print(io, repr(a._limbs[i]))
    end
    print(io, ")")
end
