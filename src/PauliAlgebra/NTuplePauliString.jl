"""
    NTupleInteger{N,W<:Union{UInt32,UInt64}}

Encodes an unsigned integer as an `NTuple{N,W}` of small words ("limbs"),
following the same idea as `MultiFloats.jl`.

`N` is the number of machine words and `W` is the word type (`UInt32` or `UInt64`).
Together they provide `N * 8 * sizeof(W)` bits, enough for `N * 4 * sizeof(W)` qubits
when used as a Pauli string.

Limbs are stored little-endian: `_limbs[1]` holds the least-significant bits and
`_limbs[N]` holds the most-significant bits. Use `chunks(x)` to access them.

Encoding for Pauli strings: each qubit occupies 2 bits -- I=00, X=01, Y=10, Z=11 --
with qubit k in bit positions 2*(k-1) and 2*k-1, matching the rest of
PauliPropagation.jl.

Typical configurations:
  - 64 qubits  : NTupleInteger{2, UInt64}
  - 128 qubits : NTupleInteger{4, UInt64}
  - 256 qubits : NTupleInteger{8, UInt64}  or  NTupleInteger{16, UInt32}

Use `UInt32` words when targeting GPUs that run 32-bit register operations natively.

The type is `isbitstype` and all operations are word-level loops with no heap
allocation, so values live in registers and run inside GPU kernels.
"""
struct NTupleInteger{N,W<:Union{UInt32,UInt64}} <: Unsigned
    _limbs::NTuple{N,W}

    # Exact-type NTuple: called by all internal paths that already have the right type.
    NTupleInteger{N,W}(limbs::NTuple{N,W}) where {N, W<:Union{UInt32,UInt64}} = new{N,W}(limbs)

    # Mixed-element tuple (e.g. (UInt64, UInt64, 0x0, 0x0) where 0x0 is UInt8):
    # accept any Tuple of length N and convert each element to W.
    function NTupleInteger{N,W}(limbs::Tuple) where {N, W<:Union{UInt32,UInt64}}
        length(limbs) == N || throw(ArgumentError("expected $N limbs, got $(length(limbs))"))
        new{N,W}(ntuple(i -> W(limbs[i]), Val(N)))
    end

    # Construct from a machine integer, zero/sign-extending into the higher
    # limbs (unsigned() reinterprets the bits, matching two's-complement).
    function NTupleInteger{N,W}(x::Base.BitInteger) where {N, W<:Union{UInt32,UInt64}}
        wb = 8 * sizeof(W)
        # Widen to the larger of W and the source so shifting/masking is lossless.
        U  = promote_type(W, typeof(unsigned(x)))
        ux = U(unsigned(x))
        xbits = 8 * sizeof(unsigned(x))
        wmask = U(typemax(W))
        new{N,W}(ntuple(Val(N)) do i
            shift = (i - 1) * wb
            shift >= xbits ? zero(W) : W((ux >> shift) & wmask)
        end)
    end
end

"""
    chunks(x::NTupleInteger)

Return the tuple of limbs (least-significant first).
"""
chunks(x::NTupleInteger) = x._limbs

# Identity "conversion": constructing from a value that already has the exact
# type must be a no-op (e.g. oneunit(x) calls T(one(x))).
NTupleInteger{N,W}(x::NTupleInteger{N,W}) where {N, W<:Union{UInt32,UInt64}} = x

# Construct from another chunked type by repacking limbs directly, no BigInt.
function NTupleInteger{N2,W2}(x::NTupleInteger{N1,W1}) where {N1, W1, N2, W2}
    return _repack(NTupleInteger{N2,W2}, x)
end


NTupleInteger{N,W}() where {N,W} =
    NTupleInteger{N,W}(ntuple(_ -> zero(W), Val(N)))


Base.zero(::Type{NTupleInteger{N,W}}) where {N,W} =
    NTupleInteger{N,W}(ntuple(_ -> zero(W), Val(N)))

Base.zero(x::NTupleInteger{N,W}) where {N,W} = zero(typeof(x))

Base.one(::Type{NTupleInteger{N,W}}) where {N,W} =
    NTupleInteger{N,W}(ntuple(i -> i == 1 ? one(W) : zero(W), Val(N)))

Base.one(x::NTupleInteger{N,W}) where {N,W} = one(typeof(x))

# Unsigned, so the minimum value is zero.
Base.typemin(::Type{NTupleInteger{N,W}}) where {N,W} = zero(NTupleInteger{N,W})
Base.typemin(x::NTupleInteger{N,W}) where {N,W} = typemin(typeof(x))

Base.typemax(::Type{NTupleInteger{N,W}}) where {N,W} =
    NTupleInteger{N,W}(ntuple(_ -> typemax(W), Val(N)))
Base.typemax(x::NTupleInteger{N,W}) where {N,W} = typemax(typeof(x))

Base.iszero(x::NTupleInteger) = x == zero(x)

_wordbits(::Type{NTupleInteger{N,W}}) where {N,W} = 8 * sizeof(W)
_wordbits(x::NTupleInteger) = _wordbits(typeof(x))

# bitsize: required by alternatingmask() in bitoperations.jl, and by maxqubits()
# in utils.jl. The definition for plain Integer types comes from Bits.jl.
Bits.bitsize(::Type{NTupleInteger{N,W}}) where {N,W} = N * 8 * sizeof(W)
Bits.bitsize(x::NTupleInteger) = Bits.bitsize(typeof(x))


"""
    getchunkedinttype(nqubits; word=UInt64)

Return the smallest `NTupleInteger{N,W}` type that can represent `nqubits` qubits.
This type is `isbitstype` and therefore compatible with GPU kernels, unlike the
types returned by `getinttype` for large qubit counts. Defaults to `UInt64` words.
"""
function getchunkedinttype(nqubits::Integer; word::Type{W}=UInt64) where {W<:Union{UInt32,UInt64}}
    bits_needed = 2 * nqubits
    bits_per_word = 8 * sizeof(W)
    N = cld(bits_needed, bits_per_word)
    return NTupleInteger{N,W}
end


@inline function Base.:~(a::NTupleInteger{N,W}) where {N, W<:Union{UInt32,UInt64}}
    NTupleInteger{N,W}(map(~, a._limbs))
end

@inline function Base.:&(a::NTupleInteger{N,W}, b::NTupleInteger{N,W}) where {N, W<:Union{UInt32,UInt64}}
    NTupleInteger{N,W}(ntuple(i -> a._limbs[i] & b._limbs[i], Val(N)))
end

@inline function Base.:|(a::NTupleInteger{N,W}, b::NTupleInteger{N,W}) where {N, W<:Union{UInt32,UInt64}}
    NTupleInteger{N,W}(ntuple(i -> a._limbs[i] | b._limbs[i], Val(N)))
end

@inline function Base.:⊻(a::NTupleInteger{N,W}, b::NTupleInteger{N,W}) where {N, W<:Union{UInt32,UInt64}}
    NTupleInteger{N,W}(ntuple(i -> a._limbs[i] ⊻ b._limbs[i], Val(N)))
end


# Shift by Int (the concrete type Julia uses for integer literals). Adding the
# W<:Union{UInt32,UInt64} bound makes this strictly more specific than
# Base.>>(::Integer, ::Int) so Julia picks it unambiguously.
@inline function Base.:>>(a::NTupleInteger{N,W}, k::Int) where {N, W<:Union{UInt32,UInt64}}
    k == 0 && return a
    wb = _wordbits(a)
    k >= N * wb && return zero(a)

    word_shift = k ÷ wb
    bit_shift  = k % wb

    new_limbs = ntuple(Val(N)) do i
        src = i + word_shift
        if src > N
            zero(W)
        elseif bit_shift == 0
            a._limbs[src]
        else
            lo = a._limbs[src] >> bit_shift
            hi = src + 1 <= N ? a._limbs[src + 1] << (wb - bit_shift) : zero(W)
            lo | hi
        end
    end
    NTupleInteger{N,W}(new_limbs)
end

@inline function Base.:(<<)(a::NTupleInteger{N,W}, k::Int) where {N, W<:Union{UInt32,UInt64}}
    k == 0 && return a
    wb = _wordbits(a)
    k >= N * wb && return zero(a)

    word_shift = k ÷ wb
    bit_shift  = k % wb

    new_limbs = ntuple(Val(N)) do i
        src = i - word_shift
        if src < 1
            zero(W)
        elseif bit_shift == 0
            a._limbs[src]
        else
            hi = a._limbs[src] << bit_shift
            lo = src - 1 >= 1 ? a._limbs[src - 1] >> (wb - bit_shift) : zero(W)
            hi | lo
        end
    end
    NTupleInteger{N,W}(new_limbs)
end

# Convert any other Integer shift amount to Int to hit the above methods.
@inline Base.:>>(a::NTupleInteger, k::Integer) = a >> Int(k)
@inline Base.:(<<)(a::NTupleInteger, k::Integer) = a << Int(k)


@inline function Base.:(==)(a::NTupleInteger{N,W}, b::NTupleInteger{N,W}) where {N, W<:Union{UInt32,UInt64}}
    eq = true
    for i in 1:N
        eq &= a._limbs[i] == b._limbs[i]
    end
    return eq
end

@inline function Base.:(==)(a::NTupleInteger{N,W}, b::Integer) where {N, W<:Union{UInt32,UInt64}}
    return a == NTupleInteger{N,W}(b)
end

@inline Base.:(==)(b::Integer, a::NTupleInteger) = a == b

@inline function Base.isless(a::NTupleInteger{N,W}, b::NTupleInteger{N,W}) where {N, W<:Union{UInt32,UInt64}}
    lt = false
    gt = false
    # Compare from most-significant limb down. Branchless to stay GPU-friendly:
    # the first differing limb (scanning high to low) decides the result.
    for i in N:-1:1
        ai = a._limbs[i]
        bi = b._limbs[i]
        decided = lt | gt
        lt |= (!decided) & (ai < bi)
        gt |= (!decided) & (ai > bi)
    end
    return lt
end

Base.:<(a::NTupleInteger, b::NTupleInteger)  = isless(a, b)
Base.:>(a::NTupleInteger, b::NTupleInteger)  = isless(b, a)
Base.:<=(a::NTupleInteger, b::NTupleInteger) = !isless(b, a)
Base.:>=(a::NTupleInteger, b::NTupleInteger) = !isless(a, b)


# Ripple-carry add/sub via recursive carry threading. No heap allocation and no
# mutable state, so the whole thing compiles and runs inside a GPU kernel.
@inline function _addlimbs(a::NTuple{N,W}, b::NTuple{N,W}, i::Int, carry::W) where {N,W}
    i > N && return ()
    ai = a[i]
    s1 = ai + b[i]
    c1 = s1 < ai
    s2 = s1 + carry
    c2 = s2 < s1
    nextcarry = (c1 | c2) ? one(W) : zero(W)
    return (s2, _addlimbs(a, b, i + 1, nextcarry)...)
end

@inline function Base.:+(a::NTupleInteger{N,W}, b::NTupleInteger{N,W}) where {N, W<:Union{UInt32,UInt64}}
    NTupleInteger{N,W}(_addlimbs(a._limbs, b._limbs, 1, zero(W)))
end

@inline function _sublimbs(a::NTuple{N,W}, b::NTuple{N,W}, i::Int, borrow::W) where {N,W}
    i > N && return ()
    ai = a[i]
    bi = b[i] + borrow
    nextborrow = (bi < b[i] || ai < bi) ? one(W) : zero(W)
    return (ai - bi, _sublimbs(a, b, i + 1, nextborrow)...)
end

@inline function Base.:-(a::NTupleInteger{N,W}, b::NTupleInteger{N,W}) where {N, W<:Union{UInt32,UInt64}}
    NTupleInteger{N,W}(_sublimbs(a._limbs, b._limbs, 1, zero(W)))
end

# Schoolbook multiplication, wrapping (mod 2^(N*wb)) to match machine-integer
# semantics. Builds the result limb by limb with a widened accumulator; the
# accumulator is threaded functionally via Base.setindex on the tuple (no
# allocation), so it stays GPU-compatible.
@inline function Base.:*(a::NTupleInteger{N,W}, b::NTupleInteger{N,W}) where {N, W<:Union{UInt32,UInt64}}
    WW = _doubleword(W)
    wb = 8 * sizeof(W)
    acc = ntuple(_ -> zero(W), Val(N))
    for i in 1:N
        carry = zero(WW)
        ai = WW(a._limbs[i])
        for j in 1:(N - i + 1)
            k = i + j - 1
            prod = ai * WW(b._limbs[j]) + WW(acc[k]) + carry
            acc = Base.setindex(acc, W(prod & WW(typemax(W))), k)
            carry = prod >> wb
        end
    end
    NTupleInteger{N,W}(acc)
end

# Long division (returns quotient), wrapping semantics. Restoring bit-by-bit
# algorithm; only needed to satisfy the Integer interface, not perf critical.
@inline function Base.div(a::NTupleInteger{N,W}, b::NTupleInteger{N,W}) where {N, W<:Union{UInt32,UInt64}}
    iszero(b) && throw(DivideError())
    q = zero(a)
    r = zero(a)
    nbits = N * 8 * sizeof(W)
    for i in (nbits - 1):-1:0
        r = r << 1
        bit = (a >> i) & one(a)
        r = r | bit
        if r >= b
            r = r - b
            q = q | (one(a) << i)
        end
    end
    return q
end

@inline function Base.rem(a::NTupleInteger{N,W}, b::NTupleInteger{N,W}) where {N, W<:Union{UInt32,UInt64}}
    return a - div(a, b) * b
end

# Mixed-integer arithmetic promotes the machine integer up first.
@inline Base.:+(a::NTupleInteger{N,W}, b::Integer) where {N,W} = a + NTupleInteger{N,W}(b)
@inline Base.:-(a::NTupleInteger{N,W}, b::Integer) where {N,W} = a - NTupleInteger{N,W}(b)
@inline Base.:+(b::Integer, a::NTupleInteger{N,W}) where {N,W} = NTupleInteger{N,W}(b) + a
@inline Base.:*(a::NTupleInteger{N,W}, b::Integer) where {N,W} = a * NTupleInteger{N,W}(b)
@inline Base.:*(b::Integer, a::NTupleInteger{N,W}) where {N,W} = NTupleInteger{N,W}(b) * a

# Overflow-aware arithmetic, required by Base's sortperm/sort machinery which
# probes Integer types with sub_with_overflow to decide whether counting sort
# applies (BitIntegers.jl defines these for its types for the same reason).
# For unsigned arithmetic: subtraction overflows iff a < b (a borrow out),
# and addition overflows iff the wrapped sum is smaller than an operand.
@inline function Base.Checked.sub_with_overflow(a::NTupleInteger{N,W}, b::NTupleInteger{N,W}) where {N,W}
    return (a - b, a < b)
end

@inline function Base.Checked.add_with_overflow(a::NTupleInteger{N,W}, b::NTupleInteger{N,W}) where {N,W}
    s = a + b
    return (s, s < a)
end


@inline function Base.count_ones(a::NTupleInteger{N,W}) where {N,W}
    s = 0
    for i in 1:N
        s += count_ones(a._limbs[i])
    end
    return s
end


# Conversions to machine integers assemble the low limbs (no BigInt).
@inline function Base.UInt64(a::NTupleInteger{N,W}) where {N,W}
    wb = 8 * sizeof(W)
    out = zero(UInt64)
    nlimbs = min(N, 64 ÷ wb)
    for i in 1:nlimbs
        out |= UInt64(a._limbs[i]) << ((i - 1) * wb)
    end
    return out
end

@inline Base.UInt32(a::NTupleInteger{N,W}) where {N,W} = UInt32(a._limbs[1])
@inline Base.Int(a::NTupleInteger) = Int(UInt64(a))

# Full-width conversion to BigInt for arbitrary-precision interop and printing.
# Not used in any hot path or GPU kernel.
function Base.BigInt(a::NTupleInteger{N,W}) where {N,W}
    val = BigInt(0)
    wb  = 8 * sizeof(W)
    for i in N:-1:1
        val = (val << wb) | BigInt(a._limbs[i])
    end
    return val
end

# Promotion: mixed arithmetic/comparisons with machine integers promote to
# the NTupleInteger type (matching how UInt64 + UInt8 promotes to UInt64).
Base.promote_rule(::Type{NTupleInteger{N,W}}, ::Type{T}) where {N, W, T<:Base.BitInteger} = NTupleInteger{N,W}
Base.promote_rule(::Type{NTupleInteger{N,W}}, ::Type{Bool}) where {N, W} = NTupleInteger{N,W}

Base.convert(::Type{NTupleInteger{N,W}}, x::Integer) where {N,W} = NTupleInteger{N,W}(x)
Base.convert(::Type{NTupleInteger{N,W}}, x::NTupleInteger{N,W}) where {N,W} = x
Base.convert(::Type{NTupleInteger{N2,W2}}, x::NTupleInteger{N1,W1}) where {N1,W1,N2,W2} =
    NTupleInteger{N2,W2}(x)


# Repack one chunked type into another by walking source bits limb by limb,
# no BigInt. Handles W1 != W2 and N1 != N2 (truncating or zero-extending).
function _repack(::Type{NTupleInteger{N2,W2}}, x::NTupleInteger{N1,W1}) where {N1,W1,N2,W2}
    wb1 = 8 * sizeof(W1)
    wb2 = 8 * sizeof(W2)
    if wb1 == wb2
        # same word width: copy/truncate/zero-extend limbs directly
        return NTupleInteger{N2,W2}(ntuple(i -> i <= N1 ? W2(x._limbs[i]) : zero(W2), Val(N2)))
    elseif wb2 > wb1
        # widening: pack (wb2/wb1) source limbs into each destination limb
        ratio = wb2 ÷ wb1
        return NTupleInteger{N2,W2}(ntuple(Val(N2)) do j
            acc = zero(W2)
            for r in 1:ratio
                si = (j - 1) * ratio + r
                if si <= N1
                    acc |= W2(x._limbs[si]) << ((r - 1) * wb1)
                end
            end
            acc
        end)
    else
        # narrowing: split each source limb into (wb1/wb2) destination limbs
        ratio = wb1 ÷ wb2
        return NTupleInteger{N2,W2}(ntuple(Val(N2)) do j
            si = (j - 1) ÷ ratio + 1
            r  = (j - 1) % ratio
            si <= N1 ? W2((x._limbs[si] >> (r * wb2)) & W1(typemax(W2))) : zero(W2)
        end)
    end
end

# Widened word type for multiplication partial products.
_doubleword(::Type{UInt32}) = UInt64
_doubleword(::Type{UInt64}) = UInt128


# hash the limbs directly (no BigInt). Combine each limb into the running hash.
function Base.hash(a::NTupleInteger{N,W}, h::UInt) where {N,W}
    for i in 1:N
        h = hash(a._limbs[i], h)
    end
    return h
end


function Base.show(io::IO, a::NTupleInteger{N,W}) where {N,W}
    wb  = 8 * sizeof(W)
    hex = ""
    for i in N:-1:1
        hex *= string(a._limbs[i]; base=16, pad=wb÷4)
    end
    print(io, "NTupleInteger{$N,$W}(0x", hex, ")")
end