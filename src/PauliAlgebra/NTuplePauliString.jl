"""
    NTupleUInt{N,W<:Union{UInt32,UInt64}}

Encodes an unsigned integer as an `NTuple{N,W}` of small words.

`N` is the number of machine words and `W` is the word type (`UInt32` or `UInt64`).
Together they provide `N * 8 * sizeof(W)` bits, enough for `N * 4 * sizeof(W)` qubits
when used as a Pauli string.

Words are stored in little-endian order: `data[1]` holds the least-significant bits
and `data[N]` holds the most-significant bits.

Encoding for Pauli strings: each qubit occupies 2 bits -- I=00, X=01, Y=10, Z=11 --
with qubit k in bit positions 2*(k-1) and 2*k-1, matching the rest of
PauliPropagation.jl.

Typical configurations:
  - 64 qubits  : NTupleUInt{2, UInt64}
  - 128 qubits : NTupleUInt{4, UInt64}
  - 256 qubits : NTupleUInt{8, UInt64}  or  NTupleUInt{16, UInt32}

Use `UInt32` words when targeting GPUs that run 32-bit register operations natively.
"""
struct NTupleUInt{N,W<:Union{UInt32,UInt64}} <: Unsigned
    data::NTuple{N,W}

    # Exact-type NTuple: called by all internal paths that already have the right type.
    NTupleUInt{N,W}(data::NTuple{N,W}) where {N, W<:Union{UInt32,UInt64}} = new{N,W}(data)

    # Mixed-element tuple (e.g. (UInt64, UInt64, 0x0, 0x0) where 0x0 is UInt8):
    # accept any Tuple of length N and convert each element to W.
    function NTupleUInt{N,W}(data::Tuple) where {N, W<:Union{UInt32,UInt64}}
        length(data) == N || throw(ArgumentError("expected $N elements, got $(length(data))"))
        new{N,W}(ntuple(i -> W(data[i]), Val(N)))
    end

    # Construct from any Integer, zero-extending into the higher words.
    function NTupleUInt{N,W}(x::Integer) where {N, W<:Union{UInt32,UInt64}}
        bx    = BigInt(x)
        bmask = BigInt(typemax(W))
        wb    = 8 * sizeof(W)
        new{N,W}(ntuple(i -> W((bx >> ((i - 1) * wb)) & bmask), Val(N)))
    end
end

# .parts is an alias for .data, used by the tests.
Base.getproperty(x::NTupleUInt, f::Symbol) =
    f === :parts ? getfield(x, :data) : getfield(x, f)


NTupleUInt{N,W}() where {N,W} =
    NTupleUInt{N,W}(ntuple(_ -> zero(W), Val(N)))


Base.zero(::Type{NTupleUInt{N,W}}) where {N,W} =
    NTupleUInt{N,W}(ntuple(_ -> zero(W), Val(N)))

Base.zero(x::NTupleUInt{N,W}) where {N,W} = zero(typeof(x))

Base.one(::Type{NTupleUInt{N,W}}) where {N,W} =
    NTupleUInt{N,W}(ntuple(i -> i == 1 ? one(W) : zero(W), Val(N)))

Base.one(x::NTupleUInt{N,W}) where {N,W} = one(typeof(x))

Base.typemax(::Type{NTupleUInt{N,W}}) where {N,W} =
    NTupleUInt{N,W}(ntuple(_ -> typemax(W), Val(N)))

Base.iszero(x::NTupleUInt) = x == zero(x)

_wordbits(::Type{NTupleUInt{N,W}}) where {N,W} = 8 * sizeof(W)
_wordbits(x::NTupleUInt) = _wordbits(typeof(x))

# bitsize: required by alternatingmask() in bitoperations.jl, and by maxqubits() in utils.jl.
# We define it for NTupleUInt here; the definition for plain Integer types comes from Bits.jl.
Bits.bitsize(::Type{NTupleUInt{N,W}}) where {N,W} = N * 8 * sizeof(W)
Bits.bitsize(x::NTupleUInt) = Bits.bitsize(typeof(x))


"""
    getchunkedinttype(nqubits; word=UInt64)

Return the smallest `NTupleUInt{N,W}` type that can represent `nqubits` qubits.
This type is `isbitstype` and therefore compatible with GPU kernels, unlike the
types returned by `getinttype` for large qubit counts.
Defaults to `UInt64` words.
"""
function getchunkedinttype(nqubits::Integer; word::Type{W}=UInt64) where {W<:Union{UInt32,UInt64}}
    bits_needed = 2 * nqubits
    bits_per_word = 8 * sizeof(W)
    N = cld(bits_needed, bits_per_word)
    return NTupleUInt{N,W}
end


@inline function Base.:~(a::NTupleUInt{N,W}) where {N, W<:Union{UInt32,UInt64}}
    NTupleUInt{N,W}(map(~, a.data))
end

@inline function Base.:&(a::NTupleUInt{N,W}, b::NTupleUInt{N,W}) where {N, W<:Union{UInt32,UInt64}}
    NTupleUInt{N,W}(ntuple(i -> a.data[i] & b.data[i], Val(N)))
end

@inline function Base.:|(a::NTupleUInt{N,W}, b::NTupleUInt{N,W}) where {N, W<:Union{UInt32,UInt64}}
    NTupleUInt{N,W}(ntuple(i -> a.data[i] | b.data[i], Val(N)))
end

@inline function Base.:⊻(a::NTupleUInt{N,W}, b::NTupleUInt{N,W}) where {N, W<:Union{UInt32,UInt64}}
    NTupleUInt{N,W}(ntuple(i -> a.data[i] ⊻ b.data[i], Val(N)))
end


# Shift by Int (the concrete type Julia uses for integer literals).
# Adding W<:Union{UInt32,UInt64} makes this strictly more specific than
# Base.>>(::Integer, ::Int) so Julia can pick it unambiguously.
@inline function Base.:>>(a::NTupleUInt{N,W}, k::Int) where {N, W<:Union{UInt32,UInt64}}
    k == 0 && return a
    wb = _wordbits(a)
    k >= N * wb && return zero(a)

    word_shift = k ÷ wb
    bit_shift  = k % wb

    new_data = ntuple(Val(N)) do i
        src = i + word_shift
        if src > N
            zero(W)
        elseif bit_shift == 0
            a.data[src]
        else
            lo = a.data[src] >> bit_shift
            hi = src + 1 <= N ? a.data[src + 1] << (wb - bit_shift) : zero(W)
            lo | hi
        end
    end
    NTupleUInt{N,W}(new_data)
end

@inline function Base.:(<<)(a::NTupleUInt{N,W}, k::Int) where {N, W<:Union{UInt32,UInt64}}
    k == 0 && return a
    wb = _wordbits(a)
    k >= N * wb && return zero(a)

    word_shift = k ÷ wb
    bit_shift  = k % wb

    new_data = ntuple(Val(N)) do i
        src = i - word_shift
        if src < 1
            zero(W)
        elseif bit_shift == 0
            a.data[src]
        else
            hi = a.data[src] << bit_shift
            lo = src - 1 >= 1 ? a.data[src - 1] >> (wb - bit_shift) : zero(W)
            hi | lo
        end
    end
    NTupleUInt{N,W}(new_data)
end

# Convert any other Integer shift amount to Int to hit the above methods.
@inline Base.:>>(a::NTupleUInt, k::Integer) = a >> Int(k)
@inline Base.:(<<)(a::NTupleUInt, k::Integer) = a << Int(k)


@inline function Base.:(==)(a::NTupleUInt{N,W}, b::NTupleUInt{N,W}) where {N, W<:Union{UInt32,UInt64}}
    for i in 1:N
        a.data[i] != b.data[i] && return false
    end
    return true
end

@inline function Base.:(==)(a::NTupleUInt{N,W}, b::Integer) where {N, W<:Union{UInt32,UInt64}}
    a.data[1] == W(b & typemax(W)) || return false
    for i in 2:N
        a.data[i] != zero(W) && return false
    end
    return true
end

@inline Base.:(==)(b::Integer, a::NTupleUInt) = a == b

@inline function Base.isless(a::NTupleUInt{N,W}, b::NTupleUInt{N,W}) where {N, W<:Union{UInt32,UInt64}}
    for i in N:-1:1
        a.data[i] < b.data[i] && return true
        a.data[i] > b.data[i] && return false
    end
    return false
end

Base.:<(a::NTupleUInt, b::NTupleUInt)  = isless(a, b)
Base.:>(a::NTupleUInt, b::NTupleUInt)  = isless(b, a)
Base.:<=(a::NTupleUInt, b::NTupleUInt) = !isless(b, a)
Base.:>=(a::NTupleUInt, b::NTupleUInt) = !isless(a, b)


# Ripple-carry addition and subtraction, needed by Bits.mask which computes
# `(one(T) << i) - one(T)`.
@inline function Base.:+(a::NTupleUInt{N,W}, b::NTupleUInt{N,W}) where {N, W<:Union{UInt32,UInt64}}
    data = Vector{W}(undef, N)
    carry = zero(W)
    @inbounds for i in 1:N
        s  = a.data[i] + b.data[i]
        c1 = s < a.data[i] ? one(W) : zero(W)
        s2 = s + carry
        c2 = s2 < s ? one(W) : zero(W)
        carry   = c1 | c2
        data[i] = s2
    end
    NTupleUInt{N,W}(NTuple{N,W}(data))
end

@inline function Base.:-(a::NTupleUInt{N,W}, b::NTupleUInt{N,W}) where {N, W<:Union{UInt32,UInt64}}
    data = Vector{W}(undef, N)
    borrow = zero(W)
    @inbounds for i in 1:N
        ai     = a.data[i]
        bi     = b.data[i] + borrow
        borrow = (bi < b.data[i] || ai < bi) ? one(W) : zero(W)
        data[i] = ai - bi
    end
    NTupleUInt{N,W}(NTuple{N,W}(data))
end

@inline Base.:+(a::NTupleUInt{N,W}, b::Integer) where {N,W} = a + NTupleUInt{N,W}(b)
@inline Base.:-(a::NTupleUInt{N,W}, b::Integer) where {N,W} = a - NTupleUInt{N,W}(b)
@inline Base.:+(b::Integer, a::NTupleUInt{N,W}) where {N,W} = NTupleUInt{N,W}(b) + a


@inline function Base.count_ones(a::NTupleUInt{N,W}) where {N,W}
    s = 0
    for i in 1:N
        s += count_ones(a.data[i])
    end
    return s
end


Base.UInt64(a::NTupleUInt{N,W}) where {N,W} = UInt64(a.data[1])
Base.Int(a::NTupleUInt) = Int(UInt64(a))

Base.convert(::Type{NTupleUInt{N,W}}, x::Integer) where {N,W} = NTupleUInt{N,W}(x)
Base.convert(::Type{NTupleUInt{N,W}}, x::NTupleUInt{N,W}) where {N,W} = x

function Base.convert(::Type{NTupleUInt{N2,W2}}, x::NTupleUInt{N1,W1}) where {N1,W1,N2,W2}
    val = BigInt(0)
    wb1 = 8 * sizeof(W1)
    for i in N1:-1:1
        val = (val << wb1) | BigInt(x.data[i])
    end
    return NTupleUInt{N2,W2}(val)
end


# hash must match Base integer hash so that hash(NTupleUInt(x)) == hash(x) for small x.
function Base.hash(a::NTupleUInt{N,W}, h::UInt) where {N,W}
    # Round-trip through BigInt and use Base.hash_integer, which is what all
    # standard integer types use. This ensures hash(zero(T)) == hash(0) etc.
    val = BigInt(0)
    wb  = 8 * sizeof(W)
    for i in N:-1:1
        val = (val << wb) | BigInt(a.data[i])
    end
    return Base.hash_integer(val, h)
end


function Base.show(io::IO, a::NTupleUInt{N,W}) where {N,W}
    wb  = 8 * sizeof(W)
    hex = ""
    for i in N:-1:1
        hex *= string(a.data[i]; base=16, pad=wb÷4)
    end
    print(io, "NTupleUInt{$N,$W}(0x", hex, ")")
end


_bitpaulimultiply(pstr1::NTupleUInt, pstr2::NTupleUInt) = pstr1 ⊻ pstr2

_paulishiftright(pstr::NTupleUInt) = pstr >> 2

# Bridge so callers can pass a Type as well as an instance.
# The @generated alternatingmask in bitoperations.jl only accepts instances.
alternatingmask(::Type{T}) where {T<:NTupleUInt} = alternatingmask(zero(T))