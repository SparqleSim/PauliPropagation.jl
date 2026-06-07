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
This type is intended to eventually become a standalone package.
"""
struct NTupleUInt{N,W<:Union{UInt32,UInt64}}
    data::NTuple{N,W}

    NTupleUInt{N,W}(data::NTuple{N,W}) where {N, W<:Union{UInt32,UInt64}} = new{N,W}(data)

    function NTupleUInt{N,W}(x::Integer) where {N, W<:Union{UInt32,UInt64}}
        bx   = BigInt(x)
        bmask = BigInt(typemax(W))
        wb   = 8 * sizeof(W)
        words = ntuple(i -> W((bx >> ((i - 1) * wb)) & bmask), Val(N))
        return new{N,W}(words)
    end
end

"""
    NTupleUInt{N,W}()

Construct the zero (identity) Pauli string.
"""
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

_wordbits(::Type{NTupleUInt{N,W}}) where {N,W} = 8 * sizeof(W)
_wordbits(x::NTupleUInt) = _wordbits(typeof(x))

bitsize(::Type{NTupleUInt{N,W}}) where {N,W} = N * 8 * sizeof(W)
bitsize(x::NTupleUInt) = bitsize(typeof(x))

bitsize(::Type{T}) where {T<:Integer} = 8 * sizeof(T)
bitsize(x::Integer) = bitsize(typeof(x))

"""
    max_qubits(::Type{NTupleUInt{N,W}})

Return the maximum number of qubits this type can represent.
"""
max_qubits(::Type{NTupleUInt{N,W}}) where {N,W} = N * 4 * sizeof(W)


"""
    getchunkedinttype(nqubits; word=UInt32)

Return the smallest `NTupleUInt{N,W}` type that can represent `nqubits` qubits.
Defaults to `UInt32` words for maximum GPU register compatibility.
"""
function getchunkedinttype(nqubits::Integer; word::Type{W}=UInt32) where {W}
    bits_needed = 2 * nqubits
    bits_per_word = 8 * sizeof(W)
    N = cld(bits_needed, bits_per_word)
    return NTupleUInt{N,W}
end


@inline function Base.:~(a::NTupleUInt{N,W}) where {N,W}
    NTupleUInt{N,W}(map(~, a.data))
end

@inline function Base.:&(a::NTupleUInt{N,W}, b::NTupleUInt{N,W}) where {N,W}
    NTupleUInt{N,W}(ntuple(i -> a.data[i] & b.data[i], Val(N)))
end

@inline function Base.:|(a::NTupleUInt{N,W}, b::NTupleUInt{N,W}) where {N,W}
    NTupleUInt{N,W}(ntuple(i -> a.data[i] | b.data[i], Val(N)))
end

@inline function Base.:⊻(a::NTupleUInt{N,W}, b::NTupleUInt{N,W}) where {N,W}
    NTupleUInt{N,W}(ntuple(i -> a.data[i] ⊻ b.data[i], Val(N)))
end


"""
    >>(a::NTupleUInt{N,W}, k::Integer)

Right-shift by `k` bits, carrying bits across word boundaries.
Words are in little-endian order (`data[1]` is the least-significant word).
"""
@inline function Base.:>>(a::NTupleUInt{N,W}, k::Integer) where {N,W}
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

"""
    <<(a::NTupleUInt{N,W}, k::Integer)

Left-shift by `k` bits, carrying bits across word boundaries.
"""
@inline function Base.:(<<)(a::NTupleUInt{N,W}, k::Integer) where {N,W}
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


@inline function Base.:(==)(a::NTupleUInt{N,W}, b::NTupleUInt{N,W}) where {N,W}
    for i in 1:N
        a.data[i] != b.data[i] && return false
    end
    return true
end

@inline function Base.:(==)(a::NTupleUInt{N,W}, b::Integer) where {N,W}
    a.data[1] == W(b & typemax(W)) || return false
    for i in 2:N
        a.data[i] != zero(W) && return false
    end
    return true
end

@inline Base.:(==)(b::Integer, a::NTupleUInt) = a == b

"""
    isless(a, b)

Compares from the most-significant word downward, giving lexicographic integer ordering.
This is what `sort!` on a `VectorPauliSum` term array relies on.
"""
@inline function Base.isless(a::NTupleUInt{N,W}, b::NTupleUInt{N,W}) where {N,W}
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


@inline function Base.count_ones(a::NTupleUInt{N,W}) where {N,W}
    s = 0
    for i in 1:N
        s += count_ones(a.data[i])
    end
    return s
end


Base.UInt64(a::NTupleUInt{N,W}) where {N,W} = UInt64(a.data[1])
Base.Int(a::NTupleUInt) = Int(UInt64(a))

Base.convert(::Type{NTupleUInt{N,W}}, x::Integer) where {N,W} =
    NTupleUInt{N,W}(x)

Base.convert(::Type{NTupleUInt{N,W}}, x::NTupleUInt{N,W}) where {N,W} = x

function Base.convert(::Type{NTupleUInt{N2,W2}}, x::NTupleUInt{N1,W1}) where {N1,W1,N2,W2}
    val = BigInt(0)
    wb1 = 8 * sizeof(W1)
    for i in N1:-1:1
        val = (val << wb1) | BigInt(x.data[i])
    end
    return NTupleUInt{N2,W2}(val)
end


function Base.hash(a::NTupleUInt{N,W}, h::UInt) where {N,W}
    for i in 1:N
        h = hash(a.data[i], h)
    end
    return h
end


function Base.show(io::IO, a::NTupleUInt{N,W}) where {N,W}
    wb = 8 * sizeof(W)
    hex = ""
    for i in N:-1:1
        hex *= string(a.data[i]; base=16, pad=wb÷4)
    end
    print(io, "NTupleUInt{$N,$W}(0x", hex, ")")
end


"""
    alternatingmask(::Type{NTupleUInt{N,W}})

Return the compile-time constant mask `...10101010101` used by the Pauli bit-twiddling
routines. Even-indexed bits are 1, odd-indexed bits are 0.
"""
@generated function alternatingmask(::Type{NTupleUInt{N,W}}) where {N,W}
    wb = 8 * sizeof(W)
    word_mask = zero(W)
    for bit in 0:(wb-1)
        if bit % 2 == 0
            word_mask |= W(1) << bit
        end
    end
    words = ntuple(_ -> word_mask, N)
    return :(NTupleUInt{$N,$W}($words))
end

alternatingmask(x::NTupleUInt{N,W}) where {N,W} =
    alternatingmask(NTupleUInt{N,W})


function _countbitweight(pstr::NTupleUInt)
    mask = alternatingmask(pstr)
    m1 = pstr & mask
    m2 = pstr & (mask << 1)
    res = m1 | (m2 >> 1)
    return count_ones(res)
end

function _countbitxy(pstr::NTupleUInt)
    mask = alternatingmask(pstr)
    op = pstr ⊻ (pstr >> 1)
    op = op & mask
    return count_ones(op)
end

function _countbityz(pstr::NTupleUInt)
    mask = alternatingmask(pstr)
    op = pstr & (mask << 1)
    return count_ones(op)
end

function _countbitx(pstr::NTupleUInt)
    mask_x = alternatingmask(pstr)
    mask_y = mask_x << 1
    xs = (pstr & mask_x) & ((~pstr & mask_y) >> 1)
    return count_ones(xs)
end

function _countbity(pstr::NTupleUInt)
    mask_x = alternatingmask(pstr)
    mask_y = mask_x << 1
    op = ((pstr & mask_y) >> 1) & (~pstr & mask_x)
    return count_ones(op)
end

function _countbitz(pstr::NTupleUInt)
    mask_x = alternatingmask(pstr)
    mask_y = mask_x << 1
    op = ((pstr & mask_y) >> 1) & (pstr & mask_x)
    return count_ones(op)
end

function _bitcommutes(pstr1::NTupleUInt, pstr2::NTupleUInt)
    mask0 = alternatingmask(pstr1)
    mask1 = mask0 << 1
    aBits0 = mask0 & pstr1
    aBits1 = (mask1 & pstr1) >> 1
    bBits0 = mask0 & pstr2
    bBits1 = (mask1 & pstr2) >> 1
    flags = (aBits0 & bBits1) ⊻ (aBits1 & bBits0)
    return (count_ones(flags) % 2) == 0
end

_bitpaulimultiply(pstr1::NTupleUInt, pstr2::NTupleUInt) = pstr1 ⊻ pstr2

_paulishiftright(pstr::NTupleUInt) = pstr >> 2