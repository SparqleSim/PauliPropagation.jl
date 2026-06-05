"""
    NTuplePauliString{N,W<:Union{UInt32,UInt64}}

Encodes a Pauli string as an `NTuple{N,W}` of unsigned words.

`N` is the number of machine words and `W` is the word type (`UInt32` or `UInt64`).
Together they provide `N * 8 * sizeof(W)` bits, enough for `N * 4 * sizeof(W)` qubits.

Words are stored in little-endian order: `data[1]` holds the least-significant bits
(qubits 1 through `8*sizeof(W)/2`) and `data[N]` holds the most-significant bits.

Encoding: each qubit occupies 2 bits -- I=00, X=01, Y=10, Z=11 -- with qubit k
in bit positions 2*(k-1) and 2*k-1, matching the rest of PauliPropagation.jl.

Typical configurations:
  - 64 qubits  : NTuplePauliString{2, UInt64}
  - 128 qubits : NTuplePauliString{4, UInt64}
  - 256 qubits : NTuplePauliString{8, UInt64}  or  NTuplePauliString{16, UInt32}

Use `UInt32` words when targeting GPUs that run 32-bit register operations natively.
"""
struct NTuplePauliString{N,W<:Union{UInt32,UInt64}}
    data::NTuple{N,W}
end


"""
    NTuplePauliString{N,W}(x::Integer)

Construct from a (small) integer, zero-extending into the higher words.
"""
function NTuplePauliString{N,W}(x::Integer) where {N, W<:Union{UInt32,UInt64}}
    words = ntuple(Val(N)) do i
        shift = (i - 1) * (8 * sizeof(W))
        W((BigInt(x) >> shift) & BigInt(typemax(W)))
    end
    return NTuplePauliString{N,W}(words)
end

"""
    NTuplePauliString{N,W}()

Construct the zero (identity) Pauli string.
"""
NTuplePauliString{N,W}() where {N,W} =
    NTuplePauliString{N,W}(ntuple(_ -> zero(W), Val(N)))


Base.zero(::Type{NTuplePauliString{N,W}}) where {N,W} =
    NTuplePauliString{N,W}(ntuple(_ -> zero(W), Val(N)))

Base.zero(x::NTuplePauliString{N,W}) where {N,W} = zero(typeof(x))

Base.one(::Type{NTuplePauliString{N,W}}) where {N,W} =
    NTuplePauliString{N,W}(ntuple(i -> i == 1 ? one(W) : zero(W), Val(N)))

Base.one(x::NTuplePauliString{N,W}) where {N,W} = one(typeof(x))

Base.typemax(::Type{NTuplePauliString{N,W}}) where {N,W} =
    NTuplePauliString{N,W}(ntuple(_ -> typemax(W), Val(N)))

_wordbits(::Type{NTuplePauliString{N,W}}) where {N,W} = 8 * sizeof(W)
_wordbits(x::NTuplePauliString) = _wordbits(typeof(x))

bitsize(::Type{NTuplePauliString{N,W}}) where {N,W} = N * 8 * sizeof(W)
bitsize(x::NTuplePauliString) = bitsize(typeof(x))

"""
    max_qubits(::Type{NTuplePauliString{N,W}})

Return the maximum number of qubits this type can represent.
"""
max_qubits(::Type{NTuplePauliString{N,W}}) where {N,W} = N * 4 * sizeof(W)


"""
    getntupleinttype(nqubits; word=UInt32)

Return the smallest `NTuplePauliString{N,W}` type that can represent `nqubits` qubits.
Defaults to `UInt32` words for maximum GPU register compatibility.
"""
function getntupleinttype(nqubits::Integer; word::Type{W}=UInt32) where {W}
    bits_needed = 2 * nqubits
    bits_per_word = 8 * sizeof(W)
    N = cld(bits_needed, bits_per_word)
    return NTuplePauliString{N,W}
end


@inline function Base.:~(a::NTuplePauliString{N,W}) where {N,W}
    NTuplePauliString{N,W}(map(~, a.data))
end

@inline function Base.:&(a::NTuplePauliString{N,W}, b::NTuplePauliString{N,W}) where {N,W}
    NTuplePauliString{N,W}(ntuple(i -> a.data[i] & b.data[i], Val(N)))
end

@inline function Base.:|(a::NTuplePauliString{N,W}, b::NTuplePauliString{N,W}) where {N,W}
    NTuplePauliString{N,W}(ntuple(i -> a.data[i] | b.data[i], Val(N)))
end

@inline function Base.:⊻(a::NTuplePauliString{N,W}, b::NTuplePauliString{N,W}) where {N,W}
    NTuplePauliString{N,W}(ntuple(i -> a.data[i] ⊻ b.data[i], Val(N)))
end


"""
    >>(a::NTuplePauliString{N,W}, k::Integer)

Right-shift by `k` bits, carrying bits across word boundaries.
Words are in little-endian order (`data[1]` is the least-significant word).
"""
@inline function Base.:>>(a::NTuplePauliString{N,W}, k::Integer) where {N,W}
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
    NTuplePauliString{N,W}(new_data)
end

"""
    <<(a::NTuplePauliString{N,W}, k::Integer)

Left-shift by `k` bits, carrying bits across word boundaries.
"""
@inline function Base.:(<<)(a::NTuplePauliString{N,W}, k::Integer) where {N,W}
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
    NTuplePauliString{N,W}(new_data)
end


@inline function Base.:(==)(a::NTuplePauliString{N,W}, b::NTuplePauliString{N,W}) where {N,W}
    for i in 1:N
        a.data[i] != b.data[i] && return false
    end
    return true
end

@inline function Base.:(==)(a::NTuplePauliString{N,W}, b::Integer) where {N,W}
    a.data[1] == W(b & typemax(W)) || return false
    for i in 2:N
        a.data[i] != zero(W) && return false
    end
    return true
end

@inline Base.:(==)(b::Integer, a::NTuplePauliString) = a == b

"""
    isless(a, b)

Compares from the most-significant word downward, giving lexicographic integer ordering.
This is what `sort!` on a `VectorPauliSum` term array relies on.
"""
@inline function Base.isless(a::NTuplePauliString{N,W}, b::NTuplePauliString{N,W}) where {N,W}
    for i in N:-1:1
        a.data[i] < b.data[i] && return true
        a.data[i] > b.data[i] && return false
    end
    return false
end

Base.:<(a::NTuplePauliString, b::NTuplePauliString)  = isless(a, b)
Base.:>(a::NTuplePauliString, b::NTuplePauliString)  = isless(b, a)
Base.:<=(a::NTuplePauliString, b::NTuplePauliString) = !isless(b, a)
Base.:>=(a::NTuplePauliString, b::NTuplePauliString) = !isless(a, b)


@inline function Base.count_ones(a::NTuplePauliString{N,W}) where {N,W}
    s = 0
    for i in 1:N
        s += count_ones(a.data[i])
    end
    return s
end


Base.UInt64(a::NTuplePauliString{N,W}) where {N,W} = UInt64(a.data[1])
Base.Int(a::NTuplePauliString) = Int(UInt64(a))

Base.convert(::Type{NTuplePauliString{N,W}}, x::Integer) where {N,W} =
    NTuplePauliString{N,W}(x)

Base.convert(::Type{NTuplePauliString{N,W}}, x::NTuplePauliString{N,W}) where {N,W} = x

function Base.convert(::Type{NTuplePauliString{N2,W2}}, x::NTuplePauliString{N1,W1}) where {N1,W1,N2,W2}
    val = BigInt(0)
    wb1 = 8 * sizeof(W1)
    for i in N1:-1:1
        val = (val << wb1) | BigInt(x.data[i])
    end
    return NTuplePauliString{N2,W2}(val)
end


function Base.hash(a::NTuplePauliString{N,W}, h::UInt) where {N,W}
    for i in 1:N
        h = hash(a.data[i], h)
    end
    return h
end


function Base.show(io::IO, a::NTuplePauliString{N,W}) where {N,W}
    wb = 8 * sizeof(W)
    hex = ""
    for i in N:-1:1
        hex *= string(a.data[i]; base=16, pad=wb÷4)
    end
    print(io, "NTuplePauliString{$N,$W}(0x", hex, ")")
end


"""
    alternatingmask(::Type{NTuplePauliString{N,W}})

Return the compile-time constant mask `...10101010101` used by the Pauli bit-twiddling
routines. Even-indexed bits are 1, odd-indexed bits are 0.
"""
@generated function alternatingmask(::Type{NTuplePauliString{N,W}}) where {N,W}
    wb = 8 * sizeof(W)
    word_mask = zero(W)
    for bit in 0:(wb-1)
        if bit % 2 == 0
            word_mask |= W(1) << bit
        end
    end
    words = ntuple(_ -> word_mask, N)
    return :(NTuplePauliString{$N,$W}($words))
end

alternatingmask(x::NTuplePauliString{N,W}) where {N,W} =
    alternatingmask(NTuplePauliString{N,W})


function _countbitweight(pstr::NTuplePauliString)
    mask = alternatingmask(pstr)
    m1 = pstr & mask
    m2 = pstr & (mask << 1)
    res = m1 | (m2 >> 1)
    return count_ones(res)
end

function _countbitxy(pstr::NTuplePauliString)
    mask = alternatingmask(pstr)
    op = pstr ⊻ (pstr >> 1)
    op = op & mask
    return count_ones(op)
end

function _countbityz(pstr::NTuplePauliString)
    mask = alternatingmask(pstr)
    op = pstr & (mask << 1)
    return count_ones(op)
end

function _countbitx(pstr::NTuplePauliString)
    mask_x = alternatingmask(pstr)
    mask_y = mask_x << 1
    xs = (pstr & mask_x) & ((~pstr & mask_y) >> 1)
    return count_ones(xs)
end

function _countbity(pstr::NTuplePauliString)
    mask_x = alternatingmask(pstr)
    mask_y = mask_x << 1
    op = ((pstr & mask_y) >> 1) & (~pstr & mask_x)
    return count_ones(op)
end

function _countbitz(pstr::NTuplePauliString)
    mask_x = alternatingmask(pstr)
    mask_y = mask_x << 1
    op = ((pstr & mask_y) >> 1) & (pstr & mask_x)
    return count_ones(op)
end

function _bitcommutes(pstr1::NTuplePauliString, pstr2::NTuplePauliString)
    mask0 = alternatingmask(pstr1)
    mask1 = mask0 << 1
    aBits0 = mask0 & pstr1
    aBits1 = (mask1 & pstr1) >> 1
    bBits0 = mask0 & pstr2
    bBits1 = (mask1 & pstr2) >> 1
    flags = (aBits0 & bBits1) ⊻ (aBits1 & bBits0)
    return (count_ones(flags) % 2) == 0
end

_bitpaulimultiply(pstr1::NTuplePauliString, pstr2::NTuplePauliString) = pstr1 ⊻ pstr2

_paulishiftright(pstr::NTuplePauliString) = pstr >> 2


function _paulimask(::Type{NTuplePauliString{N,W}}, n_sites::Integer) where {N,W}
    wb = 8 * sizeof(W)
    nbits = 2 * n_sites
    full_words = nbits ÷ wb
    rem_bits   = nbits % wb

    data = ntuple(Val(N)) do i
        if i <= full_words
            typemax(W)
        elseif i == full_words + 1 && rem_bits > 0
            (one(W) << rem_bits) - one(W)
        else
            zero(W)
        end
    end
    return NTuplePauliString{N,W}(data)
end

_paulimask(x::NTuplePauliString{N,W}, n_sites::Integer) where {N,W} =
    _paulimask(NTuplePauliString{N,W}, n_sites)

function _pauliwindowmask(::Type{NTuplePauliString{N,W}}, index1::Integer, index2::Integer) where {N,W}
    base_mask = _paulimask(NTuplePauliString{N,W}, index2 - index1 + 1)
    shift = 2 * (index1 - 1)
    return base_mask << shift
end


function _getpaulibits(pstr::NTuplePauliString{N,W}, index::Integer) where {N,W}
    return _getpaulibits(pstr, index, index)
end

function _getpaulibits(pstr::NTuplePauliString{N,W}, index1::Integer, index2::Integer) where {N,W}
    T = NTuplePauliString{N,W}
    bitindex = 2 * (index1 - 1)
    shifted = pstr >> bitindex
    msk = _paulimask(T, index2 - index1 + 1)
    return shifted & msk
end

function _setpaulibits(pstr::NTuplePauliString{N,W}, target_pauli::Integer, index::Integer) where {N,W}
    return _setpaulibits(pstr, target_pauli, index, index)
end

function _setpaulibits(pstr::NTuplePauliString{N,W}, target_pstr, index1::Integer, index2::Integer) where {N,W}
    T = NTuplePauliString{N,W}
    bitindex = 2 * (index1 - 1)
    window_mask = _pauliwindowmask(T, index1, index2)
    target = T(target_pstr)
    return (pstr & ~window_mask) | (target << bitindex)
end
