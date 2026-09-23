###
##
# An unsigned integer of 64-bit limbs, for terms wider than a machine word, where a primitive integer compiles to whole-value operations that slow down past a width set by the compiler version.
# Every operation is unrolled over the limbs at compile time, and only a shift by a runtime amount indexes the tuple with a runtime value.
# It is an `Unsigned`, so that every signature written for the machine term types takes it as well, while the generic arithmetic `Base` builds on `div`, `rem` and `*`, which a term never needs, stays undefined.
##
###

"""
    NTupleInteger{N}

An unsigned integer of `N` 64-bit limbs, little-endian: `limbs[1]` holds the lowest bits.
Every operation is unrolled over the limbs at compile time.
It is the term type of a Pauli sum on more than 64 qubits, see `getinttype`.
Values convert to and from the machine integers and `BigInt`, compare with them, and hash like the `UInt64` they equal when they fit in one.
"""
struct NTupleInteger{N} <: Unsigned
    limbs::NTuple{N,UInt64}

    # without it, the constructor generated for any other argument is ambiguous with the ones `Base`
    # defines for converting to a `Number`
    NTupleInteger{N}(limbs::NTuple{N,UInt64}) where {N} = new{N}(limbs)
end

### Construction and conversion

NTupleInteger{N}(x::NTupleInteger{N}) where {N} = x

function NTupleInteger{N}(x::NTupleInteger{M}) where {N,M}
    if M > N && !_iszeroabove(x, Val(N))
        throw(InexactError(:NTupleInteger, NTupleInteger{N}, x))
    end
    return NTupleInteger{N}(ntuple(k -> k <= M ? x.limbs[k] : zero(UInt64), Val(N)))
end

function NTupleInteger{N}(x::Union{Base.BitInteger64,Bool}) where {N}
    if x < 0
        throw(InexactError(:NTupleInteger, NTupleInteger{N}, x))
    end
    low = UInt64(x)
    return NTupleInteger{N}(ntuple(k -> ifelse(k == 1, low, zero(UInt64)), Val(N)))
end

function NTupleInteger{N}(x::Union{Int128,UInt128}) where {N}
    if x < 0
        throw(InexactError(:NTupleInteger, NTupleInteger{N}, x))
    end
    low = x % UInt64
    high = (x >> 64) % UInt64
    if N < 2 && !iszero(high)
        throw(InexactError(:NTupleInteger, NTupleInteger{N}, x))
    end
    return NTupleInteger{N}(ntuple(k -> k == 1 ? low : k == 2 ? high : zero(UInt64), Val(N)))
end

function NTupleInteger{N}(x::BigInt) where {N}
    if x < 0 || !iszero(x >> (64 * N))
        throw(InexactError(:NTupleInteger, NTupleInteger{N}, x))
    end
    return NTupleInteger{N}(ntuple(k -> UInt64((x >> (64 * (k - 1))) & typemax(UInt64)), Val(N)))
end

NTupleInteger{N}(x::Integer) where {N} = NTupleInteger{N}(BigInt(x))

Base.convert(::Type{NTupleInteger{N}}, x::Integer) where {N} = NTupleInteger{N}(x)
Base.convert(::Type{NTupleInteger{N}}, x::NTupleInteger{M}) where {N,M} = NTupleInteger{N}(x)
Base.convert(::Type{T}, x::NTupleInteger) where {T<:Union{Base.BitInteger,BigInt}} = T(x)

# arithmetic with another integer type promotes; comparison has its own methods below, which read a
# negative value that a conversion to an unsigned type cannot
Base.promote_rule(::Type{NTupleInteger{N}}, ::Type{NTupleInteger{M}}) where {N,M} = NTupleInteger{max(N, M)}
Base.promote_rule(::Type{NTupleInteger{N}}, ::Type{<:Union{Base.BitInteger,Bool}}) where {N} = NTupleInteger{N}
Base.promote_rule(::Type{NTupleInteger{N}}, ::Type{BigInt}) where {N} = BigInt

function Base.UInt64(x::NTupleInteger)
    if !_iszeroabove(x, Val(1))
        throw(InexactError(:UInt64, UInt64, x))
    end
    return x.limbs[1]
end

(::Type{T})(x::NTupleInteger) where {T<:Base.BitInteger64} = T(UInt64(x))

function (::Type{T})(x::NTupleInteger) where {T<:Union{Int128,UInt128}}
    if !_iszeroabove(x, Val(2))
        throw(InexactError(Symbol(T), T, x))
    end
    return T(_low128(x))
end

function Base.BigInt(x::NTupleInteger{N}) where {N}
    value = BigInt(0)
    for k in N:-1:1
        value = (value << 64) | BigInt(x.limbs[k])
    end
    return value
end

# the low 128 bits
@inline function _low128(x::NTupleInteger{N}) where {N}
    high = N >= 2 ? x.limbs[min(2, N)] : zero(UInt64)
    return (UInt128(high) << 64) | UInt128(x.limbs[1])
end

# whether every limb above the `K`-th is zero
@generated function _iszeroabove(x::NTupleInteger{N}, ::Val{K}) where {N,K}
    folds = [:(rest |= x.limbs[$k]) for k in K+1:N]
    return quote
        Base.@_inline_meta
        rest = zero(UInt64)
        $(folds...)
        return iszero(rest)
    end
end

Base.rem(x::NTupleInteger, ::Type{T}) where {T<:Base.BitInteger64} = x.limbs[1] % T
Base.rem(x::NTupleInteger, ::Type{T}) where {T<:Union{Int128,UInt128}} = _low128(x) % T
Base.rem(x::NTupleInteger{M}, ::Type{NTupleInteger{N}}) where {M,N} =
    NTupleInteger{N}(ntuple(k -> k <= M ? x.limbs[k] : zero(UInt64), Val(N)))

# a negative machine integer wraps as its two's complement, as `rem` to an unsigned type does
function Base.rem(x::Base.BitInteger, ::Type{NTupleInteger{N}}) where {N}
    wrapped = x % UInt128
    fill = x < 0 ? typemax(UInt64) : zero(UInt64)
    return NTupleInteger{N}(ntuple(k -> k == 1 ? wrapped % UInt64 : k == 2 ? (wrapped >> 64) % UInt64 : fill, Val(N)))
end

Base.zero(::Type{NTupleInteger{N}}) where {N} = NTupleInteger{N}(ntuple(_ -> zero(UInt64), Val(N)))
Base.one(::Type{NTupleInteger{N}}) where {N} = NTupleInteger{N}(ntuple(k -> ifelse(k == 1, one(UInt64), zero(UInt64)), Val(N)))
Base.typemin(::Type{NTupleInteger{N}}) where {N} = zero(NTupleInteger{N})
Base.typemax(::Type{NTupleInteger{N}}) where {N} = NTupleInteger{N}(ntuple(_ -> typemax(UInt64), Val(N)))
Base.zero(x::NTupleInteger) = zero(typeof(x))
Base.one(x::NTupleInteger) = one(typeof(x))
Base.typemin(x::NTupleInteger) = typemin(typeof(x))
Base.typemax(x::NTupleInteger) = typemax(typeof(x))

Base.iszero(x::NTupleInteger) = _iszeroabove(x, Val(0))
Base.isone(x::NTupleInteger) = isone(x.limbs[1]) && _iszeroabove(x, Val(1))
Base.iseven(x::NTupleInteger) = iseven(x.limbs[1])
Base.isodd(x::NTupleInteger) = isodd(x.limbs[1])

Random.rand(rng::Random.AbstractRNG, ::Random.SamplerType{NTupleInteger{N}}) where {N} =
    NTupleInteger{N}(ntuple(_ -> rand(rng, UInt64), Val(N)))

### Bitwise operations

@inline Base.:~(x::NTupleInteger{N}) where {N} = NTupleInteger{N}(ntuple(k -> ~x.limbs[k], Val(N)))
@inline Base.:&(x::NTupleInteger{N}, y::NTupleInteger{N}) where {N} = NTupleInteger{N}(ntuple(k -> x.limbs[k] & y.limbs[k], Val(N)))
@inline Base.:|(x::NTupleInteger{N}, y::NTupleInteger{N}) where {N} = NTupleInteger{N}(ntuple(k -> x.limbs[k] | y.limbs[k], Val(N)))
@inline Base.xor(x::NTupleInteger{N}, y::NTupleInteger{N}) where {N} = NTupleInteger{N}(ntuple(k -> x.limbs[k] ⊻ y.limbs[k], Val(N)))

@generated function Base.count_ones(x::NTupleInteger{N}) where {N}
    return quote
        Base.@_inline_meta
        n = 0
        Base.Cartesian.@nexprs $N k -> (n += count_ones(x.limbs[k]))
        return n
    end
end

Base.count_zeros(x::NTupleInteger{N}) where {N} = 64 * N - count_ones(x)

@generated function Base.trailing_zeros(x::NTupleInteger{N}) where {N}
    checks = [:(if !iszero(x.limbs[$k]); return $(64 * (k - 1)) + trailing_zeros(x.limbs[$k]); end) for k in 1:N]
    return quote
        Base.@_inline_meta
        $(checks...)
        return $(64 * N)
    end
end

@generated function Base.leading_zeros(x::NTupleInteger{N}) where {N}
    checks = [:(if !iszero(x.limbs[$k]); return $(64 * (N - k)) + leading_zeros(x.limbs[$k]); end) for k in N:-1:1]
    return quote
        Base.@_inline_meta
        $(checks...)
        return $(64 * N)
    end
end

# A shift by a runtime amount is a limb offset and a funnel shift between neighbouring limbs, linear
# in the limbs. The limb index depends on the amount, so this is the one place the tuple is indexed
# at runtime; the guard on the index is what makes the unchecked read safe.
@inline function Base.:>>(x::NTupleInteger{N}, k::Int) where {N}
    if k < 0
        return x << unsigned(-k)
    end
    limb_shift = k >> 6
    bit_shift = k & 63
    limbs = x.limbs
    function shifted_limb(i)
        low = i + limb_shift <= N ? (@inbounds limbs[i+limb_shift]) : zero(UInt64)
        high = i + limb_shift + 1 <= N ? (@inbounds limbs[i+limb_shift+1]) : zero(UInt64)
        return (low >> bit_shift) | (high << (64 - bit_shift))
    end
    return NTupleInteger{N}(ntuple(shifted_limb, Val(N)))
end

@inline function Base.:<<(x::NTupleInteger{N}, k::Int) where {N}
    if k < 0
        return x >> unsigned(-k)
    end
    limb_shift = k >> 6
    bit_shift = k & 63
    limbs = x.limbs
    function shifted_limb(i)
        high = i - limb_shift >= 1 ? (@inbounds limbs[i-limb_shift]) : zero(UInt64)
        low = i - limb_shift - 1 >= 1 ? (@inbounds limbs[i-limb_shift-1]) : zero(UInt64)
        return (high << bit_shift) | (low >> (64 - bit_shift))
    end
    return NTupleInteger{N}(ntuple(shifted_limb, Val(N)))
end

# an unsigned value shifts the same way either way
@inline Base.:>>>(x::NTupleInteger, k::Int) = x >> k

# `Base` forwards a shift by any other integer type to one by a `UInt`, and by more than the width
# everything has been shifted out already
@inline Base.:>>(x::NTupleInteger{N}, k::UInt) where {N} = x >> Int(min(k, UInt(64 * N)))
@inline Base.:<<(x::NTupleInteger{N}, k::UInt) where {N} = x << Int(min(k, UInt(64 * N)))
@inline Base.:>>>(x::NTupleInteger{N}, k::UInt) where {N} = x >> Int(min(k, UInt(64 * N)))

### Comparison and hashing

@generated function Base.:(==)(x::NTupleInteger{N}, y::NTupleInteger{N}) where {N}
    return quote
        Base.@_inline_meta
        differ = zero(UInt64)
        Base.Cartesian.@nexprs $N k -> (differ |= x.limbs[k] ⊻ y.limbs[k])
        return iszero(differ)
    end
end

# the narrower value is zero-extended, which is exact
function Base.:(==)(x::NTupleInteger{N}, y::NTupleInteger{M}) where {N,M}
    if N < M
        return y == x
    end
    return x == (y % NTupleInteger{N})
end

# The highest limb that differs decides, as the integer order does.
# It is found as the one with the highest score, which is odd where x is the smaller, so the comparison is a maximum over the limbs with no branch on which limb differs.
@generated function Base.isless(x::NTupleInteger{N}, y::NTupleInteger{N}) where {N}
    return quote
        Base.@_inline_meta
        top = zero(UInt64)
        Base.Cartesian.@nexprs $N k -> begin
            score_k = ifelse(x.limbs[k] != y.limbs[k], UInt64(2k) | UInt64(x.limbs[k] < y.limbs[k]), zero(UInt64))
            top = max(top, score_k)
        end
        return isodd(top)
    end
end

@inline Base.:<=(x::NTupleInteger{N}, y::NTupleInteger{N}) where {N} = !isless(y, x)

# a `Real` compares through promotion unless `<` has a method of its own
@inline Base.:<(x::NTupleInteger{N}, y::NTupleInteger{N}) where {N} = isless(x, y)

Base.:(==)(x::NTupleInteger, y::Union{Base.BitInteger64,Bool}) = _iszeroabove(x, Val(1)) && x.limbs[1] == y
Base.:(==)(x::NTupleInteger, y::Union{Int128,UInt128}) = _iszeroabove(x, Val(2)) && _low128(x) == y
Base.:(==)(x::NTupleInteger, y::BigInt) = BigInt(x) == y
Base.:(==)(y::Union{Base.BitInteger,Bool}, x::NTupleInteger) = x == y
Base.:(==)(y::BigInt, x::NTupleInteger) = x == y

function Base.isless(x::NTupleInteger, y::Union{Base.BitInteger64,Bool})
    if y < 0 || !_iszeroabove(x, Val(1))
        return false
    end
    return x.limbs[1] < UInt64(y)
end

function Base.isless(y::Union{Base.BitInteger64,Bool}, x::NTupleInteger)
    if y < 0 || !_iszeroabove(x, Val(1))
        return true
    end
    return UInt64(y) < x.limbs[1]
end

Base.isless(x::NTupleInteger, y::Union{Int128,UInt128,BigInt}) = isless(BigInt(x), y)
Base.isless(y::Union{Int128,UInt128,BigInt}, x::NTupleInteger) = isless(y, BigInt(x))

Base.:<(x::NTupleInteger, y::Union{Base.BitInteger,Bool}) = isless(x, y)
Base.:<(y::Union{Base.BitInteger,Bool}, x::NTupleInteger) = isless(y, x)
Base.:<(x::NTupleInteger, y::BigInt) = isless(x, y)
Base.:<(y::BigInt, x::NTupleInteger) = isless(y, x)

# A value that fits a machine word hashes as that word, since it compares equal to it.
@generated function Base.hash(x::NTupleInteger{N}, h::UInt) where {N}
    return quote
        Base.@_inline_meta
        rest = zero(UInt64)
        Base.Cartesian.@nexprs $(N - 1) k -> (rest |= x.limbs[k+1])
        if iszero(rest)
            return hash(x.limbs[1], h)
        end
        Base.Cartesian.@nexprs $N k -> (h = hash(x.limbs[k], h))
        return h
    end
end

### Arithmetic

# The carry is threaded through every limb by the generated code, so that inference sees one
# straight-line function instead of a recursion it may stop unrolling.
@generated function Base.:+(x::NTupleInteger{N}, y::NTupleInteger{N}) where {N}
    return quote
        Base.@_inline_meta
        carry = zero(UInt64)
        Base.Cartesian.@nexprs $N k -> begin
            partial_k = x.limbs[k] + y.limbs[k]
            sum_k = partial_k + carry
            carry = UInt64((partial_k < x.limbs[k]) | (sum_k < partial_k))
        end
        return NTupleInteger{N}(Base.Cartesian.@ntuple $N k -> sum_k)
    end
end

@generated function Base.:-(x::NTupleInteger{N}, y::NTupleInteger{N}) where {N}
    return quote
        Base.@_inline_meta
        borrow = zero(UInt64)
        Base.Cartesian.@nexprs $N k -> begin
            subtrahend_k = y.limbs[k] + borrow
            difference_k = x.limbs[k] - subtrahend_k
            borrow = UInt64((subtrahend_k < y.limbs[k]) | (x.limbs[k] < subtrahend_k))
        end
        return NTupleInteger{N}(Base.Cartesian.@ntuple $N k -> difference_k)
    end
end

### Printing

function Base.show(io::IO, x::NTupleInteger{N}) where {N}
    print(io, "NTupleInteger{", N, "}(0x")
    for k in N:-1:1
        print(io, string(x.limbs[k]; base=16, pad=16))
    end
    print(io, ")")
end

function Base.string(x::NTupleInteger; base::Union{Nothing,Integer}=nothing, pad::Integer=1)
    if base === nothing
        return sprint(show, x)
    end
    return string(BigInt(x); base, pad)
end

Base.bitstring(x::NTupleInteger{N}) where {N} = join(bitstring(x.limbs[k]) for k in N:-1:1)
