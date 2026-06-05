"""
    getinttype(nqubits::Integer)

Function to return the smallest integer type that can hold `nqubits`.
This is the type that will be used internally for representing Pauli strings.
"""
function getinttype(nqubits::Integer)
    # we need 2 bits per qubit
    nbits = 2 * nqubits


    # just over 8.3 Million is the largest integer type we can generate
    for trial_bits in nbits:2:8_300_000

        # we can check if the number of bits is divisible by 8
        # othervise we know it cannot be defined
        if !(trial_bits % 8 == 0)
            continue
        end

        # special clauses for inbuilt integer types
        if trial_bits == 8
            return UInt8
        elseif trial_bits == 16
            return UInt16
        elseif trial_bits == 32
            return UInt32
        elseif trial_bits == 64
            return UInt64
        end
        # stop at 64 bits because I am suspicious of UInt128

        trial_inttype_expr = Symbol("UInt", trial_bits)
        # check if the integer type is defined to avoid overrides
        if isdefined(PauliPropagation, trial_inttype_expr)
            return eval(trial_inttype_expr)
        end

        # defining the integer type can fail for bit numbers that are odd not not natively supported
        # just try the next number if that happens
        try
            @eval @define_integers $trial_bits
            # return the newly defined unsigned integer type
            return eval(trial_inttype_expr)
        catch ErrorException
            continue
        end
    end

    # if we reach here, we have failed to define the integer type
    # Falling back to BigInt
    @warn "Failed to define integer types for $nqubits qubits. Falling back to BigInt."
    return BigInt
end


# A priated hash function for unsigned integers from BitIntegers.jl
# This hashes for the value of the integer and is a lot faster than the default hash function.
Base.hash(v::BitIntegers.AbstractBitUnsigned, h::UInt) = Base.hash_integer(v, h)


# This function counts the number of 00 bit pairs in the integer Pauli string.
function _countbitweight(pstr::PauliStringType)
    # get our super bit mask looking like ....1010101.
    mask = alternatingmask(pstr)

    # m1 carries the 1's of the Paulis on odd bits 
    m1 = pstr & mask

    # m2 carries the 1's of the pauliP on even bits
    m2 = pstr & (mask << 1)

    # OR between m1 and left-shifted m2 to get 1's where either m1 or m2 arre 1
    res = m1 | (m2 >> 1)

    # count 1's to get the number of non-identity Paulis
    return count_ones(res)
end


# This function counts the number of 01 (X) or 10 (Y) bit pairs in the integer Pauli string.
function _countbitxy(pstr::PauliStringType)
    # we use that 01 and 10 have exactly one 1 and one 0

    # get our super bit mask looking like ....1010101.
    mask = alternatingmask(pstr)

    # XOR to put 1's where the bits are different
    op = pstr ⊻ (pstr >> 1)

    # AND with the mask to extract the 1's
    op = op & mask

    # count 1's to get the number of X or Y Paulis
    return count_ones(op)
end


# This function counts the number of 10 (Y) or 11 (Z) bit pairs in the integer Pauli string.
function _countbityz(pstr::PauliStringType)
    # we use that both have a 1 on the left bit

    # get our super bit mask looking like ....1010101.
    mask = alternatingmask(pstr)

    # AND with the shifted mask to extract the 1's on the left bit
    op = pstr & (mask << 1)

    # count 1's to get the number of Y or Z Paulis
    return count_ones(op)
end


### Count X, Y, Z occurrences in PauliString ###
function _countbitx(pstr::PauliStringType)

    # super bit mask 
    mask_x = alternatingmask(pstr) # ....1010101 representing XXX...
    mask_y = mask_x << 1  # ...101010 representing YYY...

    # AND with NOT of Y to get only X
    xs = (pstr & mask_x) & ((~pstr & mask_y) >> 1) # Shift right to align

    # count 1 pairs to get the number of X Paulis
    return count_ones(xs)
end

function _countbity(pstr::PauliStringType)

    # super bit mask 
    mask_x = alternatingmask(pstr) # ....1010101 representing XXX...
    mask_y = mask_x << 1  # ...101010 representing YYY...

    # And with NOT of X to get only Y
    op = ((pstr & mask_y) >> 1 ) & (~pstr & mask_x)  # Shift right to align
    
    # count 1's to get the number of Y Paulis
    return count_ones(op)
end

function _countbitz(pstr::PauliStringType)

    # super bit mask 
    mask_x = alternatingmask(pstr) # ....1010101 representing XXX...
    mask_y = mask_x << 1  # ...101010 representing YYY...

    # AND with both masks to extract the 1's on both bits
    op = ((pstr & mask_y) >> 1 ) & (pstr & mask_x)  

    # count 1's to get the number of Z Paulis
    return count_ones(op)
end


# This function checks if two integer Pauli strings commute.
function _bitcommutes(pstr1::PauliStringType, pstr2::PauliStringType)

    mask0 = alternatingmask(pstr1)
    mask1 = mask0 << 1

    # obtain the left (then right) bits of each Pauli pair, in-place
    aBits0 = mask0 & pstr1
    aBits1 = mask1 & pstr1
    bBits0 = mask0 & pstr2
    bBits1 = mask1 & pstr2

    # shift left bits to align with right bits
    aBits1 = aBits1 >> 1
    bBits1 = bBits1 >> 1

    # sets '10' at every Pauli index where individual pairs don't commute
    flags = (aBits0 & bBits1) ⊻ (aBits1 & bBits0)

    # strings commute if parity of non-commuting pairs is even
    return (count_ones(flags) % 2) == 0
end


# XOR between two Pauli different non-identity strings gives the third one. Ignores signs or any coefficient.
_bitpaulimultiply(pstr1::PauliStringType, pstr2::PauliStringType) = pstr1 ⊻ pstr2

# Shift to the right and truncate the first encoded Pauli string. Just a utility function.
_paulishiftright(pstr::PauliStringType) = pstr >> 2

# Computing bit shift index from Pauli site index
# this is is the amount we need to shift to get to the target Pauli
_bitshiftfromsiteindex(siteindex::Integer) = 2 * (siteindex - 1)

# a site is represented by two bits
# using Bits.jl implementation of mask
_paulimask(::Type{T}, n_sites) where T = mask(T, 2 * n_sites)

_pauliwindowmask(::Type{T}, index1::Integer, index2::Integer) where T = _paulimask(T, index2 - index1 + 1) << _bitshiftfromsiteindex(index1)

# This function extracts the Pauli at position `index` from the integer Pauli string.
function _getpaulibits(pstr::PauliStringType, index::Integer)
    return _getpaulibits(pstr, index, index)
end

# This function extracts the Pauli from `index1` to `index2`.
function _getpaulibits(pstr::PauliStringType, index1::Integer, index2::Integer)
    T = typeof(pstr)

    bitindex = _bitshiftfromsiteindex(index1)

    # shift to the right
    shifted_pstr = (pstr >> bitindex)

    # creates all 1s mask of length n bits
    # AND to get the first n bits
    return shifted_pstr & _paulimask(T, index2 - index1 + 1)
end


# This function sets the Pauli at position `index` in the integer Pauli string to `target_pauli`.
function _setpaulibits(pstr::PauliStringType, target_pauli::PauliType, index::Integer)
    return _setpaulibits(pstr, target_pauli, index, index)
end


# This function sets the Pauli from `index1` to `index2` to `target_pstr`.
function _setpaulibits(pstr::PauliStringType, target_pstr::PauliStringType, index1::Integer, index2::Integer)
    T = typeof(pstr)

    bitindex = _bitshiftfromsiteindex(index1)

    window_mask = _pauliwindowmask(T, index1, index2)

    # set bits to target pstr
    return (pstr & ~window_mask) | (T(target_pstr) << bitindex)
end


# This mask helps us to parallelize the bit operations over all qubits.
@generated function alternatingmask(pstr::T) where {T<:PauliStringType}
    # define our super bit mask looking like ....1010101.

    # length is the number of bits in the integer
    n_bits = min(bitsize(pstr), 2_048)  # for max 1024 qubits.
    mask = zero(T)
    for ii in 0:(n_bits-1)
        if ii % 2 == 0
            mask = mask | (T(1) << ii)
        end
    end
    return mask
end