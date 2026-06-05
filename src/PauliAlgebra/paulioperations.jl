### paulioperations.jl
## 
# This file contains function definitions for Pauli operations such as
# commutation and multiplication.
##
###

# LinearAlgebra dispatches

"""
    LinearAlgebra.tr(pstr::PauliString)

Compute the trace of a `PauliString` operator.

The trace of any non-identity Pauli operator (e.g., X, Y, Z, or their tensor products) is zero.
The identity operator `I` has a trace equal to the dimension of the Hilbert space, `2^N`,
where `N` is the number of qubits.

A `PauliString` represents a single term, typically of the form `coeff * P_1 P_2 ... P_N`.
If the `pstr.term` bitmask is `0x00`, it signifies the identity operator across all
`pstr.nqubits`. In this case, the trace is `pstr.coeff * 2^pstr.nqubits`. For any other
`pstr.term` value (representing a non-identity Pauli operator), the trace is `0.0`.

# Arguments
- `pstr::PauliString`: The Pauli string to trace. 

# Returns
- `CT`: The trace value of the `PauliString`.
"""
function LinearAlgebra.tr(pstr::PauliString{TT,CT}) where {TT,CT}
    pstr.term == zero(TT) ? pstr.coeff * CT(2.0^pstr.nqubits) : zero(CT)
end

"""
    LinearAlgebra.tr(psum::AbstractPauliSum)

Compute the trace of an `AbstractPauliSum` operator.

The trace is a linear operation: Tr(A + B) = Tr(A) + Tr(B). Since individual non-identity
PauliString terms have a trace of zero (as per tr(::PauliString)), only the coefficient
of the identity operator contributes to the total trace of a PauliSum.

The coefficient of the identity term (0x00) is retrieved from this mapping using get(psum.terms, 0x00, 0). 
This coefficient is then multiplied by 2^nqubits(psum) (the dimension of the Hilbert space). 
If the identity term (0x00) is not explicitly present in psum.terms, its coefficient is implicitly zero,
resulting in a total trace of 0.0.

# Arguments
`psum::PauliSum`: The Pauli sum to trace.

# Returns
`CT`: The trace value of the PauliSum.
"""
function LinearAlgebra.tr(psum::AbstractPauliSum)
    return getcoeff(psum, zero(paulitype(psum))) * convert(coefftype(psum), 2^nqubits(psum))
end

"""
    trace(pstr::PauliString)

Wrapper for `LinearAlgebra.tr(pstr::PauliString)`.

# Arguments
- `pstr::PauliString`: The Pauli string to trace. 

# Returns
- `CT`: The trace value of the `PauliString`.
"""
function trace(pstr::PauliString)
    return LinearAlgebra.tr(pstr)
end

"""
    trace(psum::PauliSum)

Wrapper for `LinearAlgebra.tr(psum::PauliSum)`.

# Arguments
`psum::PauliSum`: The Pauli sum to trace.

# Returns
`CT`: The trace value of the PauliSum.
"""
function trace(psum::AbstractPauliSum)
    return LinearAlgebra.tr(psum)
end

# TODO: generate these definitions with Macro's instead? Easier to maintain and less error-prone

"""
    countweight(pstr::PauliString)

Function to count the weight of a `PauliString`.
"""
function countweight(pstr::PauliString)
    return countweight(pstr.term)
end

"""
    countweight(pstr::Integer)

Function to count the weight of an integer Pauli string.
"""
function countweight(pstr::PauliStringType)
    return _countbitweight(pstr)
end

"""
    countweight(psum::PauliSum)

Function to count the weight Pauli strings in a `PauliSum`. Returns an array of weights.
"""
function countweight(psum::PauliSum)
    return countweight(psum.terms)
end

function countweight(psum::AbstractDict)
    return [countweight(pstr) for pstr in keys(psum)]
end

"""
    countxy(pstr::PauliString)

Function to count the number of X and Y Paulis in a `PauliString`.
"""
function countxy(pstr::PauliString)
    return countxy(pstr.term)
end

"""
    countxy(pstr::Integer)

Function to count the number of X and Y Paulis in an integer Pauli string.
"""
function countxy(pstr::PauliStringType)
    return _countbitxy(pstr)
end

"""
    countxy(psum::PauliSum)

Function to count the number of X and Y Paulis in a `PauliSum`. Returns an array of counts.
"""
function countxy(psum::PauliSum)
    return countxy(psum.terms)
end

function countxy(psum::AbstractDict)
    return [countxy(pstr) for pstr in keys(psum)]
end

"""
    countyz(pstr::PauliString)

Function to count the number of Y and Z Paulis in a `PauliString`.
"""
function countyz(pstr::PauliString)
    return countyz(pstr.term)
end

"""
    countyz(pstr::Integer)

Function to count the number of Y and Z Paulis in an integer Pauli string.
"""
function countyz(pstr::PauliStringType)
    return _countbityz(pstr)
end

"""
    countyz(psum::PauliSum)

Function to count the number of Y and Z Paulis in a `PauliSum`. Returns an array of counts.
"""
function countyz(psum::PauliSum)
    return countyz(psum.terms)
end

function countyz(psum::AbstractDict)
    return [countyz(pstr) for pstr in keys(psum)]
end

"""
    containsXorY(pstr::PauliString)

Check if a Pauli string contains an X or Y Pauli.
"""
containsXorY(pstr::PauliString) = containsXorY(pstr.term)

"""
    containsXorY(pstr::Integer)

Check if an integer Pauli string contains an X or Y Pauli.
"""
containsXorY(pstr::PauliStringType) = countxy(pstr) > 0

"""
    containsXorY(pstr::PauliString)

Check if a Pauli string contains a Y or Z Pauli.
"""
containsYorZ(pstr::PauliString) = containsYorZ(pstr.term)

"""
    containsYorZ(pstr::Integer)

Check if an integer Pauli string contains a Y or Z Pauli.
"""
containsYorZ(pstr::PauliStringType) = countyz(pstr) > 0


### Counting functions for integer Pauli strings ###

"""
    countx(pstr::PauliString)

Function to count the number of X Paulis in a `PauliString`.
"""
function countx(pstr::PauliString)
    return countx(pstr.term)
end

"""
    countx(pstr::Integer)

Function to count the number of X Paulis in an integer Pauli string.
"""
function countx(pstr::PauliStringType)
    return _countbitx(pstr)
end

"""
    countx(psum::PauliSum)
Function to count the number of X Paulis in a `PauliSum`. Returns an array of counts.
"""
function countx(psum::PauliSum)
    return countx(psum.terms)
end

function countx(psum::AbstractDict)
    return [countx(pstr) for pstr in keys(psum)]
end


"""
    county(pstr::PauliString)

Function to count the number of Y Paulis in a `PauliString`.
"""
function county(pstr::PauliString)
    return county(pstr.term)
end

"""
    county(pstr::Integer)

Function to count the number of Y Paulis in an integer Pauli string.
"""
function county(pstr::PauliStringType)
    return _countbity(pstr)
end

"""
    county(psum::PauliSum)
Function to count the number of Y Paulis in a `PauliSum`. Returns an array of counts.
"""
function county(psum::PauliSum)
    return county(psum.terms)
end

function county(psum::AbstractDict)
    return [county(pstr) for pstr in keys(psum)]
end


"""
    countz(pstr::PauliString)

Function to count the number of Z Paulis in a `PauliString`.
"""
function countz(pstr::PauliString)
    return countz(pstr.term)
end

"""
    countz(pstr::Integer)

Function to count the number of Z Paulis in an integer Pauli string.
"""
function countz(pstr::PauliStringType)
    return _countbitz(pstr)
end

"""
    countz(psum::PauliSum)
    
Function to count the number of Z Paulis in a `PauliSum`. Returns an array of counts.
"""
function countz(psum::PauliSum)
    return countz(psum.terms)
end

function countz(psum::AbstractDict)
    return [countz(pstr) for pstr in keys(psum)]
end


### All the commutation check functions
"""
    commutes(pstr1::PauliString, pstr2::PauliString)

Check if two Pauli strings of type `PauliString` commute.
"""
function commutes(pstr1::PauliString, pstr2::PauliString)
    return commutes(pstr1.term, pstr2.term)
end

"""
    commutes(pstr1::Integer, pstr2::Integer)

Check if two integer Pauli strings commute.
"""
function commutes(pstr1::PauliStringType, pstr2::PauliStringType)
    return _bitcommutes(pstr1, pstr2)
end

"""
    commutes(psum1::PauliSum, psum2::PauliSum)

Check if two Pauli sums of type `PauliSum` commute.
"""
function commutes(psum1::PauliSum, psum2::PauliSum)
    comm = commutator(psum1.terms, psum2.terms)
    return isempty(comm)
end

function commutes(psum1::Dict{TT,CT1}, psum2::Dict{TT,CT2}) where {TT,CT1,CT2}
    comm = commutator(psum1, psum2)
    return isempty(comm)
end


## Commutator
# TODO: implement commutator and pauliprod for VectorPauliSum
"""
    commutator(psum1::PauliSum, psum2::PauliSum)
    
Calculate the commutator of two `PauliSum`s.
"""
function commutator(psum1::PauliSum, psum2::PauliSum)
    new_pstr_dict = commutator(psum1.terms, psum2.terms)
    return PauliSum(psum1.nqubits, new_pstr_dict)
end

"""
    commutator(pstr1::PauliString, pstr2::PauliString)

Calculate the commutator of two `PauliString`s.
"""
function commutator(pstr1::PauliString, pstr2::PauliString)
    new_pstr, new_coeff = commutator(pstr1.term, pstr2.term)
    return PauliString(pstr1.nqubits, new_pstr, new_coeff)
end

# TODO: specialize for the commutator between PauliSum and PauliString
# this one can then be used in the general commutator function between PauliSum and PauliSum
"""
    commutator(psum::PauliSum, pstr::PauliString)
    commutator(pstr::PauliString, psum::PauliSum)

Calculate the commutator of a `PauliSum` and a `PauliString`.
"""
commutator(psum::PauliSum, pstr::PauliString) = commutator(psum, PauliSum(pstr))
commutator(pstr::PauliString, psum::PauliSum) = commutator(PauliSum(pstr), psum)


"""
    commutator(vpsum1::VectorPauliSum, vpsum2::VectorPauliSum)

Calculate the commutator of two `VectorPauliSum`s.
Returns a `VectorPauliSum` with complex coefficients.

# Example
```julia
vpsum = VectorPauliSum(3, [1, 2], [1.]) # 1.0 * X + 1.0 * Y
vpsum2 = VectorPauliSum(3, [7], [0.25]) # 0.25 * ZX
commutator(vpsum, vpsum2) # should return a VectorPauliSum with - 0.5 YX + 0.5 XX
```
"""
function PauliPropagation.commutator(vpsum1::VectorPauliSum, vpsum2::VectorPauliSum)
    nq = _checknumberofqubits(vpsum1, vpsum2)

    TT = paulitype(vpsum1)
    CT = promote_type(ComplexF64, coefftype(vpsum1), coefftype(vpsum2))
    new_pauli_dict = Dict{TT,CT}()

    for (pauli1, coeff1) in zip(vpsum1.terms, vpsum1.coeffs), (pauli2, coeff2) in zip(vpsum2.terms, vpsum2.coeffs)
        if !commutes(pauli1, pauli2)
            new_pstr, sign = commutator(pauli1, pauli2)
            new_pauli_dict[new_pstr] = get(new_pauli_dict, new_pstr, zero(CT)) + convert(CT, sign * coeff1 * coeff2)
        end
    end

    return VectorPauliSum(nq, collect(keys(new_pauli_dict)), collect(values(new_pauli_dict)))
end

"""
    commutator(pstr1::Integer, pstr2::Integer)

Calculate the commutator of two integer Pauli strings.
Returns a tuple of the coefficient and the potentially integer Pauli string.
The coefficient is zero if the Pauli strings commute.
"""
function commutator(pstr1::PauliStringType, pstr2::PauliStringType)

    if commutes(pstr1, pstr2)
        total_sign = ComplexF64(0.0)
        new_pstr = identitylike(pstr1)
    else
        new_pstr, total_sign = pauliprod(pstr1, pstr2)
    end
    # commutator is [A, B] = AB - BA = 2AB for non-commuting (meaning anti-commuting) Paulis
    return new_pstr, 2. * total_sign
end


# TODO: modernize commutator
function commutator(psum1::Dict{TT,CT1}, psum2::Dict{TT,CT2}) where {TT,CT1,CT2}
    # different types of coefficients are allowed but not different types of Pauli strings

    new_pauli_dict = Dict{TT,ComplexF64}()

    for (pauli1, coeff1) in psum1, (pauli2, coeff2) in psum2
        if !commutes(pauli1, pauli2)
            new_pstr, sign = commutator(pauli1, pauli2)
            new_pauli_dict[new_pstr] = get(new_pauli_dict, new_pstr, ComplexF64(0.0)) + sign * coeff1 * coeff2
        end
    end

    # Get rid of the pauli strings with zero coeffs
    # TODO: possibly combine this with the loop above
    for (k, v) in new_pauli_dict
        if abs(v) ≈ 0.0
            delete!(new_pauli_dict, k)
        end
    end

    return new_pauli_dict
end

## Pauli product

"""
    pauliprod(pstr1::PauliString, pstr2::PauliString)

Calculate the product of two `PauliString`s. 

# Examples
```
julia> pauliprod(PauliString(1, [:X], [1]), PauliString(1, [:Y], [1])) # X*Y=iZ
```
"""
function pauliprod(pstr1::PauliString, pstr2::PauliString)
    _checktermtype(pstr1, pstr2)
    _checknumberofqubits(pstr1, pstr2)
    new_pstr, sign = pauliprod(pstr1.term, pstr2.term)
    return PauliString(pstr1.nqubits, new_pstr, sign * pstr1.coeff * pstr2.coeff)
end


## Pauli product for PauliSums and PauliStrings
"""
    pauliprod(psum1::PauliSum, psum2::PauliSum)

Calculate the product of two `PauliSum`s. 
Default returns a `PauliSum{TT, ComplexF64}` where `TT` is the type of the new Pauli Strings.

# Examples
```julia
psum = PauliSum(PauliString(3, [:Y], [2])) 
psum_identity = PauliSum(PauliString(3, [:I], [1]))
pauliprod(psum, psum_identity) # Psum * I = Psum
```

"""
function pauliprod(psum1::PauliSum, psum2::PauliSum)

    _checktermtype(psum1, psum2)
    nq = _checknumberofqubits(psum1, psum2)

    psum = PauliSum(ComplexF64, nq)
    sizehint!(psum, length(psum1))

    for (pstr1, coeff1) in psum1
        for (pstr2, coeff2) in psum2
            pstr, sign = pauliprod(pstr1, pstr2)
            add!(psum, pstr, coeff1 * coeff2 * sign)
        end
    end
    return psum

end

"""
    pauliprod(pstr1::Integer, pstr2::Integer)

Calculate the product of two integer Pauli strings.
"""
function pauliprod(pstr1::PauliStringType, pstr2::PauliStringType)
    # This function is for when we need to globally check the sign of the product (like in general products of Paulis, not local Pauli gates)
    pstr3 = _bitpaulimultiply(pstr1, pstr2)
    sign = _impow(_calculatesignexponent(pstr1, pstr2))
    return pstr3, sign
end

# Calculate the sign of the product of two integer Pauli strings. Outcomes are either ±1 or ±i.
function _calculatesignexponent(pauli1::PauliType, pauli2::PauliType)
    # get left and right bits of a pauli    
    mask_right = alternatingmask(pauli1)

    pauli1_1 = (pauli1 >> 1) & mask_right
    pauli1_2 = pauli1 & mask_right
    pauli2_1 = (pauli2 >> 1) & mask_right
    pauli2_2 = pauli2 & mask_right

    #make sure neither pauli is the identity
    not_identity_pauli1 = pauli1_1 | pauli1_2
    not_identity_pauli2 = pauli2_1 | pauli2_2

    #make sure paulis aren't the same
    not_same = (pauli1_1 ⊻ pauli2_1) | (pauli1_2 ⊻ pauli2_2)

    #determine if the paulis commute (should return 0 if they do)
    not_commuting = not_identity_pauli1 & not_identity_pauli2 & not_same

    #use not_commuting as a mask to get the "don't cares" in the karnaugh map to always be 0
    #while the right side of the next line came from karnaugh map of the truth table for
    #the levi cevita symbol
    #    |00|01|11|10
    # 00 |--|--|--|--|
    # 01 |--|--|T |F |
    # 11 |--|F |--|T |
    # 10 |--|T |F |--|
    # which gives (a & ~d) | (~a & d) | (~b & ~c)
    # you can then recognize that (a & ~d) | (~a & d) = a ⊻ b
    # where pauli1 = ab, and pauli2 = cd.
    negative_sign = not_commuting & ((pauli1_1 ⊻ pauli2_2)
                                     |
                                     (~pauli1_2 & ~pauli2_1))
    positive_sign = not_commuting & (~negative_sign)
    # You can use modular addition to achieve addition of the exponent,
    # since it is cyclic, and the global phase can be determined by the number
    # of 1's in each expression.
    # i.e. -im = im^(3); im = im^(1); 1 = im^0.  
    return ((3 * count_ones(negative_sign) + count_ones(positive_sign)) % 4)
end

#speeds up pauliprod by a factor of 2 since we know we only want integer powers
const impowers = [1, im, -1, -im]
function _impow(power::Integer)
    ind = (power % 4) + 1
    return impowers[ind]
end
