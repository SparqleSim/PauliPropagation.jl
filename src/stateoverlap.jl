### This file contains the functions to calculate the overlap between between backpropagated pauli strings and states or general operators.

#     overlapbyorthogonality(orthogonalfunc::Function, psum::PauliSum)
#     overlapbyorthogonality(orthogonalfunc::Function, pstr::PauliString)
#     overlapbyorthogonality(orthogonalfunc::Function, pstr::Integer)

# Overlap a `PauliSum`, `PauliString` or integer Pauli string with a state or operator via function 
# that returns true if a Pauli string is orthogonal or not.
# If a Pauli string is orthogonal, it does not contribute, otherwise it contributes with its coefficient.
# This is particularly useful for overlaps with stabilizer states.
# An example `orthogonalfunc` is `containsXorY` which returns true if a Pauli string contains an X or Y Pauli.
function overlapbyorthogonality(orthogonalfunc::F, psum) where {F<:Function}
    if length(psum) == 0
        return 0.0
    end

    val = zero(numcoefftype(psum))
    for (pstr, coeff) in zip(paulis(psum), coefficients(psum))
        if overlapbyorthogonality(orthogonalfunc, pstr)
            val += tonumber(coeff)
        end
    end
    return val
end


function overlapbyorthogonality(orthogonalfunc::F, pstr::PauliString) where {F<:Function}
    return !orthogonalfunc(pstr) * tonumber(pstr.coeff)
end


function overlapbyorthogonality(orthogonalfunc::F, pstr::PauliStringType) where {F<:Function}
    return !orthogonalfunc(pstr)
end

## For the typical |0> or |+> cases
"""
    overlapwithzero(psum::PauliSum) 
    overlapwithzero(pstr::PauliString)

Calculates the overlap of a Pauli sum with the zero state |0><0|,
i.e., Tr[psum * |0><0|] = <0|psum|0> or Tr[pstr * |0><0|] = <0|pstr|0>.
"""
overlapwithzero(pobj) = overlapbyorthogonality(containsXorY, pobj)

"""
    overlapwithplus(psum::PauliSum) 
    overlapwithplus(pstr::PauliString)

Calculates the overlap of a Pauli sum or Pauli string with the plus state |+><+|,
i.e. Tr[psum * |+><+|] = <+|psum|+> or Tr[pstr * |+><+|] = <+|pstr|+>.
"""
overlapwithplus(psum) = overlapbyorthogonality(containsYorZ, psum)


# eval against |±i> not implemented


"""
    overlapwithcomputational(psum::PauliSum, onebitinds::Vector{Integer})
    overlapwithcomputational(pstr::PauliString, onebitinds::Vector{Integer})

Calculates the overlap of a Pauli sum or Pauli string with the computational basis state 
which has one-bits at all specified `indices` and zero-bits elsewhere.
If |x><x| is a computational basis state, it we compute Tr[psum * |x><x|] = <x|psum|x> or Tr[pstr * |x><x|] = <x|pstr|x>.
For example, `overlapwithcomputational(psum, [1,2,4])` returns the overlap with `|1101000...>`.
"""
function overlapwithcomputational(psum, onebitinds)
    if length(psum) == 0
        return zero(numcoefftype(psum))
    end

    val = zero(numcoefftype(psum))
    for (pstr, coeff) in zip(paulis(psum), coefficients(psum))
        val += tonumber(coeff) * _calcsignwithones(pstr, onebitinds)
    end
    return val
end


function overlapwithcomputational(pstr::PauliString, onebitinds)
    return _calcsignwithones(pstr.term, onebitinds) * tonumber(pstr.coeff)
end


function _calcsignwithones(pstr::PauliStringType, onebitinds)

    # factor is zero unless pstr is entirely I and Z
    if containsXorY(pstr)
        return 0
    end

    # factor is +-1 per the parity of pstr's Z=3 values at the bit=1 indices
    return (-1)^count(i -> getpauli(pstr, i) == 3, onebitinds)
end

"""
    overlapwithmaxmixed(psum::AbstractPauliSum)

Calculates the overlap of an `AbstractPauliSum` with the maximally mixed state I/2^n,
i.e., Tr[psum * I/2^n].
"""
function overlapwithmaxmixed(psum)
    if length(psum) == 0
        return zero(numcoefftype(psum))
    end

    return getcoeff(psum, zero(paulitype(psum)))
end

"""
    overlapwithpaulisum(rho::PauliSum, psum::PauliSum)

Calculate the overlap of a Pauli sum `psum` and a quantum state `rho` represented in the Pauli basis via another `PauliSum`.
This is equivalent to the trace `Tr[rho * psum]`.
Calls `scalarproduct(rho, psum) * (2^nqubits(rho))` to calculate the overlap.
"""
function overlapwithpaulisum(rho, psum)
    return scalarproduct(rho, psum) * (2^nqubits(rho))
end


"""
    scalarproduct(psum1::AbstractPauliSum, psum2::AbstractPauliSum)
    scalarproduct(pstr::PauliString, psum::AbstractPauliSum)
    scalarproduct(psum::AbstractPauliSum, pstr::PauliString)
    scalarproduct(pstr1::PauliString, pstr2::PauliString)

Calculates the scalar product between any combination of `PauliSum` and `PauliString`.
This  calculates the sum of the products of their coefficients for all Pauli strings that are present .
Important: This is not equivalent to the trace `Tr[psum1 * psum2]` but instead  `Tr[psum1 * psum2]/2^n`,
and equivalently for Pauli strings.
"""
function scalarproduct(psum1::AbstractPauliSum, psum2::AbstractPauliSum)

    longer_psum = psum1
    shorter_psum = psum2

    # swap psums around if the other one is sparser
    if length(longer_psum) < length(shorter_psum)
        longer_psum, shorter_psum = shorter_psum, longer_psum
    end

    # looping over the shorter psum because we are only looking for collisions
    return _scalarproduct(longer_psum, shorter_psum)

end

function _scalarproduct(lookup_psum, loop_psum)

    _checknumberofqubits(lookup_psum, loop_psum)

    CType = promote_type(numcoefftype(lookup_psum), numcoefftype(loop_psum))

    val = float(zero(CType))

    if length(lookup_psum) == 0 || length(loop_psum) == 0
        return val
    end

    # looping over the iter psum because we are only looking for collisions
    for (pstr, coeff) in zip(paulis(loop_psum), coefficients(loop_psum))
        val += tonumber(getcoeff(lookup_psum, pstr)) * tonumber(coeff)
    end
    return val

end


function scalarproduct(pstr::PauliString, psum::AbstractPauliSum)
    _checknumberofqubits(pstr, psum)
    return tonumber(getcoeff(psum, pstr)) * tonumber(pstr.coeff)

end

scalarproduct(psum::AbstractPauliSum, pstr::PauliString) = scalarproduct(pstr, psum)



function scalarproduct(pstr1::PauliString, pstr2::PauliString)
    _checknumberofqubits(pstr1, pstr2)

    if pstr1.term != pstr2.term
        return zero(numcoefftype(pstr1))
    end

    # at this point the two Pauli strings are the same
    # so we can just multiply the coefficients
    return tonumber(pstr1.coeff) * tonumber(pstr2.coeff)
end



# returns a new filtered dictionary, but doesn't overlap with anything
"""
    zerofilter(psum)

Return a filtered Pauli sum with only Pauli strings that are not orthogonal to the zero state |0><0|.
"""
zerofilter(psum) = truncate!((pstr, coeff) -> containsXorY(pstr), deepcopy(psum))

"""
    zerofilter!(psum)

Filter a Pauli sum in-place with only Pauli strings that are not orthogonal to the zero state |0><0|.
"""
zerofilter!(psum) = truncate!((pstr, coeff) -> containsXorY(pstr), psum)

"""
    plusfilter(psum)

Return a filtered Pauli sum with only Pauli strings that are not orthogonal to the plus state |+><+|.
"""
plusfilter(psum) = truncate!((pstr, coeff) -> containsYorZ(pstr), deepcopy(psum))

"""
    zerofilter!(psum)

Filter a Pauli sum in-place with only Pauli strings that are not orthogonal to the plus state |+><+|.
"""
plusfilter!(psum) = truncate!((pstr, coeff) -> containsYorZ(pstr), psum)




