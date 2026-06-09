using Bits
using BitIntegers
import Base: *
import Base: /
import Base: +
import Base: -
import Base: ==

"""
    PauliStringType

The integer types we use to represent Pauli strings. 
Pauli strings are objects like X ⊗ Z ⊗ I ⊗ Y, where each term is a Pauli acting on a qubit.
"""
const PauliStringType = Integer

"""
    PauliType

A union type for the integer types used to represent Paulis.
Paulis, also known as Pauli operators, are objects like I, X, Y, Z acting on a single qubit.
"""
const PauliType = PauliStringType

include("paulistring.jl")
include("abstractpaulisum.jl")
include("paulisum.jl")
include("vectorpaulisum.jl")
include("conversions.jl")

function _check_qind_range(nq, qind::Integer)
    if qind < 1 || qind > nq
        throw(ArgumentError("Index must be between 1 and $nq. Got $qind."))
    end
    return 
end

function _check_qind_range(nq, qinds)
    if any(qind -> qind < 1 || qind > nq, qinds)
        throw(ArgumentError("Indices must be between 1 and $nq. Got $qinds."))
    end
    return 
end