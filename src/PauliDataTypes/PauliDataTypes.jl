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

This is a Union that includes:
- Standard Julia `Integer` subtypes (UInt8 … UInt64, BitIntegers types)
- `NTuplePauliString{N,W}` for GPU-compatible wide Pauli strings
"""
const PauliStringType = Union{Integer, NTuplePauliString}

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