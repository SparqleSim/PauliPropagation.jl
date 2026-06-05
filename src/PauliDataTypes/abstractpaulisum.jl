"""
    AbstractPauliSum <: AbstractTermSum

Abstract type for objects represented sums of Paulis with coefficients.
"""
abstract type AbstractPauliSum <: AbstractTermSum end

nqubits(psum::AbstractPauliSum) = throw(ErrorException("nqubits() not implemented for type $(typeof(psum))."))
PropagationBase.nsites(psum::AbstractPauliSum) = nqubits(psum)

"""
    paulis(psum::AbstractPauliSum)

Returns an iterator over the integer pauli strings of an `AbstractPauliSum`.
Call `topaulistrings` to receive entries as `PauliString`s.
"""
paulis(psum::AbstractPauliSum) = terms(psum)

"""
    coefficients(psum::AbstractPauliSum)

Returns an iterator over the coefficients of a `PauliSum`.
Call `topaulistrings` to receive entries as `PauliString`s.
"""
PropagationBase.coefficients

"""
    paulitype(psum::AbstractPauliSum)

Get the Pauli integer type of a `AbstractPauliSum` object.
"""
paulitype(psum::AbstractPauliSum) = eltype(paulis(psum))


"""
    topaulistrings(psum::AbstractPauliSum)

Returns the Pauli strings in a, `AbstractPauliSum` and their coefficients as a list of `PauliString`.
"""
topaulistrings(psum::AbstractPauliSum) = [PauliString(psum.nqubits, pauli, coeff) for (pauli, coeff) in zip(paulis(psum), coefficients(psum))]


## The symbol conversions for getcoeff()

"""
    getcoeff(psum::AbstractPauliSum, pauli::Symbol, qind::Integer)

Get the coefficient of a Pauli string in an `AbstractPauliSum` by providing the Pauli string as a Symbol acting on qubit `qind`. 
This is consistent with how Pauli strings can be added to a `PauliSum` via `add!()`. 
Defaults to 0 if the Pauli string is not in the `AbstractPauliSum`.
"""
function PropagationBase.getcoeff(psum::AbstractPauliSum, pauli::Symbol, qind::Integer)
    return getcoeff(psum, symboltoint(psum.nqubits, pauli, qind))
end

"""
    getcoeff(psum::AbstractPauliSum, pstr::Vector{Symbol}, qinds::Vector{Int})

Get the coefficient of a Pauli string in an `AbstractPauliSum` by providing the Pauli string `pstr` as a vector of Symbols acting on qubits `qinds`. 
This is consistent with how Pauli strings can be added to a `PauliSum` via `add!()`. 
Defaults to 0 if the Pauli string is not in the `AbstractPauliSum`.
"""
function PropagationBase.getcoeff(psum::AbstractPauliSum, pstr, qinds)
    return getcoeff(psum, symboltoint(psum.nqubits, pstr, qinds))
end

"""
    getcoeff(psum::AbstractPauliSum, pstr::Vector{Symbol})

Get the coefficient of a Pauli string in a `AbstractPauliSum` by providing the Pauli string `pstr` as a vector of Symbols acting on all qubits. 
This is consistent with how Pauli strings can be added to a `PauliSum` via `add!()`. 
Defaults to 0 if the Pauli string is not in the `AbstractPauliSum`.
"""
function PropagationBase.getcoeff(psum::AbstractPauliSum, pstr::Vector{Symbol})
    return getcoeff(psum, symboltoint(pstr))
end

"""
    getcoeff(psum::AbstractPauliSum, pstr::Integer)

Get the coefficient of a `PauliString` in a `PauliSum`. Defaults to 0 if the Pauli string is not in the `PauliSum`.
Requires that the integer Pauli string in `pstr` is the same type as the integer Pauli strings in `psum`.
"""
function PropagationBase.getcoeff(psum::AbstractPauliSum, pstr::PauliString)
    return getcoeff(psum, pstr.term)
end


### Adding symbols or PauliStrings
# The lower-level add! functions are defined in PropagationBase.jl
"""
    add!(psum::AbstractPauliSum, pauli::Symbol, qind::Integer, coeff=1.0)
    add!(psum::AbstractPauliSum, paulis::Vector{Symbol}, qinds::Vector{Integer}, coeff=1.0)

Add a Pauli string to a `AbstractPauliSum` `psum`. Changes `psum` in-place.
Provide the Pauli string as a `Symbol` (:I, :X, :Y, :Z) or `Vector{Symbol}`.
Provide the index or indices for those symbols as `qind` or `qinds`.
The coefficient of the Pauli string in the Pauli sum defaults to 1.0.
"""
function PropagationBase.add!(psum::AbstractPauliSum, paulis::Union{Symbol,Vector{Symbol}}, qinds, coeff=coefftype(psum)(1.0))
    return add!(psum, PauliString(psum.nqubits, paulis, qinds, coeff))
end

"""
    add!(psum::AbstractPauliSum, pstr::PauliString)

Add a `PauliString` `pstr` to a `PauliSum` `psum`. Changes `psum` in-place.
`psum` and `pstr` need to be defined on the same number of qubits and have the same coefficient type.
"""
function PropagationBase.add!(psum::AbstractPauliSum, pstr::PauliString)
    _checknumberofqubits(psum, pstr)

    # this is supposed to error if pstr.coeff cannot be converted to CT1
    # because this is an in-place operation
    pstr_coeff = convert(coefftype(psum), pstr.coeff)
    add!(psum, pstr.term, pstr_coeff)
    return psum
end

"""
    add!(psum1::AbstractPauliSum, psum2::AbstractPauliSum)

Add two `AbstractPauliSum`s `psum1` and `psum2`. Changes `psum1` in-place.
`psum1` and `psum2` need to be defined on the same number of qubits and have the same coefficient type.
"""
PropagationBase.add!(::AbstractPauliSum, ::AbstractPauliSum)


"""
    add!(psum::AbstractPauliSum, pstr, coeff)

Add a Pauli string `pstr` with coefficient `coeff` to a `AbstractPauliSum` `psum`. This changes `psum` in-place.
`pstr` needs to have the same type as `paulitype(psum)`, and `coeff` needs to have the same type as `coefftype(psum)`.
"""
PropagationBase.add!(::AbstractPauliSum, ::Any, ::Any)


"""
    +(pstr1::PauliString, pstr2::PauliString)

Addition of two `PauliString`s. Returns a PauliSum.
"""
function Base.:+(pstr1::PauliString, pstr2::PauliString)

    nq = _checknumberofqubits(pstr1, pstr2)

    # get a compatibel coefficient type
    CType = promote_type(coefftype(pstr1), coefftype(pstr2))
    psum = PauliSum(CType, nq)
    add!(psum, pstr1)
    add!(psum, pstr2)
    return psum
end

"""
    +(pstr::PauliString, psum::AbstractPauliSum)
    +(psum::AbstractPauliSum, pstr::PauliString)

Addition of a `PauliString` to a `PauliSum`. Returns a `PauliSum`.
"""
function Base.:+(psum::PS, pstr::PauliString) where PS<:AbstractPauliSum
    nq = _checknumberofqubits(psum, pstr)

    # get a compatible coefficient type
    CType = promote_type(paulitype(psum), coefftype(pstr))
    PlainPS = Base.typename(PS).wrapper
    new_psum = PlainPS(CType, nq)

    add!(new_psum, psum)
    add!(new_psum, pstr)
    return new_psum
end

Base.:+(pstr::PauliString, psum::AbstractPauliSum) = psum + pstr


### Subtraction

"""
    -(pstr1::PauliString, pstr2::PauliString)

Subtract two `PauliString`s. Returns a PauliSum.
"""
function Base.:-(pstr1::PauliString, pstr2::PauliString)
    return pstr1 + (-1 * pstr2)
end

"""
    -(pstr::PauliString, psum::AbstractPauliSum)
    -(psum::AbstractPauliSum, pstr::PauliString)

Subtract a `PauliString` from a `PauliSum` or vice versa.
Returns a `PauliSum`.
"""
function Base.:-(psum::AbstractPauliSum, pstr::PauliString)
    return psum + (-1 * pstr)
end

Base.:-(pstr::PauliString, psum::AbstractPauliSum) = mult!(psum - pstr, -1)


## Set in Pauli sum
"""
    set!(psum::AbstractPauliSum, pstr, coeff)

In-place setting the coefficient of a Pauli string in an `AbstractPauliSum`.
The type of the Pauli string needs to be the typeof(pstr)==paulitype(psum) and `typeof(coeff)==coefftype(psum)`.
"""
PropagationBase.set!


### TODO: general products between AbstractPauliSums and PauliStrings
# TODO: in-place pauliprod()



"""
    filter!(filterfunc::Function, psum::AbstractPauliSum)

Filter a `AbstractPauliSum` by copying and removing all Pauli strings for which `filterfunc(pstr, coeff)` returns `false`.
"""
Base.filter(filterfunc::F, psum::AbstractPauliSum) where {F<:Function} = truncate!((pstr, coeff) -> !filterfunc(pstr, coeff), deepcopy(psum))

"""
    filter!(filterfunc::Function, psum::AbstractPauliSum)

Filter a `AbstractPauliSum` in-place by removing all Pauli strings for which `filterfunc(pstr, coeff)` returns `false`.
"""
filter!(filterfunc::F, psum::AbstractPauliSum) where {F<:Function} = truncate!((pstr, coeff) -> !filterfunc(pstr, coeff), psum)