

"""
    PauliSum(nqubits::Int, terms::Dict) <: AbstractPauliSum

`PauliSum` is a `struct` that represents a sum of Pauli strings acting on `nqubits` qubits.
It is a wrapper around a dictionary Dict(Pauli string => coefficient}, where the Pauli strings are typically unsigned Integers for efficiency reasons.
"""
struct PauliSum{TT,CT} <: AbstractPauliSum
    nqubits::Int
    terms::Dict{TT,CT}

    function PauliSum(nq::Int, terms::Dict{TT,CT}) where {TT,CT}
        if nq <= 0
            throw(ArgumentError("Number of qubits must be a positive integer."))
        end

        return new{TT,CT}(nq, terms)
    end
end

"""
    PauliSum(nqubits::Integer)

Contructor for an empty `PauliSum` on `nqubits` qubits. Element type defaults for Float64.
"""
PauliSum(nqubits::Int) = PauliSum(Float64, nqubits)

"""
    PauliSum(CoeffType, nq::Int)

Contructor for an empty `PauliSum` on `nqubits` qubits. The type of the coefficients can be provided.
"""
function PauliSum(::Type{CT}, nq::Int) where {CT}
    TT = getinttype(nq)
    return PauliSum(nq, Dict{TT,CT}())
end

PropagationBase.storage(psum::PauliSum) = psum.terms

"""
    nqubits(psum::PauliSum)

Get the number of qubits that the `PauliSum` is defined on.
"""
nqubits(psum::PauliSum) = psum.nqubits


"""
Copy a `PauliSum` by copying its `terms` field.
"""
Base.copy(psum::PauliSum) = PauliSum(psum.nqubits, copy(psum.terms))


### Products
# TODO: enable for Vector PauliSum


#     *(pstr::PauliString, psum::PauliSum)
#     *(psum::PauliSum, pstr::PauliString)
# Perform a Pauli product of a `PauliString` with a `PauliSum`.
# Returns a `PauliSum` with complex coefficients.
function Base.:*(psum::PauliSum, pstr::PauliString)

    psum2 = PauliSum(pstr)
    return pauliprod(psum, psum2)
end

function Base.:*(pstr::PauliString, psum::PauliSum)

    psum2 = PauliSum(pstr)
    return pauliprod(psum2, psum)
end



#     *(psum1::PauliSum, psum2::PauliSum)
# Perform a Pauli product of two `PauliSum`s.
# Returns a `PauliSum` with complex coefficients.
function Base.:*(psum1::PauliSum, psum2::PauliSum)

    return pauliprod(psum1, psum2)
end


# Pretty print for `PauliSum`.
function Base.show(io::IO, psum::PauliSum)
    if length(psum.terms) == 0
        dict_string = "(no Pauli strings)"
    else
        dict_string = _getprettystr(psum.terms, psum.nqubits)
    end
    print(io, "PauliSum(nqubits: $(psum.nqubits), $dict_string)")
end



#     sizehint!(psum::PauliSum, n)
# Hint to the `PauliSum` to reserve space for `n` terms.
Base.sizehint!(psum::PauliSum, n) = sizehint!(psum.terms, n)



#     similar(psum::PauliSum)
# Create a new `PauliSum` with the same number of qubits and coefficient type as `psum`.
# Calls `sizehint!()` with `length(psum)` on the dictionary of the new `PauliSum`. 
function Base.similar(psum::PauliSum)
    return PauliSum(psum.nqubits, similar(psum.terms))
end

function Base.similar(psum::Dict{TT,CT}) where {TT,CT}
    new_psum = Dict{TT,CT}()
    sizehint!(new_psum, length(psum))
    return new_psum
end

