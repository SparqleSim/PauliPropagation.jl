"""
    PauliString(nqubits::Int, term::TermType, coeff::CoeffType)

`PauliString` is a `struct` that represents a Pauli string on `nqubits` qubits.
Commonly `term` is an unsigned Integer. 
See the other `PauliString` constructors for higher-level usage. 
"""
struct PauliString{TT<:PauliStringType,CT}
    nqubits::Int
    term::TT
    coeff::CT

    function PauliString(nqubits::Int, term::TT, coeff::CT) where {TT<:PauliStringType,CT}
        if nqubits <= 0
            throw(ArgumentError("Number of qubits must be a positive integer."))
        end

        return new{TT,CT}(nqubits, term, coeff)
    end
end

"""
    PauliString(nqubits::Int, pauli::Symbol, qind::Integer, coeff=1.0)
    PauliString(nqubits::Int, paulis::Vector{Symbol}, qinds::Vector{Integer}, coeff=1.0)

Constructor for a `PauliString` on `nqubits` qubits from a `Symbol` (:I, :X, :Y, :Z) or `Vector{Symbol}`.
Provide the index or indices for those symbols as `qind` or `qinds`.
The coefficient of the Pauli string in the Pauli sum defaults to 1.0.
"""
function PauliString(nqubits::Int, paulis, qinds, coeff=1.0)

    # `symboltoint()` also does checks, but we want to catch them here.
    term = symboltoint(nqubits, paulis, qinds)

    # In case an integer is passed, we probably want it to be a float.
    coeff = float(coeff)

    return PauliString(nqubits, term, coeff)
end


"""
    nqubits(pstr::PauliString)
    
Get the number of qubits that the `PauliString` is defined on.
"""
nqubits(pstr::PauliString) = pstr.nqubits

"""
    paulitype(pstr::PauliString)

Get the Pauli integer type of a `PauliString`.
"""
function paulitype(pstr::PauliString)
    return typeof(pstr.term)
end

"""
    coefftype(pstr::PauliString)

Get the coefficient type of a `PauliString`.
"""
function PropagationBase.coefftype(pstr::PauliString)
    return typeof(pstr.coeff)
end

"""
    numcoefftype(pstr::PauliString)

Get the type of the numerical coefficient of a `PauliString`. 
Will return the type of the output of  `numcoefftype(coefftype(pstr))`.
"""
function PropagationBase.numcoefftype(pstr::PauliString{TT,CT}) where {TT,CT}
    return numcoefftype(CT)
end



#     *(pstr::PauliString, c::Number)
# Multiply a `PauliString` by a scalar `c`. Returns a new `PauliString`.
*(pstr::PauliString, c::Number) = PauliString(pstr.nqubits, pstr.term, pstr.coeff * c)
*(c::Number, pstr::PauliString) = pstr * c


#     *(pstr1::PauliString, pstr2::PauliString)
# Perform a Pauli product of two `PauliString`s. 
function *(pstr1::PauliString, pstr2::PauliString)

    return pauliprod(pstr1, pstr2)
end

#     /(pstr::PauliString, c::Number)
# Divide a `PauliString` by a scalar `c`. Returns a new `PauliString`.
/(pstr::PauliString, c::Number) = pstr * (1 / c)


# Pretty print for `PauliString`.
function Base.show(io::IO, pstr::PauliString)
    pauli_string = inttostring(pstr.term, pstr.nqubits)
    if length(pauli_string) > 20
        pauli_string = pauli_string[1:20] * "..."
    end
    if isa(pstr.coeff, Number)
        coeff_str = round(pstr.coeff, sigdigits=5)
    elseif isa(pstr.coeff, PathProperties) && hasfield(typeof(pstr.coeff), :coeff)
        PProp = string(typeof(pstr.coeff).name.name)
        if isa(pstr.coeff.coeff, Number)
            coeff_str = "$PProp($(round(pstr.coeff.coeff, sigdigits=5)))"
        else
            coeff_str = "$PProp($(typeof(pstr.coeff.coeff)))"
        end
    else
        coeff_str = "$(typeof(pstr.coeff))"
    end
    print(io, "PauliString(nqubits: $(pstr.nqubits), $(coeff_str) * $(pauli_string))")
end
