###
##
# A file to define a Pauli sum consisting of a vector of terms and a vector of coefficients.
# Can be used for multithreaded CPU and GPU propagation.
##
###

# This type multi-threads where possible.
using AcceleratedKernels
const AK = AcceleratedKernels

const _MIN_ELEMS_PER_TASK = PropagationBase._MIN_ELEMS_PER_TASK

"""
    VectorPauliSum{TV,CV} <: AbstractPauliSum

`VectorPauliSum` is a `mutable struct` that represents a sum of Pauli strings acting on `nqubits` qubits.
It is a wrapper around two vectors: one for the Pauli strings (as unsigned Integers for efficiency reasons), and one for the coefficients.
Using it defaults to multi-threaded operations where possible.
"""
mutable struct VectorPauliSum{TV,CV} <: AbstractPauliSum
    nqubits::Int
    terms::TV
    coeffs::CV
    # number of leading terms known sorted by integer value and duplicate-free; 0 is always safe
    _terms_sorted::Int

    function VectorPauliSum(nqubits::Int, terms::TV, coeffs::CV, _terms_sorted::Int=0) where {TV,CV}
        @assert length(terms) == length(coeffs) "Length of terms and coeffs must be the same. Got $(length(terms)) and $(length(coeffs))."
        @assert 0 <= _terms_sorted <= length(terms) "Sorted prefix length cannot be greater than the number of terms. Got $(length(terms)) and $( _terms_sorted)."
        return new{TV,CV}(nqubits, terms, coeffs, _terms_sorted)
    end
end

"""
    VectorPauliSum(nqubits::Int)

Constructor for an empty `VectorPauliSum` on `nqubits` qubits. Element type defaults for Float64.
"""
VectorPauliSum(nqubits::Int) = VectorPauliSum(Float64, nqubits)

"""
    VectorPauliSum(::Type{CT}, nqubits::Int)

Contructor for an empty `VectorPauliSum` on `nqubits` qubits. The type of the coefficients can be provided.
"""
VectorPauliSum(::Type{CT}, nqubits::Int) where {CT} = VectorPauliSum(nqubits, getinttype(nqubits)[], CT[])

PropagationBase.storage(vpsum::VectorPauliSum) = (vpsum.terms, vpsum.coeffs)

PropagationBase.sortedprefix(vpsum::VectorPauliSum) = vpsum._terms_sorted
PropagationBase.setsortedprefix!(vpsum::VectorPauliSum, n::Int) = (vpsum._terms_sorted = n; vpsum)

"""
    nqubits(vpsum::VectorPauliSum)

Get the number of qubits that the `VectorPauliSum` is defined on.
"""
nqubits(vpsum::VectorPauliSum) = vpsum.nqubits


Base.similar(vpsum::VectorPauliSum) = VectorPauliSum(vpsum.nqubits, similar(vpsum.terms), similar(vpsum.coeffs))

function Base.resize!(vpsum::VectorPauliSum, n_new::Int)
    resize!(vpsum.terms, n_new)
    resize!(vpsum.coeffs, n_new)
    setsortedprefix!(vpsum, min(sortedprefix(vpsum), n_new))  # clamp on shrink, no-op on grow
    return vpsum
end


function Base.show(io::IO, vecpsum::VectorPauliSum)
    n_paulis = length(vecpsum)
    if n_paulis == 0
        println(io, "Empty VectorPauliSum.")
        return
    elseif n_paulis == 1
        println(io, "VectorPauliSum with 1 term:")
    else
        println(io, "VectorPauliSum with $(n_paulis) terms:")
    end

    for i in 1:length(vecpsum)
        if i > 20
            println(io, "  ...")
            break
        end
        pauli_string = inttostring(vecpsum.terms[i], vecpsum.nqubits)
        if length(pauli_string) > 20
            pauli_string = pauli_string[1:20] * "..."
        end
        println(io, vecpsum.coeffs[i], " * $(pauli_string)")
    end
end


function Base.conj!(vpsum::VectorPauliSum)
    vpsum.coeffs .= conj.(vpsum.coeffs)
    return vpsum
end


function Base.sort!(vpsum::VectorPauliSum; by=nothing, kwargs...)
    # instead of using sortperm, we use sort!() on an index array 
    # this is to be able to sort on any properties of the terms of coeffs 

    indices = collect(1:length(vpsum))

    # default for if "by" is not provided
    byfunc = isnothing(by) ? i -> vpsum.terms[i] : by

    AK.sort!(indices; by=byfunc, kwargs...)
    vpsum.terms .= view(vpsum.terms, indices)
    vpsum.coeffs .= view(vpsum.coeffs, indices)
    setsortedprefix!(vpsum, 0)  # arbitrary order, no dedup: can't assume sorted+unique after this
    return vpsum
end
