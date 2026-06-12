###
##
# This file contains functions to convert a circuit or a PTM to a transfer map.
##
###


"""
    TransferMap(columns)

A contiguous lookup table for Pauli transfer maps.

The map is indexed by the partial Pauli integer directly, so `tmap[0]`
returns the image of the identity column.
"""
struct TransferMap{TT,CT}
    entries::Vector{Tuple{TT,CT}}
    offsets::Vector{Int}
end

function _power_exponent(dim::Integer, base::Integer, label)
    dim > 0 || throw(ArgumentError("$label dimension must be positive. Got $dim."))

    exponent = 0
    value = 1
    while value < dim
        value *= base
        exponent += 1
    end

    value == dim || throw(ArgumentError("$label dimension must be a power of $base. Got $dim."))
    return exponent
end

function TransferMap(columns::AbstractVector{<:AbstractVector{Tuple{TT,CT}}}) where {TT,CT}
    _power_exponent(length(columns), 4, "TransferMap column")

    total_entries = sum(length, columns)
    entries = Vector{Tuple{TT,CT}}(undef, total_entries)
    offsets = Vector{Int}(undef, length(columns) + 1)

    next_entry = 1
    for (column_index, column) in enumerate(columns)
        offsets[column_index] = next_entry
        for item in column
            entries[next_entry] = item
            next_entry += 1
        end
    end
    offsets[end] = next_entry

    return TransferMap{TT,CT}(entries, offsets)
end

function TransferMap(mat::AbstractMatrix; ptm::Bool=false)
    nrows, ncols = size(mat)
    nrows == ncols || throw(ArgumentError("TransferMap input matrix must be square. Got size $(size(mat))."))
    _power_exponent(nrows, ptm ? 4 : 2, ptm ? "PTM" : "unitary")

    return totransfermap(ptm ? mat : calculateptm(mat))
end

_ncolumns(tmap::TransferMap) = length(tmap.offsets) - 1
_max_terms_per_column(tmap::TransferMap) = maximum(diff(tmap.offsets))

Base.length(tmap::TransferMap) = length(tmap.entries)
Base.IteratorSize(::Type{<:TransferMap}) = Base.SizeUnknown()
nqubits(tmap::TransferMap) = _power_exponent(_ncolumns(tmap), 4, "TransferMap")

function Base.show(io::IO, tmap::TransferMap)
    print(
        io,
        "TransferMap(",
        _ncolumns(tmap),
        " columns, ",
        length(tmap),
        " entries, max ",
        _max_terms_per_column(tmap),
        " mapped terms/column)",
    )
end

function Base.:(==)(left::TransferMap, right::TransferMap)
    return left.entries == right.entries && left.offsets == right.offsets
end

function Base.hash(tmap::TransferMap, h::UInt)
    return hash(tmap.offsets, hash(tmap.entries, h))
end

function Base.getindex(tmap::TransferMap, pauli_int::Integer)
    index = Int(pauli_int)
    if index < 0 || index >= _ncolumns(tmap)
        throw(BoundsError(tmap, pauli_int))
    end

    start = tmap.offsets[index+1]
    stop = tmap.offsets[index+2] - 1
    return @view tmap.entries[start:stop]
end

function Base.iterate(tmap::TransferMap, state::Int=0)
    state >= _ncolumns(tmap) && return nothing
    return (tmap[state], state + 1)
end


"""
    totransfermap(nq::Integer, circuit::Vector{Gate}, thetas=nothing)

Computes the Pauli transfer map acting on `nq` qubits from a circuit with parameters `thetas`.
`thetas` defaults to `nothing` but is required if the circuit contains parametrized gates.
The returned lookup map is a `TransferMap` whose columns contain entries like `(pstr1, coeff1)`, `(pstr2, coeff2)`, ...,
where the `pstr` are *partial* Pauli strings on the affected qubits.
"""
function totransfermap(nq::Integer, circuit, thetas=nothing)

    TermType = getinttype(nq)

    # max integer to feed into the circuit
    max_integer = 4^nq - 1

    # Do one propagation per initial Pauli string on the number of qubits (can be very expensive)
    psums = [propagate(circuit, PauliString(nq, TermType(ii), 1.0), thetas; min_abs_coeff=0) for ii in 0:max_integer]

    # Convert our transfer map style, i.e., vector of vector of tuples
    columns = [[(TermType(paulis), coeff) for (paulis, coeff) in psum] for psum in psums]
    return TransferMap(columns)

end

"""
    totransfermap(ptm::AbstractMatrix)

Computes the Pauli transfer map acting on `nq` qubits from a Pauli Transfer Matrix (PTM).
The PTM should be the matrix representation of a gate in Pauli basis.
The returned lookup map is a `TransferMap` whose columns contain entries like `(pstr1, coeff1)`, `(pstr2, coeff2)`, ...,
where the `pstr` are *partial* Pauli strings on the affected qubits.
"""
function totransfermap(ptm::AbstractMatrix)
    nrows, ncols = size(ptm)
    nrows == ncols || throw(ArgumentError("PTM must be square. Got size $(size(ptm))."))
    nq = _power_exponent(nrows, 4, "PTM")
    col_length = nrows

    columns = Vector{Vector{Tuple{getinttype(nq),eltype(ptm)}}}(undef, col_length)
    # each column becomes on entry in the lookup map vector
    for (colind, colvals) in enumerate(eachcol(ptm))
        # the Pauli integers need to to from 0 to 3, so subtract 1
        columns[colind] = [(rowind - 1, ptm[rowind, colind]) for (rowind, val) in enumerate(colvals) if val != 0]
    end
    return TransferMap(columns)
end
