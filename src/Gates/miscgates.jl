### miscgates.jl
##
# A file for gates that don't fit into the other categories.
##
###


## T gate

struct TGate <: StaticGate
    qind::Int

    @doc """
        TGate(qind::Integer)

    Returns a T gate acting on qubit `qind`.
    It acts on qubit `qind` like a `PauliRotation(:Z, qind)` with angle π/4.
    """
    TGate(qind::Integer) = (_qinds_check(qind); new(qind))

end

"""
    tomatrix(gate::TGate)

Compute the unitary matrix for a `TGate`.
The returned unitary is returned in Schrödinger picture form.
"""
function tomatrix(::TGate)
    return _tgate_unitary
end

const _tgate_unitary = [[1 0]; [0 exp(1.0im * pi / 4)]]


## TransferMapGate
"""
    TransferMapGate(transfer_map, qinds::Vector{Int})

A non-parametrized `StaticGate` defined by a transfer map acting on the qubits `qinds`.
Transfer maps can be constructed manually or generated via `totransfermap()`.
"""
struct TransferMapGate{TM<:TransferMap} <: StaticGate
    transfer_map::TM
    qinds::Vector{Int}

    function TransferMapGate(transfer_map::TM, qinds) where {TM<:TransferMap}
        # accept anything that can be converted to a vector of integers
        qinds = vec(collect(qinds))
        nq = length(qinds)
        _qinds_check(qinds)

        if length(transfer_map) != 4^nq
            throw(ArgumentError("The length of `qinds` `n=$nq` does not match the length of the transfer map `$(length(transfer_map)) != 4^$nq`."))
        end

        return new{TM}(transfer_map, qinds)
    end
end

TransferMapGate(transfer_map::AbstractVector{<:AbstractVector{<:Tuple}}, qinds) = TransferMapGate(TransferMap(transfer_map), qinds)


"""
A constructor for `TransferMapGate` that accepts matrix representations in the 0/1 basis or the Pauli basis (a PTM).
"""
function TransferMapGate(mat::AbstractMatrix, qinds)
    # turns number or tuple of numbers into vector of numbers
    qinds = vec(collect(qinds))
    # number of qubits acted on
    nq = length(qinds)

    # infer from the size of the matrix and nq whether it is a matrix in the 0/1 basis or the Pauli basis
    mat_size = size(mat)

    if mat_size != (2^nq, 2^nq) && mat_size != (4^nq, 4^nq)
        throw(ArgumentError("The matrix must be square and have size (2^$nq x 2^$nq) or (4^$nq x 4^$nq) " *
                            "given the passed qinds=$qinds."))
    end

    if mat_size == (2^nq, 2^nq)
        # the matrix is assumed to be in the 0/1 basis
        # transform it into a PTM
        mat = calculateptm(mat)
    end

    # here mat is already a PTM
    ptmap = TransferMap(mat; ptm=true)

    return TransferMapGate(ptmap, qinds)

end
