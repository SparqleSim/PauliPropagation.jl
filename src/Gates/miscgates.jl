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
struct TransferMapGate{TM<:TransferMap,STM<:TransferMap,TMask<:PauliStringType,Contiguous} <: StaticGate
    transfer_map::TM
    qinds::Vector{Int}
    qind_mask::TMask
    shifted_transfer_map::STM
    qind_start::Int
    qind_stop::Int

    function TransferMapGate(transfer_map::TM, qinds) where {TM<:TransferMap}
        # accept anything that can be converted to a vector of integers
        qinds = vec(collect(qinds))
        nq = length(qinds)
        _qinds_check(qinds)

        if _ncolumns(transfer_map) != 4^nq
            throw(ArgumentError("The length of `qinds` `n=$nq` does not match the transfer map column count `$(_ncolumns(transfer_map)) != 4^$nq`."))
        end

        mask_type = getinttype(maximum(qinds))
        qind_mask = _pauliqindmask(mask_type, qinds)
        shifted_transfer_map = _shifttransfermap(mask_type, transfer_map, qinds)
        qind_start = first(qinds)
        qind_stop = last(qinds)
        contiguous_qinds = _qindsarecontiguous(qinds)

        return new{TM,typeof(shifted_transfer_map),typeof(qind_mask),contiguous_qinds}(
            transfer_map,
            qinds,
            qind_mask,
            shifted_transfer_map,
            qind_start,
            qind_stop,
        )
    end
end

TransferMapGate(transfer_map::AbstractVector{<:AbstractVector{<:Tuple}}, qinds) = TransferMapGate(TransferMap(transfer_map), qinds)

function _qindsarecontiguous(qinds)
    for ii in 2:length(qinds)
        qinds[ii] == qinds[ii-1] + 1 || return false
    end
    return true
end

function _pauliqindmask(::Type{TT}, qinds) where {TT<:PauliStringType}
    qind_mask = zero(TT)
    for qind in qinds
        qind_mask |= TT(3) << _bitshiftfromsiteindex(qind)
    end
    return qind_mask
end

function _shiftpartialpauli(::Type{TT}, target_paulis, qinds) where {TT<:PauliStringType}
    shifted_pstr = zero(TT)
    for (ii, qind) in enumerate(qinds)
        shifted_pstr |= TT(getpauli(target_paulis, ii)) << _bitshiftfromsiteindex(qind)
    end
    return shifted_pstr
end

function _shifttransfermap(::Type{TT}, transfer_map::TransferMap, qinds) where {TT<:PauliStringType}
    columns = [[(_shiftpartialpauli(TT, new_pstr, qinds), factor) for (new_pstr, factor) in column] for column in transfer_map]
    return TransferMap(columns)
end


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
