### Gates.jl
##
# The top level file for gates.
# Gates are defined as structs that subtype either `ParametrizedGate` or `StaticGate`.
# How the gates act is defined in the `Propagation` module.
##
###


include("frozengates.jl")
include("paulirotations.jl")
include("rotationlayers.jl")
include("imaginarytime.jl")
include("cliffordgates.jl")
include("noisechannels.jl")
include("miscgates.jl")
include("heisenbergschrodinger.jl")


## Helper functions

"""
    maxqind(circuit)

Returns the highest index of a qubit that a gate of the circuit acts on, or 0 for a circuit without gates.
"""
function maxqind(circuit)
    max_qind = 0
    for gate in circuit
        max_qind = max(max_qind, maximum(qinds(gate); init=0))
    end
    return max_qind
end

function _qinds_check(qinds)
    if any(qind -> qind <= 0, qinds)
        throw(ArgumentError("Qubit indices must be positive integers. Got $qinds."))
    end

    if !allunique(qinds)
        throw(ArgumentError("Qubit indices must be unique. Got $qinds."))
    end

    if !all(qind -> isa(qind, Integer), qinds)
        throw(ArgumentError("Qubit indices must be integers. Got $qinds."))
    end


end

function _qinds_check(qind::Integer)
    if qind <= 0
        throw(ArgumentError("Qubit index must be positive integer. Got $qind."))
    end
end