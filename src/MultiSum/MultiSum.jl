###
##
# A Pauli sum split over one work zone per thread, with an owning zone per Pauli string.
# Every zone is a VectorPauliSum and every operation on it is single-threaded.
##
###

module MultiSum

using PauliPropagation
using PauliPropagation.PropagationBase
using Base.Threads
using Random

const PB = PauliPropagation.PropagationBase
const PF = PauliPropagation.Performance

include("multivectorpaulisum.jl")
include("gates.jl")

export
    MultiVectorPauliSum,
    nzones,
    zonesizes,
    applygate!

end
