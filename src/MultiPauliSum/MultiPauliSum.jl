# datatypes.jl defines the MultiPauliSum type: a Pauli sum split over one work zone per thread,
# with an owning zone per Pauli string.
include("datatypes.jl")

# propagationcache.jl carries one propagation cache per zone and one outbox per zone.
include("propagationcache.jl")
