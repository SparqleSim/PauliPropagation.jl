# multisum.jl defines the MultiSum type: a term sum split over one work zone per thread,
# with an owning zone per term.
include("multisum.jl")

# propagationcache.jl carries one propagation cache per zone and one outbox per zone.
include("propagationcache.jl")

# gates.jl applies a gate zone by zone, in a pass that makes terms and a pass that delivers them.
include("gates.jl")
