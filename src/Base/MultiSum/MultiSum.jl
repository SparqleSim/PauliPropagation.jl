# multisumstorage.jl defines the MultiSumStorage trait: a term sum split over one work zone per thread,
# with an owning zone per term.
include("multisumstorage.jl")

# propagationcache.jl carries one propagation cache per zone and one outbox per zone.
include("propagationcache.jl")
