###
##
# How a `RotationLayer` is propagated as a whole.
# The rotations of a layer commute, so the Pauli strings that they turn into each other form orbits that do not mix.
# A sublayer collects the Pauli strings of every orbit and applies its rotations to the coefficients of the orbit,
# instead of branching every Pauli string and merging the branches rotation by rotation.
##
###

# sublayerplan.jl finds the rotations that anticommute with a Pauli string, and the orbit they span.
include("sublayerplan.jl")

# workspace.jl contains the scratch memory of a layer.
include("workspace.jl")

# records.jl writes a record for every Pauli string and sorts the records into partitions.
include("records.jl")

# grouping.jl collects the records of the same orbit.
include("grouping.jl")

# orbits.jl applies the rotations to an orbit.
include("orbits.jl")

# longorbits.jl applies the rotations to an orbit of many stages.
include("longorbits.jl")

# sublayer.jl applies a sublayer to a propagation cache.
include("sublayer.jl")

# multisum.jl applies a sublayer to the propagation cache of a multi sum.
include("multisum.jl")
