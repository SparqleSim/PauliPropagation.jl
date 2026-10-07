# gatelayer.jl defines `GateLayer`, gates that commute with each other applied together.
include("gatelayer.jl")

# labelpass.jl applies a function to the terms of a sum label by label, in one pass over the sum, and holds the
# workspaces that the passes of a propagation reuse.
include("labelpass.jl")

# classes.jl applies a layer of commuting rotations class by class: the plan of a layer, what a basis provides, the lookup
# of the rotations acting on a term, and the classes.
include("classes.jl")

# classkernels.jl rotates one class, densely if it has few distinguishing bits and sparsely otherwise.
include("classkernels.jl")
