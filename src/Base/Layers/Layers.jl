# gatelayer.jl defines `GateLayer`, gates that commute with each other applied together.
include("gatelayer.jl")

# classes.jl applies a layer of commuting gates class by class through the gate path: the plan of a layer, what a basis
# provides, the pass over the classes, the class of a term, and the lookup of the gates acting on a term.
include("classes.jl")

# classpass.jl applies a function to the terms of a sum class by class, in one pass over the sum, and holds the
# workspaces that the passes of a propagation reuse.
include("classpass.jl")
