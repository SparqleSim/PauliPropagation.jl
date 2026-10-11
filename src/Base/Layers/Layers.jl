# gatelayer.jl defines `GateLayer`, gates that commute with each other applied together.
include("gatelayer.jl")

# labelpass.jl applies a function to the terms of a sum label by label, in one pass over the sum, and holds the
# workspaces that the passes of a propagation reuse.
include("labelpass.jl")

# cosets.jl applies a layer of commuting gates coset by coset through the gate path: the plan of a layer, what a basis
# provides, the lookup of the gates acting on a term, and the pass over the cosets.
include("cosets.jl")
