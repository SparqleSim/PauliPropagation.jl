# gatelayer.jl defines `GateLayer`, gates that commute with each other applied together.
include("gatelayer.jl")

# grouppass.jl applies a function to the terms of a sum grouped by a label, in one pass over the sum.
include("grouppass.jl")

# classes.jl applies a layer of commuting rotations class by class: the plan of a pass, what a basis provides, and the classes.
include("classes.jl")

# classtables.jl rotates a class of many key bits in a table of its own.
include("classtables.jl")

# classblocks.jl rotates a class of few key bits as a dense block.
include("classblocks.jl")

# workspace.jl contains the scratch memory of a pass, which the layers of a propagation reuse.
include("workspace.jl")
