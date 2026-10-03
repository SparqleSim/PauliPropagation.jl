# propagationcache.jl contains the PropagationCache type and related functions.
# it carries the main Pauli sum and auxiliary data structures for efficient propagation.
include("propagationcache.jl")

# generics.jl contains the core functionality of the `propagation` function.
include("generics.jl")

# specializations.jl contains the gates of the library, written once for every propagation cache.
include("specializations.jl")


# mcsample.jl contains the Monte Carlo (mcsample!) specializations for VectorPauliSum.
include("mcsample.jl")

# rotationlayers.jl contains how a `RotationLayer` is propagated as a whole: its passes over the sum, records and outputs.
include("rotationlayers.jl")

# rotationlayerclasses.jl contains how the classes of a `RotationLayer` are found and rotated, a class of many key bits in a table.
include("rotationlayerclasses.jl")

# rotationlayerblocks.jl contains how a class of few key bits is rotated as a dense block.
include("rotationlayerblocks.jl")