# propagationcache.jl contains the PropagationCache type and related functions.
# it carries the main Pauli sum and auxiliary data structures for efficient propagation.
include("propagationcache.jl")

# generics.jl contains the core functionality of the `propagation` function.
include("generics.jl")

# specializations.jl contains the gates of the library, written once for every propagation cache.
include("specializations.jl")


# mcsample.jl contains the Monte Carlo (mcsample!) specializations for VectorPauliSum.
include("mcsample.jl")

# rotationlayers.jl contains how a `RotationLayer` is propagated orbit by orbit.
include("rotationlayers.jl")

# rotationlayerclasses.jl contains how a `RotationLayer` is propagated class by class.
include("rotationlayerclasses.jl")