# propagationcache.jl contains the PropagationCache type and related functions.
# it carries the main Pauli sum and auxiliary data structures for efficient propagation.
include("propagationcache.jl")

# generics.jl contains the core functionality of the `propagation` function.
include("generics.jl")

# specializations.jl contains specialized implementations of lower level propagation functions for specific gates
# and the PauliSum type. It assumes a Dict container and is currently not multithreaded.
include("specializations.jl")

# vectorspecializations.jl contains specializations for the VectorPauliSum type,
# which is intended for multithreaded CPU and GPU propagation.
include("vectorspecializations.jl")