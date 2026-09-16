# propagationcache.jl contains the PropagationCache type and related functions.
# it carries the main Pauli sum and auxiliary data structures for efficient propagation.
include("propagationcache.jl")

# generics.jl contains the core functionality of the `propagation` function.
include("generics.jl")

# specializations.jl contains the gates of the library, written once for every propagation cache.
include("specializations.jl")

# vectormontecarlo.jl contains the Monte Carlo (mcapplytoall!) specializations for VectorPauliSum.
include("vectormontecarlo.jl")