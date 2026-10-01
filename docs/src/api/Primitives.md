# Primitives

These functions are defined in the `PropagationBase` submodule for generic term sums, so their signatures name a `term_sum::AbstractTermSum` or a `prop_cache::AbstractPropagationCache` rather than a Pauli sum.
Every `PauliSum`, `VectorPauliSum` and `MultiPauliSum` is an `AbstractTermSum`, and `PropagationCache(psum)` wraps one of them into the cache that `propagate!` works on, so each function below accepts either.

## Map
```@autodocs
Modules = [PauliPropagation.PropagationBase]
Pages = ["src/Base/Primitives/map.jl", "src/Base/Primitives/flatmap.jl"]
```

## Reduce
```@autodocs
Modules = [PauliPropagation.PropagationBase]
Pages = ["src/Base/Primitives/mapreduce.jl", "src/Base/Primitives/maxabscoeff.jl"]
```

## Filter
```@autodocs
Modules = [PauliPropagation.PropagationBase]
Pages = ["src/Base/Primitives/filter.jl"]
```

## Sort
```@autodocs
Modules = [PauliPropagation.PropagationBase]
Pages = ["src/Base/Primitives/sort.jl"]
```
