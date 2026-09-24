# Propagation

## Generics

```@autodocs
Modules = [PauliPropagation]
Pages = ["src/Propagation/generics.jl"]
Filter = t -> !(t in (PauliPropagation.mcpropagate, PauliPropagation.mcpropagate!, PauliPropagation.mcsample, PauliPropagation.mcsample!, PauliPropagation.resample, PauliPropagation.resample!))
```

## Specializations

```@autodocs
Modules = [PauliPropagation]
Pages = ["src/Propagation/specializations.jl"]
```

## Merging and fused propagation passes

The merges, and the passes that a gate can build from without materializing separate application,
merge, and truncation phases. They are defined in the `PropagationBase` submodule for generic
propagation caches.

```@autodocs
Modules = [PauliPropagation.PropagationBase]
Pages = ["src/Base/Merge/mergeandtruncate.jl", "src/Base/xorbranch.jl", "src/Base/Merge/xortailmerge.jl", "src/Base/Primitives/mapandtruncate.jl"]
```

## Counting Pauli Strings

```@autodocs
Modules = [PauliPropagation]
Pages = ["src/countpaulis.jl"]
```
