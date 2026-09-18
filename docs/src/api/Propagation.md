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

## Branching

The pass that a gate branching by a fixed Pauli string is built from. Like the primitives, it is defined in the `PropagationBase` submodule for generic term sums and propagation caches.

```@autodocs
Modules = [PauliPropagation.PropagationBase]
Pages = ["src/Base/xorbranch.jl"]
```

## Counting Pauli Strings

```@autodocs
Modules = [PauliPropagation]
Pages = ["src/countpaulis.jl"]
```
