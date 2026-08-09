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

## Vector Specializations

```@autodocs
Modules = [PauliPropagation]
Pages = ["src/Propagation/vectorspecializations.jl"]
```
