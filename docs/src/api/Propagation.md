# Propagation

## Generics

```@autodocs
Modules = [PauliPropagation]
Pages = ["src/Propagation/generics.jl"]
Filter = t -> !(t in (PauliPropagation.mcpropagate!, PauliPropagation.mcsample!))
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
