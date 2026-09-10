# Data types

## PauliString
```@autodocs
Modules = [PauliPropagation]
Pages = ["src/PauliDataTypes/paulistring.jl"]
```

## PauliSum
```@autodocs
Modules = [PauliPropagation]
Pages = ["src/PauliDataTypes/paulisum.jl"]
```

## VectorPauliSum
```@autodocs
Modules = [PauliPropagation]
Pages = ["src/PauliDataTypes/vectorpaulisum.jl"]
```

## MultiPauliSum
```@autodocs
Modules = [PauliPropagation]
Pages = ["src/MultiPauliSum/multipaulisum.jl", "src/MultiPauliSum/propagationcache.jl"]
```

```@autodocs
Modules = [PauliPropagation.PropagationBase]
Pages = ["src/Base/MultiSum/multisum.jl"]
Filter = t -> t in (PauliPropagation.zones, PauliPropagation.nzones, PauliPropagation.zonesizes, PauliPropagation.zoneof, PauliPropagation.PropagationBase.defaultnzones)
```

## Conversions
```@autodocs
Modules = [PauliPropagation]
Pages = ["src/PauliDataTypes/conversions.jl"]
```

## AbstractPauliSum
```@autodocs
Modules = [PauliPropagation]
Pages = ["src/PauliDataTypes/abstractpaulisum.jl"]
```

```@autodocs
Modules = [PauliPropagation.PropagationBase]
Pages = ["src/Base/termsum.jl"]
Filter = t -> t in (PauliPropagation.pushterm!, PauliPropagation.PropagationBase.emptylike)
```
