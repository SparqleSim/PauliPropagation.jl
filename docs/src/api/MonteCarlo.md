# Monte Carlo

## Resampling

```@autodocs
Modules = [PauliPropagation.PropagationBase]
Pages = ["src/Base/MonteCarlo/mcpropagate.jl"]
```

```@autodocs
Modules = [PauliPropagation]
Pages = ["src/Propagation/generics.jl"]
Filter = t -> t === PauliPropagation.mcpropagate!
```

## Resampling Strategies

```@autodocs
Modules = [PauliPropagation.PropagationBase]
Pages = ["src/Base/MonteCarlo/resample.jl"]
```

## Path Sampling

```@autodocs
Modules = [PauliPropagation.PropagationBase]
Pages = ["src/Base/MonteCarlo/mcsample.jl"]
```

```@autodocs
Modules = [PauliPropagation]
Pages = ["src/Propagation/generics.jl"]
Filter = t -> t === PauliPropagation.mcsample!
```

### VectorPauliSum Specialization

```@autodocs
Modules = [PauliPropagation]
Pages = ["src/Propagation/vectormontecarlo.jl"]
```