# MultiPauliSum

`MultiPauliSum` is the way to get the most out of multithreading in `PauliPropagation.jl`. It takes an ordinary `PauliSum` or `VectorPauliSum` and splits it into *work zones*, each of which is a Pauli sum of that same type owned by a single thread. Everything else stays the same: you build it from the observable you already have, hand it to `propagate()` as usual, and read expectation values off it directly.

```julia
using PauliPropagation

nqubits = 32
observable = PauliString(nqubits, :Z, 16)
circuit = tfitrottercircuit(nqubits, 32)
parameters = ones(countparameters(circuit)) * 0.1

msum = MultiPauliSum(VectorPauliSum(observable))
psum = propagate(circuit, msum, parameters; min_abs_coeff=1e-4)

overlapwithzero(psum)
```

Start Julia with several threads (`julia -t 8`, for example) and the zones are worked in parallel. On a single thread, a `MultiPauliSum` simply behaves like the Pauli sum it wraps.

## Why zones

`VectorPauliSum` is already multithreaded, but it threads by splitting every gate application across all threads, which means the threads have to coordinate on the same Pauli sum and only pays off once the sum is large. A `MultiPauliSum` threads differently. Every Pauli string belongs to exactly one zone, decided by a fixed rule that spreads the strings evenly over the zones no matter what the sum looks like. Because all copies of a Pauli string land in the same zone, merging duplicates never has to look outside a zone, and because each zone is worked by one thread, no two threads ever write to the same place.

When a gate creates Pauli strings that belong to another zone, the thread does not reach into that zone but leaves them in an outbox, and the owner collects them afterwards. That is the whole coordination between threads: one hand-off per gate. The result is parallelism that helps at all sizes, not only at the largest ones.

## Using it

The constructors accept whatever you would otherwise pass to `propagate()`. The type of the Pauli sum you pass in is the type of the zones, so `MultiPauliSum(PauliSum(observable))` gives dictionary-backed zones and `MultiPauliSum(VectorPauliSum(observable))` gives array-backed ones. For performance, prefer `VectorPauliSum` zones.

```julia
MultiPauliSum(observable)                       # PauliSum zones, one per thread
MultiPauliSum(VectorPauliSum(observable))       # VectorPauliSum zones
MultiPauliSum(VectorPauliSum(observable), 16)   # over 16 zones
MultiPauliSum(nqubits)                          # an empty sum with PauliSum zones
```

The number of zones defaults to one per thread and has to be a power of two, which is what lets the Pauli rotations move terms between zones cheaply. If your thread count is not a power of two, the default picks a larger zone count so that no thread sits idle.

A `MultiPauliSum` works with the rest of the library the way any other Pauli sum does:

- `propagate()` and the in-place `propagate!()`, with all coefficient and weight truncations.
- `Performance.propagate()` for the fused single-pass application inside every zone. This needs `VectorPauliSum` zones; with `PauliSum` zones it falls back to the ordinary application.
- `PropagationCache` and `resize!`, to size the memory up front for the peak number of terms.
- `mcpropagate()` and its resampling strategies.
- `rewindgradient()` for gradients.
- `overlapwithzero()`, `overlapwithplus()`, `getcoeff()`, iteration, `length`, and the other functions you would use to inspect a Pauli sum.

To get a plain Pauli sum back, call `PauliSum(msum)` or `VectorPauliSum(msum)`. `nzones(msum)` and `zonesizes(msum)` tell you how the terms are distributed.

The [advanced performance notebook](../../examples/advanced_performance.ipynb) walks through this in context, together with the other performance tricks in the library.