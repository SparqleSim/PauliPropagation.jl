# MultiSum

A term sum split over `n_zones` work zones, each a term sum of the type it carries that one thread
owns. Every operation on a zone is single-threaded; parallelism comes from the zones alone. This is
the multi-node scheme of the data structure appendix, run over threads on one node.

## The idea

A fixed assignment sends each term to one zone, so all copies of a term reach the same owner and
deduplication never has to look outside a zone. A gate makes terms that belong to other zones; a
thread parks those in an outbox instead of writing into a zone it does not own, and the owner picks
them up in a second pass. No zone is written by two threads and no operation on a zone needs to be
thread-safe.

The assignment is what makes this cheap for Pauli rotations. Each bit of the zone index is a parity
of the term under a fixed mask, so it is linear over GF(2):

    zoneof(t ⊻ m) - 1 == (zoneof(t) - 1) ⊻ (zoneof(m) - 1)

A Pauli rotation moves every term it branches by the same `⊻ m`, so it permutes the zones: each zone
sends all of its terms to exactly one other zone and receives from exactly one. An owner therefore
has a single outbox to take delivery of, and that outbox is `parent ⊻ m` over a sorted, duplicate-free
zone -- exactly the input `xorsortedtailmerge!` wants.

Balance comes for free: parities of fixed pseudo-random masks spread the terms evenly no matter how
the sum is shaped. Zone sizes scatter like the square root of their size, so the largest and smallest
of 8 zones differ by 2% over 19k terms, and by less as the sum grows.

## How a gate is applied

Two zone-parallel passes, separated by a barrier:

1. Every zone applies the gate to its own terms. A gate that branches rescales the terms it branches
   and parks the terms they make in its outbox -- in a single box of it, since they all move by the
   same `⊻ m`; any other gate routes what it makes term by term and empties the zone.
2. Every zone appends what the outboxes hold for it. `merge!` and `truncate!` then run zone by zone,
   which is the library's own `applytoall!`-`merge!`-`truncate!` order.

An outbox is itself a `MultiSum`, so parking a term is the same routing as adding one. A gate that
leaves every term where it is -- `staysinzone(gate)`, as for `PauliNoise` -- skips both passes and
runs inside the zones instead.

Everything below the two passes dispatches on the `StorageType` of the sum the zones carry, so a
`MultiSum` of `PauliSum`s and one of `VectorPauliSum`s share the same code.

## Capacity

`resize!(prop_cache, n)` gives the zones room for `n` terms between them. Sizing for the terms that
survive is not enough: a zone holds the terms addressed to it next to the terms it already has, so
its share has to cover that peak, and a hint that only covers the result still reallocates near the
end of a run.

## What it buys

A 36-qubit tilted-field Ising circuit run out to 6.7M terms on 8 threads, against the same sum
propagated single-threaded: 4.4x over a `PauliSum`, 2.6x over a `VectorPauliSum`, and 1.5x over a
`VectorPauliSum` propagated with the library's own threading.

## Not here yet

The fused single-pass gate applications, which branch straight into the outbox and merge it with the
XOR tail sort, belong in the `Performance` module and are reached through `Performance.propagate`.

Monte Carlo propagation (`mcpropagate`, `mcsample`, `resample`) and `rewindgradient` do not take a
`MultiSum`: resampling weighs the whole sum at once, and neither has a zone-parallel form yet.
