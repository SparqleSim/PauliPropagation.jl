# MultiPauliSum

A Pauli sum split over `n_zones` work zones, each a Pauli sum of the type it carries that one thread
owns. Every operation on a zone is single-threaded; parallelism comes from the zones alone. This is
the multi-node scheme of the data structure appendix, run over threads on one node.

The machinery is basis-agnostic and lives in `PropagationBase` under `Base/MultiSum`, behind the
`MultiSumStorage` storage trait. This directory holds what is specific to the Pauli basis: the type
itself, its propagation cache, and the gates.

## The idea

A fixed assignment sends each term to one zone, so all copies of a term reach the same owner and
deduplication never has to look outside a zone. A gate makes terms that belong to other zones; a
thread parks those in an outbox instead of writing into a zone it does not own, and the owner picks
them up in a second pass. No zone is written by two threads and no operation on a zone needs to be
thread-safe.

Each bit of the assignment is a parity of the term under a fixed mask. Balance comes for free:
parities of fixed pseudo-random masks spread the terms evenly no matter how the sum is shaped. Zone
sizes scatter like the square root of their size, so the largest and smallest of 8 zones differ by 2%
over 19k terms, and by less as the sum grows.

Over a power-of-two number of zones the parity bits are the zone index itself, which makes the
assignment linear over GF(2):

    zoneof(t ⊻ m) - 1 == (zoneof(t) - 1) ⊻ (zoneof(m) - 1)

A Pauli rotation moves every term it branches by the same `⊻ m`, so it permutes the zones: each zone
sends all of its terms to exactly one other zone and receives from exactly one. An owner therefore
has a single outbox to take delivery of, and that outbox is `parent ⊻ m` over a sorted, duplicate-free
zone -- exactly the input `xorsortedtailmerge!` wants.

Any other zone count folds more parity buckets than there are zones onto the zones, which costs the
linearity: `⊻` is only closed on the buckets, so a rotation no longer permutes the zones and has to
route the terms it makes one by one, as any other gate does. Nothing else changes, and the two maps
are `XorZoneMap` and `FoldedZoneMap`.

## How a gate is applied

Two zone-parallel passes, separated by a barrier:

1. Every zone applies the gate to its own terms. A gate that branches by a fixed `⊻ m` over a
   power-of-two number of zones parks the terms it makes in a single box of its outbox; any other
   gate routes what it makes term by term and empties the zone.
2. Every zone appends what the outboxes hold for it -- from the one zone that sends to it if the
   gate permuted the zones, from all of them otherwise. `merge!` and `truncate!` then run zone by
   zone, which is the library's own `applytoall!`-`merge!`-`truncate!` order.

A box is emptied by the zone that takes delivery, so every box is empty when a gate picks it up and
the fast path never has to clear the boxes it does not use. A gate that branches by a fixed mask
always parks in the same box, the first, rather than in the box of the zone it happens to send to:
which zone that is moves with the mask, and parking by it would leave every box of every outbox grown
to the size of a zone.

An outbox is itself a `MultiPauliSum`, so parking a term is the same routing as adding one. A gate that
leaves every term where it is -- `staysinzone(gate)`, as for `PauliNoise` -- skips both passes and
runs inside the zones instead.

A `MultiPauliSum` carries the `MultiSumStorage` trait, which routes the term sum interface, `merge!`
and `truncate!` through the owning zone. The trait carries the storage of the zones in turn, so a
`MultiPauliSum` of `PauliSum`s and one of `VectorPauliSum`s share the same code.

Gate application dispatches on the cache instead. A `MultiPauliPropagationCache` is an
`AbstractPauliPropagationCache`, so it inherits `propagate!` and everything above the gate, and every
gate reaches `applytoallzones!` or, if it branches by a fixed mask, `applyxorbranch!`.

## Capacity

`resize!(prop_cache, n)` gives the zones room for `n` terms between them. Sizing for the terms that
survive is not enough: a zone holds the terms addressed to it next to the terms it already has, so
its share has to cover that peak, and a hint that only covers the result still reallocates near the
end of a run.

## What it buys

A 24-qubit circuit of six Rx-Rz-CNOT-Rzz layers, run out to 9.1M terms over 8 zones on 8 threads,
against the same sum over a single zone: 5.1x for `PauliSum` zones and 2.9x for `VectorPauliSum`
zones, the latter 2.0x over a `VectorPauliSum` propagated with the library's own threading.

A 36-qubit tilted-field Ising circuit run out to 6.9M terms on 8 threads, against a `PauliSum`
propagated single-threaded: 9.6x over 8 `VectorPauliSum` zones, and 18.8x with the fused application
inside the zones. The fused zones are 1.4x the fused `VectorPauliSum` the library threads itself,
which is otherwise the fastest way to run that circuit.

Zone count and thread count are separate choices, and a power of two is worth giving up threads for.
Over the same 8 zones on the same 8 threads, folding the zone assignment instead of keeping it linear
costs 3-6% for `PauliSum` zones, which route every term they make in any case, and 29-34% for
`VectorPauliSum` zones, which lose the XOR tail sort with it. Both maps balance the zones equally
well.

## Fused application

`Performance.propagate` runs the fused single-pass application inside every zone: a zone writes what
it branches straight into its box, and the truncations that read the coefficient are paid in the merge
that takes delivery, rather than in a pass of their own. `applyxorbranchzones!` is
`applyxorbranch!` with the first pass handed to the caller, so both share the delivery and the merge.

Fusing asks more of the zones than the default does. The rotations need array-backed zones and a
power-of-two zone count, since a zone writes its products contiguously into the one box it owns;
`PauliNoise` only needs the former, staying where it is. Anything else falls back to the default
application.

## Not here yet

A gate that routes its terms leaves every zone holding a concatenation of runs, each of them a sorted
zone under a fixed `⊻`: one run per zone that sent to it, and for a Clifford gate one more per Pauli
the gate maps on its qubits. Sorting the runs by XOR passes and merging them, rather than re-sorting
the zone, would cover the Clifford gates, which pay a full sort per gate as it is, and would give a
folded zone assignment back most of what it loses.

Monte Carlo propagation (`mcpropagate`, `mcsample`, `resample`) and `rewindgradient` do not take a
`MultiPauliSum`: resampling weighs the whole sum at once, and neither has a zone-parallel form yet.
