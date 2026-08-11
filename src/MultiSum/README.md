# MultiVectorPauliSum

A Pauli sum split over `n_zones` work zones, each a `VectorPauliSum` that one thread owns. Every
operation on a zone is single-threaded; parallelism comes from the zones alone. This is the
multi-node scheme of the data structure appendix, run over threads on one node.

## The idea

A fixed assignment sends each Pauli string to one zone, so all copies of a string reach the same
owner and deduplication never has to look outside a zone. A gate makes products that belong to other
zones; a thread parks those in an outbox instead of writing into a zone it does not own, and the
owner picks them up in a second pass. No zone is written by two threads and no operation on a zone
needs to be thread-safe.

The assignment is what makes this cheap for Pauli rotations. Each bit of the zone index is a parity
of the Pauli string under a fixed random mask, so it is linear over GF(2):

    zoneof(p ⊻ m) - 1 == (zoneof(p) - 1) ⊻ (zoneof(m) - 1)

A Pauli rotation moves every product by the same `⊻ m`, so it permutes the zones: each zone sends all
of its products to exactly one other zone and receives from exactly one. An owner therefore has a
single outbox to merge, not one per source, and that outbox is `parent ⊻ m` over a sorted,
duplicate-free zone -- exactly the input `xorsortedtailmerge!` wants. Each zone runs the library's
fused apply and XOR tail merge unchanged, single-threaded, on its own share of the terms.

Balance comes for free: parities of random masks spread the strings evenly no matter how the sum is
shaped. On `example2.jl` the largest and smallest of 8 zones differ by 0.2%.

## Per gate family

| gate | traffic between zones | per-zone work |
| --- | --- | --- |
| `PauliRotation` | zone `s` sends to zone `s ⊻ δ` | branch into the outbox, then XOR tail merge |
| `PauliNoise` | none: Pauli strings do not move | the library's fused single-threaded overload |
| `CliffordGate` | all-to-all, so counted and scattered by owner | gather and sort, nothing to deduplicate |

`ImaginaryPauliRotation` is not covered: it branches the same way, but normalizing by the identity
coefficient needs a value that lives in whichever zone owns the identity.

## Capacity

`resize!(mpsum, n)` gives the zones room for `n` terms between them and every outbox room for the
products of its own zone. Sizing for the terms that survive is not enough: a zone holds the products
addressed to it next to the terms it already has, and on `example2.jl` that peak is 1.45x the final
share, so a hint that only covers the result still reallocates near the end of the run.

Growing on demand instead costs 1.12x at 18 layers and 1.17x at 12, and it is worse under threading
than the arithmetic suggests, because a zone that reallocates stops every other thread for the
collection. Pre-sized, an 18-layer run allocates 23 MB in total and spends no measurable time in the
collector; grown from empty it allocates 2.6 GB and spends 0.24 s.

## Files

- `multivectorpaulisum.jl` -- the type, the zone assignment, capacity, conversions back to the library types
- `gates.jl` -- the three gate families and the propagation loop
- `correctness.jl` -- checks every gate family against the library's `PauliSum` propagation
- `example2.jl` -- the benchmark below; arguments are layer count, zone counts, repetitions
- `profile.jl` -- splits the run into the branch and merge passes and reports straggler and
  barrier losses per zone count
- `machine.jl` -- the machine's memory bandwidth against thread count, the spread of identical
  work across cores, and the cost of one `@threads` region

## Measured

`bench/example2.jl` (6x6 tilted-field Ising, 132 rotations per layer, `min_abs_coeff=2^-20`),
8 threads on an 8-core Ryzen AI 7 350, best of 3-5 runs. The baselines are the fused
`VectorPauliSum` path. Timings on this laptop swing by up to 2x run to run, so single-shot numbers
are not worth reading.

| layers | terms | fused, 1 thread | fused, 8 threads | 16 zones | vs 1 thread | vs 8 threads |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 385k | 0.221 s | 0.263 s | 0.082 s | 2.70x | 3.21x |
| 18 | 6.9M | 6.73 s | 5.16 s | 3.29 s | 2.05x | 1.57x |

Medians of 9 runs at 12 layers and 5 at 18, every capacity pre-sized. Overlaps agree with the fused
path to ~1e-15 and term counts agree exactly.

The zones themselves are worth 3.6x at 12 layers and 2.4x at 18 over the same code on one thread
(0.292 s and 7.84 s), so most of the 18-layer shortfall is the parallel region, not the partitioning.

Worker threads that sleep between gates cost about 5%: 2376 parallel regions in an 18-layer run, and
the last task into each one starts late. Running with `JULIA_THREAD_SLEEP_THRESHOLD=infinite` takes
the 18-layer figure from 3.32 s to 3.14 s and the fused 8-thread baseline from 5.16 s to 4.80 s.

Set `n_zones` to twice the thread count. The gate permutes the zones, so they pair off; a task that
owns both halves of a pair needs no barrier, and the outbox it writes is read back by a merge on the
same core. Below `nthreads()` pairs there are not enough tasks left to fill the threads, and
`applygate!` falls back to two barrier-separated passes.

## Where the time goes

From `profile.jl` at 18 layers, and `machine.jl` for the ceilings.

Timing each task of each parallel region at 18 layers, 16 zones and 8 threads: the merge is 72% of
the busy time and the branch pass 28%, the regions are 97% of the run, and the tasks keep the eight
threads 74% busy. The quarter that is lost splits into 11% waiting for the slowest task and 14%
between the region opening and the last task starting.

That 74% is the packing, not the payoff. The same work that takes 7.84 s of CPU time on one thread
takes 22.6 s of it spread over eight, so the threads make the work itself 2.9x more expensive and
even a perfectly packed region would only reach 2.8 s. Scheduling is not what is left to win here;
traffic is. Four things limit the speedup, in the order they bite:

**Memory bandwidth, out of cache.** One thread already reads at 43-45 GiB/s; eight threads reach 51
GiB/s. Eight times the threads buy 1.15x the bandwidth. Within L3 the same test goes from 63 GiB/s on
one thread to over 140 on eight. At 18 layers the sum is 6.9M terms, 165 MB against a 16 MiB L3, so
the merge streams DRAM and extra threads have almost nothing left to win. This is the machine, not
the code, and it is why the speedup falls from 2.8x to 2.0x with size.

**Stragglers from unequal cores.** Zone *sizes* agree to 0.2%, but zone *times* differ by 18% in the
branch pass at 18 layers and by 31-62% at 12 layers. Identical compute-bound work on 8 threads
spreads by 23% (and not at all on 4), so the cores are not equally fast. Equal-size partitioning is
the wrong split on an asymmetric CPU. Pinning one thread per physical core recovers only a few
percent, so this is core asymmetry rather than threads doubling up on a core.

At the recommended `n_zones = 2 x nthreads()` the pairing leaves exactly one task per thread, so
there is no slack for a fast core to absorb a slow one's share and the slowest task alone accounts
for 84% of the region. Giving the scheduler something to work with does not help: at 18 layers,
32 zones matches 16 to within a percent, and `@threads :greedy`, which hands out pairs one at a time
instead of chunking the range, is 3% *slower* at both counts. A pair is large enough that the loss
is the core, not the assignment.

**Per-gate fixed cost.** An empty `@threads` region costs 3.9 us and a gate at 12 layers takes about
64 us, so the parallel regions alone are several percent -- but the time unaccounted for by even the
slowest zone runs to 30-43% at 12 layers, i.e. mostly task start skew. Pairing the zones removes both
barriers and is worth 1.04x at 18 layers and up to 1.24x at 12. Even at 18 layers the last task into
a region still starts 16% of the region late, which is what the sleep threshold above buys back.

**The partitioning itself.** Going from 1 to 8 zones with threading off costs 10% more total work at
18 layers and 23% at 12: per-gate fixed costs are paid once per zone, and shorter runs amortize the
XOR passes and the merge worse. Against the fused path on one thread the whole zone structure costs
16% at 18 layers and 32% at 12, and the parallelism has to pay that back before it shows a profit.

Three things that did *not* help, all measured: one `Threads.@spawn` per zone instead of `@threads`
(0.65x at 12 layers -- the region is too short to pay for a task each), more zones than threads under
`@threads` (no change, because it splits the range into contiguous chunks up front rather than
work-stealing), and `@threads :greedy`, which does work-steal (3% slower).

## The backward merge, measured and set aside

A delivered outbox can be merged in place instead of into `aux`: walking both inputs from the top, the
write pointer starts `n_tail` above the head pointer and can only fall behind it, so the result lands
in the zone's own array. Implemented and checked against the library exactly, it is worth this much:

| layers | terms | zone | forward merge | backward merge |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 385k | 0.6 MB | 0.186 s | 0.195 s |
| 18 | 6.9M | 10 MB | 4.03 s | 3.72 s |

It pays only once a zone stops fitting in cache, which is the regime it was meant for: the write into
`aux` is a cold stream that has to be fetched before it can be written, while the backward merge writes
over head lines it has just read. In cache there is nothing to recover and the aliasing costs a few
percent. Two things it needs before it is worth anything at all -- carrying both terms across the loop,
since the output aliases the head and the reloads cannot be hoisted otherwise (worth 25%), and pulling
the window back once truncation has drifted it its own width, folded into the next branch pass, which
reads every term anyway. The 18-layer figure also assumes the zone's array at twice the capacity, which
is what it can afford once `aux` no longer holds a whole merged output.

7% for a storage model that every other path has to be taught about is not the best next move, so the
zones stay single-anchored for now.
