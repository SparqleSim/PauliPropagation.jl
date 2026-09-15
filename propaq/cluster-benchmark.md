# [on CLUSTER] Threading strategy of the fused `MultiPauliSum` path

Second cluster session. The first one (below, "Previous sweep") measured the library's own paths and
found `fused multi` peaking around 8x and *losing* speed from 32 to 64 threads. This session went
after that: where the parallel efficiency is lost, what the node can actually deliver, how Propaq
scales on the same hardware, and three changes that close most of the gap.

## Setup

- **Cluster**: SLURM `standard`, job `66506936`, node `jst179`, `--exclusive`, 2 h walltime.
  (The PR #180 section below is a later session: job `66509141`, node `jst037`, same hardware.)
  Everything below was run on that node via `srun --jobid=66506936`; the login node `jed1` only saw
  file edits and `squeue`. Confirmed with `sacct` (all steps on `jst179`) and `ps`.
- **CPU / memory** (confirmed on `jst179` this time, not inferred): 2x Intel Xeon Platinum 8360Y
  @ 2.40 GHz, 72 cores (36/socket, no hyperthreading), 4 NUMA nodes (18 cores each), 503 GB RAM,
  L3 108 MiB over 2 instances.
- **Software**: Julia 1.12.5, branch `propaq-benchmark`. Propaq 0.1.5 from `propaq/.venv`,
  driven by `propaq/bench/run_propaq.py` over `circuits/6x6_steps{18,20}.json`.
- **Problem**: Propaq Fig. 1(a), 6x6 TFIM Trotterized, cutoff `1e-6`. Steps 18 / 20 / 22 give
  18.0M / 42.2M / 84.4M terms.
- **Zones**: `n_zones = nthreads` throughout unless stated. Sweeping zones from 1x to 8x the thread
  count moved the total by under 5%, so zone granularity is not a lever here (see "Ruled out").

## Summary

*Fifth session (below, "Zone workers on a laptop"): the workers as first written were 4x
**slower** than `@threads` on a machine with as many cores as threads, because Julia 1.12 runs the
main thread outside the default pool and a thread outside the pool cannot be woken through a wall
of spinners. The gate loop now runs on a pool thread when the caller is not one; 8 cores / 8
threads 30.8 -> 6.3 s (`@threads` 7.0), 64 threads unchanged (1.85 / 10.59 s at 18 / 22 steps).
Two of last session's hazards are gone with it: a task spawned inside a zone costs nothing, and
the thread-1 rule is an atomic slot. Staggering zone growth: no effect.*

*Fourth session (below, "Zone workers"): the 64-thread flatness was `@threads` launching its tasks
serially, 200 us per round, plus zones landing on remote memory. A task pinned per thread for the
whole propagation fixes both: 18 / 22 / 24 steps at 64 threads went 3.25 / 16.14 / 33.75 s ->
2.15 / 11.45 / 23.50 s, ahead of Propaq at every size.*

Three changes, all in the multi-sum machinery, no change to the propagation algorithm:

1. **`_eachzone` runs serially below `_MIN_ELEMS_PER_TASK` terms.** A round over a small sum was
   costing more in fork-join than the work itself.
2. **The branch path's `collect` and `merge` are one parallel pass instead of two.** Both are indexed
   by the owning zone and touch disjoint zones, so they were always fusible; this also merges the
   delivered tail while it is still warm. (The file header already described the path as two passes;
   it was three.)
3. **`_deliver!` reserves room for the merge's tail scratch.** Without it `_tailscratch` fell back to
   a fresh `similar()` on a quarter of all merges.

Result at 18 steps, best of 3, wall-clock seconds:

| threads | before | after | speedup after |
|---|---|---|---|
| 1 | 31.49 | 30.19 | 1.00x |
| 8 | 6.34 | 6.28 | 4.81x |
| 16 | 4.55 | 4.41 | 6.85x |
| 32 | 3.94 | 3.40 | 8.89x |
| 64 | 4.53 | 3.25 | 9.29x |

Single-threaded is unchanged (marginally faster). 64 threads is 28% faster and no longer loses to 32.
Scaling improves with problem size, as it should once the per-gate fixed cost is amortised:

| steps | terms | 1 thread | 64 threads | speedup |
|---|---|---|---|---|
| 18 | 18.0M | 30.19 s | 3.25 s | 9.3x |
| 20 | 42.2M | 78.84 s | 7.41 s | 10.6x |
| 22 | 84.4M | 189.26 s | 16.14 s | 11.7x |
| 24 | 158.4M | not measured | 33.75 s | - |

(20 and 22 with the cache pre-sized, see "The remaining lever".)

## Where the time was going

Per-gate cost against sum size, 18 steps, 1728 gates, **before** the changes. The circuit spends most
of its *gates* on small sums and most of its *time* on large ones:

| terms in sum | gates | 1 thread | 32 threads | 64 threads |
|---|---|---|---|---|
| < 10k | 589 | 0.007 s | 0.278 s | 0.499 s |
| 10k - 100k | 288 | 0.103 s | 0.106 s | 0.249 s |
| 100k - 1M | 329 | 1.415 s | 0.460 s | 0.708 s |
| 1M - 3M | 183 | 3.226 s | 0.758 s | 0.764 s |
| 3M - 10M | 222 | 12.677 s | 1.895 s | 1.922 s |
| 10M - 20M | 117 | 15.452 s | 1.899 s | 1.634 s |

The top row is the whole story of the 64-thread regression: 589 gates that cost 7 ms in total
single-threaded cost **0.5 s on 64 threads**. That is a fixed cost of ~846 us per gate, and it is
Julia's fork-join, paid three times per gate. Measured on this node with a no-op `@threads`:

| threads | fork-join (mean) |
|---|---|
| 8 | 12.5 us |
| 32 | 57.7 us |
| 64 | 208.2 us |

3 x 208 us x 1728 gates = 1.08 s, which is 19% of the 5.8 s the 64-thread run took. Changes 1 and 2
remove most of it: change 1 skips threading where it cannot pay, change 2 takes the count from three
fork-joins per gate to two.

The third change came from GC. Allocation churn was **33 GB** for a problem with a ~3 GB live set,
and GC time was flat in the thread count — 3.8 s at 1 thread, 1.9 s at 8, 1.9 s at 64 — so it was a
pure Amdahl term that grew to **17.6% of runtime at 64 threads**. Counting the call sites showed
`_tailscratch` taking its `similar()` fallback on 16,505 of 67,222 merges, 770M terms, 17.2 GB.
`_deliver!` was growing a zone to hold exactly the terms delivered, leaving the merge that follows no
room for the scratch it takes from beyond them.

## Is the merge the problem? No.

The merge dominates the profile — 70% of runtime single-threaded — and was the obvious suspect for a
sorting/merging redesign. It is not the limiter. With the allocation fixed, at 20 steps:

| phase | 1 thread | 64 threads | speedup |
|---|---|---|---|
| scan (`pass1`) | 22.93 s | 2.21 s | 10.4x |
| sort + merge | 55.68 s | 4.91 s | **11.3x** |
| total | 78.84 s | 7.41 s | 10.6x |

The sort+merge now scales *better* than the scan. Its earlier 7.6x was the per-gate allocation, not
the algorithm. The XOR-pass tail sort and the two-pointer head merge are both clean streaming
passes and they parallelise across zones as well as anything here does. **A different sorting or
merging scheme is not indicated by the evidence** and would trade the single-threaded path — which is
where we are strongest — for a ratio that does not need fixing.

## What the node can actually deliver, and why "10x copy limit" was the wrong frame

I measured the memory system directly (1.6 GB arrays, best of 3):

| threads | copy, pages on one NUMA node | copy, NUMA-local | read, NUMA-local |
|---|---|---|---|
| 1 | 29.3 GB/s | 25.1 GB/s | 6.5 GB/s |
| 8 | 79.6 | 116.9 | 50.7 |
| 16 | 83.7 | 156.8 | 97.4 |
| 32 | 79.8 | 217.6 | 171.5 |
| 64 | 58.9 | **262.2** | **272.4** |

I first read this as "parallel copy only scales 10.4x, so 20x is impossible here". **That was wrong,
and the same table disproves it**: on the same machine, the copy kernel scales 10.4x and the read
kernel scales 42x. Both saturate at ~265 GB/s. The difference is entirely the *single-thread*
number — copy already pulls 25 GB/s from one core because it prefetches perfectly, the scalar read
pulls 6.5 GB/s because it is latency-bound.

So there is no 10x machine ceiling. The real statement is:

> aggregate ceiling ~265 GB/s; **the speedup you can show is 265 GB/s divided by whatever one thread
> already achieves.**

A fast serial path mechanically produces a small speedup ratio. That is the position we are in, and
it is the right position to be in.

## Propaq on the same node

Same node, same circuit, same cutoff, both best-of-3. Propaq's own paper claims "roughly a 10-20x
speedup from one to 64 threads" on exactly this circuit (on 64 AMD EPYC 7H12 cores), which is where
the 20x target comes from.

**18 steps (18.0M terms):**

| threads | Propaq | ours | ours faster by |
|---|---|---|---|
| 1 | 53.20 s | 30.19 s | 1.76x |
| 8 | 9.98 s | 6.28 s | 1.59x |
| 16 | 6.39 s | 4.41 s | 1.45x |
| 32 | 4.14 s | 3.40 s | 1.22x |
| 64 | 3.75 s | 3.25 s | 1.15x |
| **1 -> 64 speedup** | **14.2x** | **9.3x** | |

**20 steps (42.2M terms):**

| threads | Propaq | ours | |
|---|---|---|---|
| 1 | 157.42 s | 78.84 s | ours 2.0x faster |
| 8 | 27.43 s | 14.60 s | ours 1.9x faster |
| 32 | 10.70 s | 9.18 s | ours 1.2x faster |
| 64 | 6.69 s | 7.41 s | Propaq 1.11x faster |
| **1 -> 64 speedup** | **23.5x** | **10.6x** | |

This answers the question the 20x target was really asking. **Propaq's larger speedup ratio is mostly
a slower starting point**: they are 1.8-2.0x slower than us on one thread, so the same absolute
finish line is a much bigger ratio for them. At 18 steps we are faster than Propaq at *every* thread
count including 64. At 20 steps we are ~2x faster up to 8 threads, still ahead at 32, and 11% behind
at 64.

We could "reach 20x" tomorrow by making the single-threaded path 2x slower. It would be a worse
library. **The 11% at 64 threads on the large problem is the only real gap, and that is the number
worth chasing** — not the ratio.

## At 22-24 layers we are still behind (and the gap widens)

64 threads only, one run each, ours with the cache pre-sized. Propaq generated its own 22/24-step
circuits from the same `ising_trotter_problem`.

| steps | our terms | Propaq terms | ours | Propaq | |
|---|---|---|---|---|---|
| 18 | 18.0M | 19.4M | **3.25 s** | 3.75 s | ours 1.15x faster |
| 20 | 42.2M | 46.7M | 7.41 s | **6.69 s** | Propaq 1.11x faster |
| 22 | 84.4M | 96.3M | 16.14 s | **14.57 s** | Propaq 1.11x faster |
| 24 | 158.4M | 181.8M | 33.75 s | **27.35 s** | Propaq 1.23x faster |

So the crossover sits between 18 and 20 steps, and past it the gap grows rather than settling: 11% at
20-22 steps, 23% at 24. Both engines double their runtime per two Trotter steps, so this is a level
difference, not a divergence in complexity.

Two caveats, pulling in opposite directions:

- **Propaq retains 10-15% more terms at every size**, so those times cover somewhat more work. Their
  runs also report a separate `terms_below_cutoff` (2.7M of 19.4M at 18 steps), so it is not settled
  whether `n_terms` is directly comparable to ours. Not resolved here — worth pinning down before
  quoting a per-term figure either way.
- **Our column is our best configuration, theirs is their default.** Ours needs the `resize!`
  pre-sizing described below; Propaq needs no tuning to hit its number.

64 threads is genuinely our best setup at this size — 32 threads gives 36.81 s against 33.75 s, i.e.
doubling the threads buys 9%. That flatness at the top end, not the single-thread speed, is what the
remaining work has to attack.

## Zone workers: why 64 threads was flat, and the fix (job `66509731`, node `jst037`)

Fourth session, same hardware. Question: why does `fused multi` gain only 9% from 32 to 64 threads
when Propaq gains more, and what is the simplest thing that fixes it. The answer turned out to be
one mechanism, measured three ways, and one contained change: the zones are now worked by a task
pinned to every thread for the whole propagation (`src/Base/MultiSum/zoneworkers.jl`) instead of
by `@threads` twice per gate.

Everything here was run on `jst037` via `srun --jobid=66509731`; `/tmp` is node-local, so the
scripts live in `~/.ppscratch/s2/` (`probe.jl` is the instrumented run, `forkjoin.jl`,
`imbalance.jl`, `ntstore.jl` the microbenchmarks).

### Result

64 threads, 64 zones, no pre-sizing, best of 3, wall-clock seconds; "before" is the code at the
top of this file (with the cache pre-sized, its best configuration):

| steps | terms | before | zone workers | Propaq | |
|---|---|---|---|---|---|
| 18 | 18.0M | 3.25 | **1.92** | 3.75 | ours 1.95x faster |
| 22 | 84.4M | 16.14 | **10.92** | 14.57 | ours 1.33x faster |
| 24 | 158.4M | 33.75 | **21.88** | 27.35 | ours 1.25x faster |

The workers alone gave 2.15 / 11.45 / 23.50 s; the rest is the in-place box merge below. Other
thread counts, 18 steps: 8 threads 6.28 -> **4.51** s; 24 steps at 32 threads 36.81 -> **26.90** s
(workers alone). From one thread (30.2 s, unchanged by construction: no workers are started) that is a
14x speedup at 18 steps. `Pkg.test()` passes; the fused vector and fused multi paths agree on
every term and digit checked.

### Where the round went

`probe.jl` times every zone's task in both passes of every gate and splits the wall time of a round
into three parts: the *mean* work per thread, the *imbalance* (slowest zone minus mean), and what is
left, which is the fork and the join. 18 steps, 64 threads, cache pre-sized, `@threads` as before:

| pass | wall | mean work | imbalance | fork-join |
|---|---|---|---|---|
| 1, scan | 0.99 s | 0.47 s | 0.15 s | **0.38 s** |
| 2, deliver + sort + merge | 1.97 s | 1.23 s | 0.42 s | **0.32 s** |

Of 3.2 s, 1.7 s is not work. The fork-join is ~200 us per round, and it is not a barrier cost:
`@threads` starts its tasks *one after the other* from the calling thread, ~3 us each, so with 64
threads the median task begins 100 us after the fork and the last one at 200 us. Measured with a
no-op body, microseconds per round on 64 threads:

| mechanism | per round | note |
|---|---|---|
| `@threads` (dynamic) | 209 | median task starts at 103 us, last at 207 us |
| `@threads :static` | 248 - 364 | 106 with `JULIA_THREAD_SLEEP_THRESHOLD=infinite` |
| `@spawn` x 64 + `wait` | 181 | |
| pinned workers spinning on a counter | **15** | |

That is the whole 8 / 32 / 64 -> 12 / 58 / 208 us table of the second session: linear in the
thread count because the launch is serial. With ~3500 rounds per run it is 0.7 s at 64 threads,
and in a real run threads also fall asleep in the serial gaps and cost more to wake, so
`@threads :static` measured 0.6 s per pass here (a median 450 us until the last task started).

### The work itself depends on which thread gets the zone

Pinning zone `z` to thread `z` (`@threads :static`) cut the *work* by 24-34% (pass 1 mean
0.47 -> 0.36 s, pass 2 1.23 -> 0.81 s) but its slower launch ate the gain, which is why the second
session saw it as a wash. The work gets faster because a zone's arrays then live on the memory of
the node whose thread wrote them (first touch; `numa_balancing` is on and helps only if the access
pattern is stable). Under dynamic scheduling a zone ran on a different CPU than the round before
61% of the time, so three quarters of its traffic crossed the socket. Thread pinning on top of the
static assignment is worth ~4% (`pin=numa` 10.57 s vs unpinned 10.98 s at 22 steps): the OS keeps
threads put on its own (152 migrations in 2936 rounds), so no `ThreadPinning` dependency is needed.

### The imbalance is the machine, not the zones

`imbalance.jl` records all 64 zone durations of pass 2 for every gate over 3M terms. The work per
zone is perfectly balanced -- max/mean of `n + 2 * delivered` is 1.006, and duration does not
correlate with it (r = -0.07) -- yet the slowest zone takes a median 1.21x the mean. With threads
pinned in order, the ten threads that landed alone on NUMA node 3 ran **20% faster** (0.76-0.85 of
the mean) than the eighteen on each other node. So per-thread throughput is the share of *node-local*
bandwidth, and pass 2 is bandwidth-bound per node once the data is local. Spreading threads evenly
over the nodes (16 each) evens it out; the rest of the imbalance is GC pauses landing on one zone.

### The fix

`withzoneworkers(f)` starts a task per thread, pinned with the same `jl_set_task_tid` call
`@threads :static` uses, for the duration of `f`; `_propagate!` wraps its gate loop in it for any
multi-sum target. `_eachzone` publishes the round's zone function, bumps an atomic counter, works
its own stripe on the calling thread, and waits for the others. Zone `z` always goes to thread
`(z - 1) % nthreads + 1`. Both spin loops pass a `GC.safepoint()`, without which a collection
triggered by a worker deadlocks. There is no fallback to `@threads` inside the loop; there is one
around it: no workers on one thread, inside a `@threads` region (where a pinned task could not run),
from a thread other than 1, or for a task that is not the one that opened them, so concurrent
propagations from several tasks do not share a set of workers.

After the change, 22 steps at 64 threads: fork-join 0.06 s in total, mean work 2.56 + 6.90 s,
imbalance 0.78 s, and both passes move ~225 GB/s of logical traffic (scan 638 GB in 2.8 s, pass 2
1.72 TB in 7.5 s). 32 threads moves ~200 GB/s.

### What the workers must not do: spin forever

The first version spun forever and deadlocked the test suite. `min_rel_coeff` calls
`maxabscoeff` on the multi cache between gates, which reduced every zone through AcceleratedKernels
and so `@spawn`ed tasks -- onto default-pool threads that the workers held. (It passed once by
luck: Julia 1.12 starts `-t8` as one interactive plus eight default threads, and the first version
left one default thread free, so the stray tasks ran there, slowly.) Yielding from the spin does
not help either: a yielded task goes back onto its own thread's queue, which the scheduler looks at
before the global one. So a worker now spins for a bounded window and then sleeps on a condition,
and the round wakes only the workers that are asleep (`sleeping` counter). The window has to be
long: a worker that dozes off while waiting for the slowest zone of a round costs a few
microseconds to wake, per worker, serially from the calling thread -- exactly the `@threads` cost.
With a 200 us window the 18-step run went from 1.99 to 2.26 s; at 20 ms it does not sleep
between rounds. `maxabscoeff` on a multi cache now reduces its zones through `_eachzone` with
no task inside a zone, so nothing in the library spawns while the workers are up; a user's gate
that does would wait for the window, not forever.

### Growth is a 64-thread hazard

The in-place box merge (below) first grew a zone to `1.5 * n_new` where `_deliver!` had grown it
to `1.5 * (n_new + n_tail)`. That alone moved 18 steps at 64 threads from 1.92 to **2.78 s**, with
the cache pre-sized both 1.84 s. The zones are hash-balanced, so all 64 cross a capacity threshold
on the same gate and reallocate together, and every such gate pays a collection. The larger
headroom is back in `xorsortedboxmerge!`. Sweeping the geometric step there (`growth.jl`, 18
steps, 64 threads, best of 3):

| step | zone growths | wall |
|---|---|---|
| 1.5x | 642 | 1.98 s |
| 2x | 464 | 1.80 s |
| 3x | 349 | 1.84 s |

About **1 ms of wall time per growth event** at 64 threads, so growing every zone once to a
projected peak -- the second session's suggestion -- is worth ~0.5 s of 1.98 s here, a quarter. A
bigger step buys most of it for up to a third more peak memory, which is the library's weak side
against Propaq; left at 1.5x, the projection is the better route.

### The deliver copy is gone

`_collectbranch!` no longer copies the box onto the zone's tail before sorting it: `xorsortedboxmerge!`
runs the XOR passes with the box as one ping-pong buffer and the room past the active terms as the
other, so the tail is moved by the sort alone, and the merge reads it from whichever buffer it
landed in. That also retires the merge's scratch beyond `n_new`. 22 steps at 64 threads: 11.45 ->
**10.92 s**; pre-sized 18 steps: pass 2 work 0.84 -> 0.74 s.

### What it would help on a laptop

Confining 16 threads to one NUMA node is the nearest thing to a laptop here: one memory domain, no
hyperthreading. 18 steps: 7.42 -> 6.56 s (-12%); 8 threads 7.95 -> 7.13 s (-10%). Half of that is
the launch and wake-up, the other half is a thread always touching the same zone. So about a tenth,
against a third on 64 threads.

### What is left: bytes

At 64 threads the passes now run at the node's streaming ceiling for their access pattern. Measured
NUMA-local at 64 threads, 100M `UInt128` elements:

| kernel | regular stores | non-temporal stores |
|---|---|---|
| one stream copied (16 B) | 255 GB/s | 300 GB/s |
| two streams written interleaved (16 B + 8 B, as the merge writes) | 189 GB/s | 231 GB/s |

The regular copy's 255 GB/s logical is ~390 GB/s physical once the read-for-ownership of every
written line is counted, which is the theoretical peak of the node's DDR4. So further gains at
this thread count come from moving fewer bytes per gate, and each option has a measured or
computed size. Per gate over `n` terms and `t = 0.2n` products at 24 B per term:

| traffic per gate | bytes | share | how to cut it |
|---|---|---|---|
| scan reads terms + coefficients | 24n | 22% | -- |
| merge reads head and tail, writes both | 48n + 48t | 53% | in-place merge (needs an offset representation of the zone); non-temporal stores measured, see below |
| deliver copies the box onto the tail | 48t | 9% | done, see "The deliver copy is gone" |
| XOR passes, ~1.4 per gate | 67t | 12% | -- |
| products written to the box | 32t | 6% | -- |

Nothing here is as simple as the workers, and each is worth 10% or so. The cutoff, the term
width and the coefficient width are the only things that change the 24 B.

**Non-temporal stores do not help, though**, and that says something about the merge. Replacing
`_writeandadvance!` with `llvmcall` streaming stores for `UInt128` / `Float64` (`ntmerge.jl`, which
covers the scan's products and the merge's output) made 22 steps at 64 threads *slower*: 11.74 ->
12.31 s, same terms and value. The pure streaming ceiling rises by 20% with them, so if the merge
were bound by the read-for-ownership of its output it would have shown. It is not: the two-pointer
loop's data-dependent branch and its four read streams leave the write path enough slack that
bypassing the cache only costs. So the merge at 64 threads is bandwidth-bound in the sense that
its *reads* are, and the traffic to cut is what it reads: the head, twice per gate (scan and
merge), and the tail, three to four times (box, deliver, XOR passes, merge).

### Ruled out this session

- **Non-temporal stores in the write path**: slower, see above.

- **Thread pinning by itself** (`pinthreads(:cores)` under dynamic scheduling): 3.57 s vs 3.21 s,
  worse, since it pins threads but not zones.
- **Zone count above the thread count** with the workers: no reason to; a thread's zones are its
  own either way.

## Zone workers on a laptop: the caller has to be in the pool (job `66511767`, node `jst172`)

Fifth session, same hardware, a different question: do the workers hold up against the library's
principles -- no manual pre-sizing, no power-of-two thread count, no 64 cores, and above all no
catastrophe for a user who runs `julia -t auto` on a laptop. The laptop was simulated with
`taskset`: 8 cores of one NUMA node (one memory domain, no hyperthreading) for `-t 8`, six for
`-t 6`. Scripts in `~/.ppscratch/s3/` (`lap.jl` runs the library under one of several round
mechanisms, `busyround.jl`, `mainidle.jl`, `stray.jl`, `interrupt.jl`).

### The catastrophe, and its mechanism

Julia 1.12 starts `-t 8` as one interactive thread plus eight default threads, and the main thread
is the interactive one: `threadpooltids(:default) == 2:9`, `threadid() == 1`. `@threads` uses the
eight default threads and the main thread waits. Last session's workers pinned a task to every
default thread *and* worked on the main thread too: nine busy threads for `-t 8`, which on 72 cores
was invisible and on 8 cores is one thread too many. 18 steps, 8 threads, best of 2:

| cores | mechanism | wall |
|---|---|---|
| 8 | last session's workers (main thread works and spins) | **30.8 s** |
| 8 | main thread sleeps on an event every round | 29.7 s |
| 8 | ... and the workers `sched_yield` while they spin | 10.0 s |
| 8 | `@threads` | 7.0 s |
| 8 | `@threads :static` | 7.3 s |
| 9 | last session's workers | 6.3 s |
| 8 | **gate loop on a pool thread** (the fix) | **6.3 s** |
| 8, `-t 8,0` | main thread in the pool, no hop needed | 6.3 s |

The first guess -- the main thread's serial part of each gate fights a spinner for a core -- was
wrong: timed inside the run, the serial parts are 0.02 s of 10 s, and all of the excess is inside
the parallel rounds. `mainidle.jl` shows the main thread does sleep properly in `wait` (0.04 ms of
CPU in a 5 ms wait). What is slow is *waking it*: a thread outside the pool, asleep in the OS,
becomes runnable at the end of every round on a machine whose every core has a spinning worker on
it, and CFS gives it the core a scheduler slice later. `busyround.jl`, 8 workers each busy for
2 ms per round, 8 cores:

| caller | spin | round |
|---|---|---|
| main thread outside the pool, sleeps per round | `pause` | 16.4 ms |
| main thread outside the pool, sleeps per round | `pause` + `sched_yield` | 5.2 ms |
| main thread in the pool (`-t 8,0`), works and spins | either | **2.12 ms** |
| main thread outside the pool, 9 cores | `pause` | 2.25 ms |

So a per-round handshake with a thread outside the pool costs 3-14 ms per round on a full machine,
and nothing at all when the driving thread is one of the pool. `sched_yield` in the spin helps
(it is what Intel's OpenMP runtime does in its default "throughput" mode) but does not cure it.
`--gcthreads=1` changed nothing; the 8 GC threads are not involved.

### The fix

`withzoneworkers(f)` now runs `f` itself on a task pinned to the first pool thread whenever the
calling thread is not one of the pool (or the calling task is not sticky, i.e. was `@spawn`ed and
could move), and the caller waits once for the whole propagation instead of once per round. The
driving task works its share of every round and spins for the rest, as the main thread did on
Julia 1.11 where it *is* pool thread 1; on 1.11 or with `-t 8,0` nothing is moved. What this
changes for `f`: it runs on a different task (`threadid() != 1`; a task-local RNG seeded from the
caller's, as any child task), an exception inside it is rethrown as itself (tested: a short
parameter vector comes back as `ArgumentError`), and Ctrl-C at the waiting caller is passed on
through the workers' stop flag, which the driving task checks at every round: `interrupt.jl`
interrupts an 18-step run 0.18 s after the signal, leaves no workers behind, and the next run is
at full speed.

Everything else got simpler in the process. The `sleeping` counter is gone (the round takes the
condition's lock every time, tens of nanoseconds); the "only from thread 1" rule, which doubled as
the check-and-set of the global, is an `@atomicreplace` on a slot plus the sticky-or-move rule;
the spin loop yields the core to the OS (`sched_yield`, `SwitchToThread` on Windows), measured
free at 64 threads (1.85 vs 1.82 s with `pause` alone).

Results, best of 2-3, 18 steps unless noted:

| machine | threads | mechanism | wall |
|---|---|---|---|
| 8 cores | `-t 8` | workers | **6.34 s** (`@threads` 6.97) |
| 6 cores | `-t 6`, 16 zones | workers | **7.74 s** (`@threads` 8.34) |
| node | `-t 64` | workers | **1.85 s** (last session 1.92; old code on this node 2.20) |
| node | `-t 64`, 22 steps | workers | **10.59 s** (last session 10.92, Propaq 14.57) |
| node | `-t 64`, 24 steps | workers | **22.06 s** (last session 21.88, Propaq 27.35) |

Six threads is the non-power-of-two case: `defaultnzones()` gives 16 zones and the stripes are
3,3,3,3,2,2, which is 89% of the threads busy, and the workers still beat `@threads` by 7%. The
smallest machines, 16 steps: 2 cores / `-t 2` workers 5.29 s (`@threads` 5.92), 3 cores / `-t 3`
(8 zones) 4.21 s (`@threads` 4.59). Hyperthreading could not be simulated here (one thread per
core); a laptop with `-t auto` on 16 logical cores is untested.

### A task spawned inside a zone costs nothing now

Last session found that anything `@spawn`ed while the workers are up waits for a worker to sleep
(20 ms), because the first version of `maxabscoeff` did exactly that. Re-measured with the
mistake put back on purpose (`stray.jl`: the threaded reduction inside every zone, called every
gate through `min_rel_coeff`), 16 threads, 16 steps: workers **2.10 s**, `@threads` 2.45 s. No
stall at all: a worker that waits for tasks it spawned yields *its own* thread to them, and so
does the driving task now that it is a pool thread. The 20 ms stall was specific to spawning from
the interactive main thread, which cannot run default-pool tasks -- the very thread the hop takes
out of the loop. What remains of the hazard: a task started by *some other* task in the process
waits up to the spin window for a thread. `findspawn.jl` over the multi-sum tests confirms the
library starts no task from the driver while the workers are up.

### Staggering zone growth does nothing

"Growth is a 64-thread hazard" (above) measured ~1 ms of wall per zone growth and guessed that 64
zones reallocating on the same gate was the cost. Tested by giving zone `z` a growth factor of
`1.5 + 0.5 (z-1)/n` so that the zones cross their thresholds on different gates, 32 threads on
two NUMA nodes, 18 steps, best of 3:

| growth | wall |
|---|---|
| 1.5x, all zones | 2.88 / 2.90 s |
| 1.5x .. 2.0x by zone | 2.85 s |
| 1.25x .. 1.75x by zone | 2.90 s |
| 2.0x, all zones | 2.69 s |

Within noise for the staggered versions, so the cost is per growth event and not the storm, and
the only lever is fewer events, which is more memory (2x: +33% peak for 7%). Reverted; the code
grows by 1.5x as before, automatically, and nothing relies on a manual `resize!`.

### What is still true

- Two propagations from two tasks at once: the second falls back to `@threads`, whose tasks wait
  up to the spin window for a worker to sleep. It finishes, slowly. Not a target.
- More busy threads than cores in *other* ways (`-t 72` on 72 cores now costs nothing extra since
  the main thread sleeps; two Julia processes on one machine does) meets the spinners like it
  would meet an OpenMP runtime with `KMP_BLOCKTIME` set: the `sched_yield` softens it, and the
  spin window bounds it.
- Both test-suite runs pass (5327 tests) with the final code.

## PR #180 (single-pass fused apply) on this node

Third cluster session, job `66509141`, node `jst037` (same model as `jst179`: 2x Xeon 8360Y, 72
cores, 4 NUMA nodes, 503 GB). Question: does the PR speed up this benchmark or its thread scaling,
and is there a better version of the idea?

### What the PR changes, and what it does not

The PR replaces the **dry run + write** shape of the fused rotation apply on `VectorPauliSum` with
**one pass + compact**: each task writes its products into its own stretch of the aux arrays, and a
parallel block copy then moves the stretches onto the end of the main arrays.

It does not touch the `MultiPauliSum` path this benchmark is about. `_fusedbranchzone!` in
`fused_multi.jl` already has the PR's shape and always did: a zone runs one pass and writes its
products into the outbox of the zone that owns them, and `_deliver!` is the compaction. **On
`fused multi` the PR is a no-op by construction**, which the numbers confirm. What it does change is
the `fused` single-`VectorPauliSum` path, which is the one that scales badly here (3.3x from 1 to 64
threads, against 9.3x for `fused multi`).

### Methodology: pair the A and B runs

The first sweeps compared A and B as separate processes minutes apart and were useless: the same
unmodified code measured 8.88 s and 8.40 s at 64 threads on two sweeps, and 8.31 s and 8.83 s on
two others. **Drift on this node over tens of minutes is +-6%, the same size as the effect.** Every
number below instead comes from one process that alternates the two shapes round by round (a
`Ref{Bool}` selects which), best of 3 rounds each, so drift hits both equally. Both shapes are
checked to produce the same expectation value in the same process.

### Result: the PR wins everywhere, and the win grows with the width of the Pauli type

| problem | Pauli type | terms | threads | dry run + write | one pass + compact | |
|---|---|---|---|---|---|---|
| 6x6 TFIM, 18 steps | `UInt128` (16 B) | 18.0M | 8 | 11.887 s | **11.423 s** | -3.9% |
| 6x6 TFIM, 18 steps | `UInt128` | 18.0M | 64 | 9.059 s | **8.937 s** | -1.3% |
| 9x9 TFIM, 12 steps | `UInt192` (24 B) | 11.0M | 8 | 12.604 s | **11.269 s** | -10.6% |
| 9x9 TFIM, 12 steps | `UInt192` | 11.0M | 64 | 9.320 s | **8.644 s** | -7.3% |

So: a real win, worth merging, and **the width dependence reported on the PR reproduces here**. It is
not a weight-cap effect -- neither problem has a weight cap.

### Why, in terms that predict rather than describe

Per gate, over `n` terms of `T` bytes with 8-byte coefficients, producing `p` products
(branch fraction `b = p/n`):

- **dry run + write** reads the terms twice, makes every product twice, writes each product once,
  straight to its final place: `T*n + (T+8)*n` read, `(T+8)*p` written.
- **one pass + compact** reads the terms once, makes every product once, then writes each product
  twice and reads it once: `(T+8)*n + (T+8)*p` read, `2*(T+8)*p` written.

The single pass therefore always saves one making of every product, and it moves fewer bytes exactly
when `2*(T+8)*b < T`, i.e.

> **the single pass is cheaper in traffic iff the branch fraction is below `T / (2(T+8))`**
> -- 0.25 for `UInt64`, 0.33 for `UInt128`, 0.375 for `UInt192`, 0.4 for `UInt256`, rising to 0.5.

Both terms grow with `T`, which is the whole width story: a wider Pauli string makes the re-read
and the re-making of the product more expensive, while the products that have to be staged are
the same count. It also explains the PR's own caveat that dense sums where half the terms branch can
be a few percent slower: `b = 0.5` is above the threshold for every width.

The measured phase split says the same thing, in seconds:

| problem | threads | dry run | write | sum | one pass | compact | sum |
|---|---|---|---|---|---|---|---|
| 6x6 `UInt128` | 8 | 1.583 | 1.858 | 3.441 | 2.302 | 0.832 | **3.134** |
| 6x6 `UInt128` | 64 | 1.127 | 1.316 | 2.443 | 1.746 | 0.605 | **2.351** |
| 9x9 `UInt192` | 8 | 2.594 | 1.964 | 4.558 | 2.946 | 0.628 | **3.574** |
| 9x9 `UInt192` | 64 | 1.861 | 1.404 | 3.265 | 2.217 | 0.544 | **2.761** |

The dry run costs **1.65x more** going from `UInt128` to `UInt192` (1.127 -> 1.861 s at 64 threads),
close to the 1.5x the width alone predicts. The compaction over the same step gets *cheaper*
(0.605 -> 0.544 s), because it moves products and there are fewer of them (632.5M vs 286.6M). That is
the whole asymmetry.

### The two thirds still on the table

The single pass is **not** free: it costs more than the write it replaces -- 1.316 -> 1.746 s at
`UInt128`/64 threads, 1.404 -> 2.217 s at `UInt192`/64. Same products, same bytes written, different
destination. So the apply phase decomposes into three numbers, and at `UInt192` / 64 threads:

| | apply phase |
|---|---|
| dry run + write | 3.265 s |
| one pass + compact | 2.761 s |
| one pass writing straight to its final place | **1.404 s** |

The third row is not hypothetical: it is the measured `write` pass of the dry-run shape, which by
construction writes every product where it belongs. (It is a mild underestimate, since it runs with
the terms freshly read by the dry run.) **The PR captures about a third of what removing the dry run
is worth; the rest is the staging copy and the cost of writing to a staging area at all.**

Why writing to aux should be dearer than writing to `main` past `n_old`, for identical bytes, is a
question the full runs cannot answer, so I built the component benchmark for it: one prebuilt sorted
sum, one `PauliRotation([:X,:X],[1,2])`, the same branch pass run with the products sent to different
places, best of 5 (`~/.ppscratch/staging.jl`). Random terms make half of them branch, so `b = 0.5`
here, above the real propagation's 0.19. Every destination page is touched beforehand by the task
that will write it, so first-touch NUMA placement cannot be the answer.

**It is not the array, and it is not the layout. It is the offset.** With the destination fixed at
`n_old + ...` in either array, all four combinations of read-from / write-to are within 2%:

| 40M `UInt128` terms, 64 threads | read main | read aux |
|---|---|---|
| write main past `n_old` | 0.0212 s | 0.0212 s |
| write aux past `n_old` | 0.0215 s | 0.0210 s |

Move the destination back to the offsets the pass is *reading* from, and it collapses. Sweeping the
staging base, reading `main[r.start:r.stop]` and writing from `base + r.start` in aux:

| staging base | 64 threads | vs writing to main past `n_old` |
|---|---|---|
| 0 -- **this is what the PR does** | 0.0342 s | **+45%** |
| 4096 | 0.0334 s | +42% |
| n/64 | 0.0339 s | +44% |
| n/8 | 0.0321 s | +36% |
| n/2 | 0.0279 s | +18% |
| n (past `n_old`) | **0.0207 s** | **-12%** |

Dense versus per-task-start layout within a base makes no difference (0.0336 vs 0.0336 at base 0).
On 8 threads the whole effect is +7% instead of +45%: it is 64 read streams and 64 write streams at
matching offsets in two equally sized allocations, and a small shift does not fix it -- the penalty
falls off smoothly with distance, so it is not a single cache-set collision but the two stream sets
sharing the same part of the address space.

**The practical upshot for the PR**: staging at `rng.start` is the worst offset measured, and staging
at `n_old + rng.start` is not only free but 12% *faster* than writing straight to main. That is worth
roughly the entire pass penalty in the full runs -- 0.43 s at `UInt128`/64 threads and 0.81 s at
`UInt192`/64 -- which would take the PR from -1.3% / -7.3% on total runtime to something
substantially better, and would close most of the gap to the "straight to its final place" row above.
The catch is capacity: staging past `n_old` needs aux room for `n_old + p` at pass time, and only
`n_old` is guaranteed. Cheap ways out are to stage past `n_old` when the capacity is already there
(in steady state it usually is, since growth is geometric and the merge's `_tailscratch` wants room
beyond `n_new` anyway) and fall back to `rng.start` when it is not.

### Where the copy could go instead

Both shapes write the products once into a staging area and copy them once into place. The copy
exists only because the destination offset is unknown while the pass runs. Three ways out, in
increasing order of how much they buy:

1. **Partition the sum.** Then there is no global offset to agree on: each part appends to its own
   tail. This is exactly what `MultiPauliSum` does, which is why its fused path never had a dry run.
   It does still pay the copy, as `_deliver!`. Measured over a whole 18-step run at 64 threads, in
   thread-seconds: scan 35.7, **deliver 8.8**, merge 75.6 -- the delivery copy is **7.3%** of the
   parallel work (6.3% at one thread), moving 633M terms / 28.3 GB per run.
2. **Absorb the copy into the tail sort.** Right after the apply, `xorsortedtailmerge!` reads the
   tail and permutes it into scratch -- it copies the tail *again*. A first XOR pass that read the
   staging blocks in place would make the compaction free. The obstacle is that an XOR pass's blocks
   are cut by term values, not by task boundaries, so this is not a small change.
3. **Do not stage at all**, which needs the offsets in advance, which needs the count -- the dry run.
   Closing the circle: the choice is genuinely between paying a count and paying a copy, and the
   cost model above says which is cheaper for a given width and branch fraction.

### Ruled out here

- **Typing "no weight cap" so the dry run stops making products.** With `max_weight` reaching
  `_truncateweight` as a `Float64`, a dry run under no cap still computes every product, because
  `isinf` is a runtime test. Passing `nothing` instead makes the test fold away at compile time and
  the product with it. Measured: no improvement, if anything slightly worse, on both `fused` and
  `fused multi` -- so the dry run is not limited by making the products. Reverted. (Worth keeping in
  mind for a *capped* wide-type problem, which this is not.)

## The remaining lever: allocation from array growth

Change 3 removed the merge-scratch allocations but not the churn from growing the zone arrays
themselves. Reserving the cache before propagating shows what is still on the table (20 steps,
64 threads):

| | wall | GC | allocated |
|---|---|---|---|
| after changes 1-3 | 9.48 s | 1.28 s | 24.3 GB |
| plus `resize!(cache, 100_000_000)` | **7.41 s** | 0.32 s | 8.6 GB |

**That is a further 22%, and it is available today with no code change** — build the cache, `resize!`
it above the expected peak, and call `propagate!`:

```julia
cache = PropagationCache(MultiPauliSum(VectorPauliSum(obs), n_zones))
resize!(cache, 100_000_000)          # above the expected peak term count
Performance.propagate!(circuit, cache, thetas; min_abs_coeff=1e-6)
```

Worth documenting as a tuning knob regardless. To get it automatically, the two candidates are:

1. **Stop carrying `flags` and `indices` per zone.** `resize!(::VectorPauliPropagationCache, n)`
   grows four arrays; the fused path never touches `flags` (1 B/term) or `indices` (8 B/term). That
   is ~16% of the bytes reallocated on every growth event, for arrays the fused-multi path does not
   read. Allocating them lazily would cut the churn and the peak footprint.
2. **A less timid growth factor for zones.** 1.5x means many growth events, each copying the whole
   zone. The zones are hash-balanced and the peak is predictable from the previous gate's growth
   rate, so a zone could grow to a projected peak rather than creeping toward it.

Beyond that, the one structural idea worth considering: because the zone map is linear, a fixed
`⊻ mask` pairs every zone with exactly one partner (`_xortarget` is an involution). One task per
*pair* could scan both zones, cross-write, and merge both with **no barrier at all**, taking the
remaining two fork-joins per gate to one. Worth ~0.2-0.4 s per run at 64 threads on the 18-step
problem — real, but smaller than the allocation items, and a bigger change.

## Ruled out (measured, no effect)

Recorded so nobody spends the time again:

- **NUMA placement.** `numactl --interleave=all` and `JULIA_EXCLUSIVE=1` each moved the 18-step
  runtime by under 3% at 32 and 64 threads. Zone arrays grow on worker threads, so their pages are
  already spread.
- **`@threads :static`** for stable zone-to-thread affinity: within noise (3.97 s vs 3.98 s at 32
  threads; worse at 64), and it makes nesting inside a user's own `@threads` a hard error. Reverted.
- **`JULIA_THREAD_SLEEP_THRESHOLD=infinite`**: no change to fork-join cost. The cost is task spawn
  and join, not waking sleeping threads.
- **Zone count.** 1x to 8x the thread count moved the 18-step total under 5% (32 threads: 3.94 s at
  32 zones, 3.75 s at 256 zones). Zone sizes are hash-balanced, so there is no imbalance to fix.

## Validation

`Pkg.test()` passes (5338 tests) after the changes. `propaq/bench/benchmark.jl` cross-checks the term
count of every path against the first and errors on disagreement; all paths agree.

## Next

- Bytes per gate are the remaining lever at 64 threads (see "What is left: bytes"): the reads,
  not the writes: the head twice per gate, the tail three times.
- Zone growth costs ~1 ms per event at 64 threads and staggering it does not help (see
  "Staggering zone growth does nothing"); fewer events means more memory. Growing to a projected
  peak is the only route left, at the same memory cost as the peak itself.
- Take PR #180's single pass, and move its staging base off `rng.start` -- see "The two thirds
  still on the table". Measured worth: up to 45% of the pass at 64 threads.
- Decide on the two allocation items above; they are worth ~22% at 64 threads.
- Document `resize!` + `propagate!` as a tuning knob in the `MultiPauliSum` README.
- Push past 84M terms. This session reached 84.4M; the memory headroom (503 GB, and we used ~16 GB
  at 84M) allows several hundred million, it was walltime that ran out.
- monoprop is still not wired into this comparison.

---

## Previous sweep (job `66503229`, node `jst003`)

Kept for reference. Best of 3, one Julia process per thread count, zones = thread count. These are
the numbers the changes above were measured against; `fused multi` is the column of interest.

### 1 thread
| steps | gates | terms | VectorPauliSum | fused | fused multi | indexed | indexed multi |
|---|---|---|---|---|---|---|---|
| 10 | 960 | 188,082 | 0.394 s | 0.136 s | 0.146 s | 0.084 s | 0.085 s |
| 14 | 1,344 | 2,230,398 | 8.397 s | 2.949 s | 2.971 s | 2.475 s | 2.480 s |
| 18 | 1,728 | 18,044,878 | 91.270 s | 29.595 s | 30.733 s | 33.509 s | 33.640 s |

### 8 threads
| steps | gates | terms | VectorPauliSum | fused | fused multi | indexed | indexed multi |
|---|---|---|---|---|---|---|---|
| 10 | 960 | 188,082 | 0.539 s | 0.171 s | 0.091 s | 0.088 s | 0.074 s |
| 14 | 1,344 | 2,230,398 | 4.045 s | 1.458 s | 0.802 s | 2.524 s | 1.001 s |
| 18 | 1,728 | 18,044,878 | 32.417 s | 11.786 s | 6.379 s | 33.572 s | 12.645 s |

### 32 threads
| steps | gates | terms | VectorPauliSum | fused | fused multi | indexed | indexed multi |
|---|---|---|---|---|---|---|---|
| 10 | 960 | 188,082 | 0.597 s | 0.201 s | 0.204 s | 0.085 s | 0.127 s |
| 14 | 1,344 | 2,230,398 | 3.467 s | 0.972 s | 0.701 s | 2.532 s | 0.548 s |
| 18 | 1,728 | 18,044,878 | 25.009 s | 9.356 s | 3.644 s | 32.975 s | 4.801 s |

### 64 threads
| steps | gates | terms | VectorPauliSum | fused | fused multi | indexed | indexed multi |
|---|---|---|---|---|---|---|---|
| 10 | 960 | 188,082 | 0.669 s | 0.174 s | 0.655 s | 0.086 s | 0.414 s |
| 14 | 1,344 | 2,230,398 | 3.432 s | 1.022 s | 1.296 s | 2.566 s | 0.859 s |
| 18 | 1,728 | 18,044,878 | 24.470 s | 8.927 s | 4.274 s | 33.997 s | 3.909 s |
