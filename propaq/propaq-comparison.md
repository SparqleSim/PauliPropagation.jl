# propaq vs. PauliPropagation.jl — design differences and what to take

Study of the `propaq` Rust/Python Pauli-propagation package, focused on the propagation
algorithm itself: how it selects branching terms, how it stores and merges them, how it
truncates, and how it threads. Other-basis support (Majorana), the plugin ABI, surrogate
mode, extrapolators, and general modularity are deliberately out of scope.

| | |
|---|---|
| propaq | `hkbelagali/propaq` @ `0086ea1` (2026-09-08), v0.1.5 |
| PauliPropagation.jl | `multi-vector` @ `24cce0b` (2026-09-10), v0.8.2 |
| Machine | 24 logical cores, Linux, Julia 1.12.6 |

propaq's own source credits its core storage architecture — the inverted index and the
operator index — to [monoprop](https://github.com/Algorithmiq/monoprop) (Algorithmiq).
See the header comments in `crates/core/src/lib.rs`,
`crates/core/src/storage/inverted_index.rs`, and
`crates/core/src/storage/operator_index.rs`. The lineage matters when deciding what is
genuinely novel here.

---

## 1. Measured baseline

Reference problem: 64 qubits, TFI Trotter (`tfitrottercircuit`, bricklayer), 16 layers,
2032 gates, all angles 0.31, observable `Z` on qubit 32, `min_abs_coeff = 1e-8`,
2,041,914 final terms. Best of 3, after warm-up, GC before each run.

### Our own paths differ by 39x

| path | 12 threads | vs. best |
|---|---|---|
| `PauliSum` (Dict) — the type used in the README and every example | **20.44 s** | 39.1x |
| `VectorPauliSum`, plain `propagate` | **3.27 s** | 6.3x |
| `VectorPauliSum`, `Performance.propagate` (`fused=true`) | **0.52 s** | 1.0x |

Anyone benchmarking us through the documented API measures the 20 s path. That gap alone
accounts for an order of magnitude, before any architectural difference.

### Thread scaling of our best path

| threads | 1 | 2 | 4 | 8 | 12 | 24 |
|---|---|---|---|---|---|---|
| `VectorPauliSum` + fused | 1.104 s | 1.732 s | 0.977 s | 0.586 s | 0.809 s | 0.475 s |
| speedup | 1.00x | 0.64x | 1.13x | 1.88x | 1.36x | **2.32x** |

~2.3x from 24 threads, i.e. ~10% parallel efficiency. This will not improve at 64 threads:
every per-gate pass is bandwidth-bound streaming over the whole pool and they all contend
for the same DRAM. **This is the single most important number in this document.**

### MultiPauliSum (current branch) already helps, but is zone-count sensitive

| config | `VectorPauliSum` fused | `MultiPauliSum` fused |
|---|---|---|
| 8 threads, 8 zones | 0.815 s | **0.542 s** |
| 24 threads, 16 zones | 0.605 s | 0.725 s |

The zone architecture wins when zones match threads and regresses when they do not.
propaq guards exactly this case (see §2.7).

### The number that explains the architectural gap

Instrumenting our own propagation over the same run: summed over 2032 gates and
324,668,235 term-visits, **9.07%** of terms anticommute with the gate generator.

propaq touches only those. We read and rewrite all of them, several times per gate.

---

## 2. Their design, mechanism by mechanism

### 2.1 Selection: a transposed bitmap index, not a scan

`crates/core/src/storage/inverted_index.rs`

They keep a transposed view of the pool: one column per symplectic bit position (2n of
them), each column holding the set of rows that set that bit. A column is a dense
`Vec<u64>` bitmap (one bit per row) or a sparse `Vec<u32>` of row indices, promoted to
dense once at least 1/64 of rows touch it (`PROMOTE_DENSITY_INV = 64`).

Anticommutation of term `t` with generator `g` is the parity of `t & pair_swap(g)`. With
the interleaved encoding, that is **an XOR of the columns named by the set bits of
`pair_swap(g)`** — at most 2 per gate qubit. `combine()` XORs those columns into a
bitmap; `for_each_set_bit()` then walks only the hits.

Cost per gate: `k` columns x N/64 words, versus N full-key commutation tests. For a
two-qubit generator that is 2 x N/64 word-XORs instead of N tests.

The index is **incremental**: `sync_to()` indexes only rows appended since the last call,
and rotations only ever append (keys are never mutated in place), so it stays valid across
the whole run. It is reset only by `reclaim()` / `repack()`.

Cost to be aware of: with all columns dense the index is `2n * N/8` bytes, roughly
doubling resident memory (n=64, N=12.6M gives ~200 MB on top of a ~164 MB store).

### 2.2 No pass over the pool, ever

`crates/core/src/engine/termsum.rs::scan_into` / `emit_from_row`

The ~91% of terms that commute are **not read and not written**. The branching rows get
`coeffs[i] *= cos` in place, and their sine branch is pushed into an outbox bucket keyed by
the destination partition. Merging is `absorb_routed()`: one open-addressed hash probe per
emitted branch, adding into an existing coefficient or appending a row.

There is **no sort anywhere in their engine** — confirmed by grep over
`crates/core/src/engine/` and `crates/core/src/storage/`.

### 2.3 Truncation is a creation gate, not a pool filter

`EmitCutoff` / `EmitPrecheck` in `crates/core/src/engine/termsum.rs`

`EmitPrecheck::declines` decides from `|coeff * sin| < min_coeff` **alone**. When it fires,
the Pauli product, the hash, and the insert never happen. The weight cutoff likewise
rejects `weight(child) > max_weight` before insertion, and the store's inline row width is
sized from the weight cutoff so rows never spill.

Consequence: **there is no truncate pass over the pool at all.** `reclaim()` is invoked
only after a noise layer (`crates/pauli/src/engine.rs:159`). Without noise, a term once
created stays forever; its coefficient just decays. Dead terms cost one cache-line read
plus one multiply on the gates they happen to anticommute with.

They also have a free-accuracy trick worth noting (`hold_back` / `claim` / `drain_claims`):
a rotation pairs rows s <-> r, and the product map is an involution. If s emitted into r,
then r's own below-cutoff branch lands on s, which already exists — so it is added anyway.
No new term is created, so it costs nothing structurally. Implemented as a second, usually
tiny, exchange round.

### 2.4 Clifford gates cost O(n), not O(N)

`crates/core/src/algebra/tableau.rs`

A rotation with `|cos theta| < 1e-9` is never applied to the terms. It is composed into a
2n-row `CliffordTableau` frame; the **generator** of every later non-Clifford rotation is
pushed back through the frame's inverse before the normal path runs, and the frame is
applied once at readout. Disabled only when a weight cutoff is set and the step changes
weight (`changes_weight()`), or when a term-aware noise/truncation plugin is loaded.

This is invisible in a TFI Trotter comparison. It is decisive in their own benchmark
(`benchmarks/bench_ucj.py`), which transpiles LUCJ circuits to
`cp, xx_plus_yy, p, x, swap` — where `x` and `swap` are Clifford and abundant.

### 2.5 Encoding

`crates/core/src/algebra/strings.rs`, `crates/pauli/src/algebra.rs`

`BasisString<W>` is `[u64; W]` with **W a compile-time const generic**, x/z interleaved two
bits per qubit (bit 2q = X, bit 2q+1 = Z). `crates/pauli/src/engine.rs::runner_for`
monomorphizes W and the position type at 32/64/128/256/512/1024/2048 qubits, so every word
loop fully unrolls and the key is `Copy`.

The interleaving buys:

- weight = `((w | (w >> 1)) & 0x5555...).count_ones()`, one pass
- the anticommutation fold is exactly `pair_swap()`
- the product is one XOR of the whole key

This is essentially what our `getinttype` + BitIntegers encoding already does. **Not a
significant differentiator.**

### 2.6 Storage: position lists, not bitmasks

`crates/core/src/storage/operator_index.rs`

This *is* a differentiator. Rows are stored as position lists —
`[count, pos0, pos1, ...]` in `u8`/`u16` — with `stride = 1 + inline_width`. The inline
width is `2 * max_weight` when a weight cutoff is set, otherwise starts at 24 and adaptively
repacks (doubling, capped at 32) once more than 20% of rows spill into an overflow
`HashMap`.

**Row size is O(weight), not O(n_qubits).** A weight-8 term on 512 qubits is ~25 bytes;
ours is 128 bytes regardless of weight.

The dedup table is separate: `Vec<Slot { idx: u32, hash: u32 }>`, 8 bytes per slot, linear
probing, 0.7 load factor, with the stored 32-bit hash as an equality prefilter so the row
itself is rarely touched. Compare Julia's `Dict{UInt256,Float64}`, which stores 32-byte keys
inline in the table plus a separate slots byte-array: three random accesses per probe and
several times the footprint.

### 2.7 Threading

`crates/core/src/engine/partitioned_termsum.rs`, `crates/core/src/engine/affinity.rs`

- One partition per worker, owned exclusively. Key -> partition by `hash >> 32 % S`.
- Two phases per gate: **scan** (each worker fills `outbox[src][dst]`) then **absorb**
  (each worker drains column `dst` from every source into its own store). No locks, no
  atomics on the hot path — only two relaxed timing counters.
- Outboxes are preallocated `Vec<Vec<Vec<Routed>>>`, `clear()`ed and reused. Zero
  allocation per gate.
- `broadcast_applies(n) = n_partitions == rayon::current_num_threads()` switches to
  `rayon::broadcast`, so worker *i* always gets partition *i* across every gate.
- `pin_threads = true` **by default**: `sched_setaffinity` binds each worker to its own CPU,
  so a partition's rows and hash table stay in one core's cache slice for the entire run.
- `_mm_prefetch` on the hash slot `PREFETCH_GROUP = 16` messages ahead in the absorb loop.

---

## 3. Ours, side by side

Per Pauli rotation with generator G over a pool of N terms, A of which branch (A/N ~ 0.09
measured):

| step | propaq | PauliPropagation.jl (`VectorPauliSum`, non-fused) |
|---|---|---|
| select branching rows | k x N/64 word-XORs over the transposed index | `flagterms!` — read all N keys, test each, write N bools |
| index the branches | `for_each_set_bit` over the bitmap | `flagstoindices!` — full parallel prefix sum over N |
| branch | A products, A pushes into own outbox | `_applypaulirotation!` — read N flags, write A pairs |
| merge | A hash probes, prefetched | `sortbyterm!` (`AK.sortperm!` over N+A) + permute + `_flaggroupbegin!` + prefix sum + `_mergegroups!`, or `sortedtailmerge!` |
| truncate | none (folded into emit) | full flag + prefix sum + filter pass |
| Clifford gate | O(n) tableau compose | O(N) lookup-map pass over every term |

`Performance.propagate` (`fused=true`) removes the flag/cumsum passes and folds weight
truncation into the branch write — hence the 6.3x — but the shape is unchanged: it still
walks all N terms, and it still rewrites the whole pool in the merge.

Two specific costs in our code that propaq structurally cannot have:

**`_taskedbranchwrite!` runs every gate twice.**
`src/Performance/fused_vector.jl:138-171` (and `_fusedbranchwrite!` at `:191`). Because several tasks append into one shared
array, they must agree on offsets, so there is a dry run that counts followed by a real run.
`_fusedbranchwrite!` with `DoWrite=false` still calls `_gatecommutes` *and* `_gateproduct` —
only the writes are skipped. That is why 2 threads is *slower* than 1 in the scaling table.
propaq never needs this: each worker appends to its own outbox, so offsets are never shared.
`MultiPauliSum` + `fused_multi.jl` already fixes this, and the 8-thread measurement shows it.

**Every merge rewrites the whole pool.**
`src/Base/sortedtailmerge.jl:58-117` two-pointer-merges head against tail into the aux arrays
— all N keys and coefficients copied, every gate, even though ~91% of them were untouched.
At 12.7M terms on 64 qubits that is roughly 300 MB read + 300 MB written per gate of pure
copying. `xorsortedtailmerge!` avoids the *sort* but not the *rewrite*.

---

## 4. What to do, ranked

### 4.1 Transposed anticommutation index — biggest single win

Build the inverted index per zone in `MultiPauliSum`. Columns keyed by the 2n bit
positions, dense `Vector{UInt64}` bitmap or sparse `Vector{UInt32}` row list with the same
1/64 promotion rule. Sync incrementally after each merge; reset on truncation.

At our measured 9.07% branching fraction this is ~11x fewer rows touched per gate, and the
selection itself collapses from N tests to a couple of bitmap XORs. Independent of
everything else on this list, and it composes with the existing `ByteMask`/`WordMask` work
(which optimizes the *per-term* test we would then largely stop doing).

Open question to settle first: our pool shrinks under `min_abs_coeff` truncation, which
forces an index rebuild. Theirs never shrinks, so their index is pure amortized gain.
Measure the rebuild cost against a scheme that tolerates tombstoned rows.

### 4.2 Stop rewriting the pool

In-place `coeffs[i] *= cos` on the selected rows, append branches, dedup by hash. This is
what removes the bandwidth wall capping us at 2.3x — the merge is the pass that scales
worst because it is pure streaming over data nothing happened to.

Requires a hash index per zone rather than the sorted-prefix scheme. Worth prototyping
against `sortedtailmerge!` on the same problem before committing: our sorted-prefix
representation buys fast `getcoeff` and ordered iteration that a hash store loses.

### 4.3 Truncate at emit

Check `|coeff * sin| < min_abs_coeff` **before** computing the product, and
`weight(child) > max_weight` before writing. `fused_vector.jl` already does the weight half
(`_truncateweight` before `_writeandadvance!`); the coefficient half is the one that
currently waits for the merge. Doing both at emit deletes the separate truncate pass
entirely.

Their `hold_back`/`claim` pair rule is a genuinely free accuracy improvement and worth
copying alongside it, since it only ever adds mass to rows that already exist.

### 4.4 Clifford deferral via a tableau frame

Roughly 200 lines: a 2n-row tableau, compose on each Clifford gate, conjugate the generator
of each subsequent rotation, apply once at readout. Turns every `CliffordGate` from O(N)
into O(n). Currently `src/Propagation/vectorspecializations.jl:187-213` runs a lookup map
over the whole pool per Clifford gate.

Must be disabled when a weight cutoff is active and the deferred frame changes weight, and
when a custom truncation function reads the Pauli string. Dominant on the circuit families
propaq benchmarks; irrelevant for TFI Trotter, so measure on a CNOT/SWAP-heavy circuit.

### 4.5 Pin zones to cores; require `nzones == nthreads`

Our 24-thread/16-zone regression (0.725 s vs 0.542 s at 8/8) is exactly the case propaq's
`broadcast_applies` guard exists to prevent. Two parts:

- Default `defaultnzones()` to the thread count where it is a power of two, and document the
  mismatch penalty otherwise.
- Pin zone-owning tasks to cores. Julia has no `sched_setaffinity` wrapper in Base; a small
  `ccall` on Linux is enough, or `ThreadPinning.jl` as a weak dependency.

At 64 threads this is the difference between the zone architecture paying off and the
scheduler shuttling zones between NUMA domains.

### 4.6 Position-list rows

Store rows as `[count, positions...]` in `UInt8`/`UInt16` rather than a full-width integer,
sized from the weight cutoff, with an overflow map and adaptive repack. Makes row size
O(weight) rather than O(n_qubits) — irrelevant at 64 qubits, large above ~256.

Lowest priority of the six: it is a constant-factor memory win, and it costs a
reconstruct on every algebra operation. Only worth it once §4.1 and §4.2 have removed the
passes that make the pool's total size matter.

### 4.7 Not architectural, but the largest measured number here

Make `Performance.propagate` the default, or at minimum the documented path. A 39x gap
between what the README shows and what the library can do will keep costing us in other
people's benchmarks, and it is the single cheapest thing on this list to fix.

---

## 5. Benchmark-fairness notes

Their cited gap is plausible, but several things need controlling before accepting the
comparison as apples-to-apples.

**Different truncation semantics.** Our `min_abs_coeff` is a *pool filter*: it deletes terms
after every gate. Their `coeff_cutoff` is a *creation gate*: terms below it are never
created, but terms already in the pool stay forever regardless of how far their coefficients
decay. At the same nominal cutoff, their pool is a superset of ours. This works *against*
them on term count and *for* them on accuracy, and it means "same cutoff" is not the same
computation. **A fair comparison should control for final term count, not for cutoff value.**

**Which of our APIs was measured.** See §1 — the spread across our own paths is 39x. The
documented path is the slow one.

**Julia thread count.** `maxtasks(thread) = Threads.nthreads()`. Started without `-t` or
`JULIA_NUM_THREADS`, everything runs on one task regardless of `thread=true`.

**JIT warm-up.** A first `propagate` call includes compilation.

None of these change the conclusion that the architectural gap in §2.1-2.4 is real. They do
change its size.

---

## 6. Reproduction

Scripts used for §1, against `24cce0b`:

```julia
# Path comparison (12 threads)
using PauliPropagation
nq, nl, minc = 64, 16, 1e-8
topo  = bricklayertopology(nq)
obs   = PauliString(nq, :Z, nq ÷ 2)
circ  = tfitrottercircuit(nq, nl; topology=topo)
params = fill(0.31, countparameters(circ))
P = PauliPropagation.Performance

propagate(circ, PauliSum(obs), params; min_abs_coeff=minc)                          # 20.44 s
propagate(circ, VectorPauliSum(PauliSum(obs)), params; min_abs_coeff=minc)          #  3.27 s
P.propagate(circ, VectorPauliSum(PauliSum(obs)), params; min_abs_coeff=minc)        #  0.52 s
```

```julia
# Anticommuting fraction: 29,440,582 / 324,668,235 = 9.07% over 2032 gates
cache = PropagationCache(VectorPauliSum(PauliSum(obs)))
pit = Iterators.Stateful(params)
tot_n = tot_a = 0
for gate in circ
    p = isa(gate, ParametrizedGate) ? popfirst!(pit) : nothing
    n = PauliPropagation.activesize(cache)
    if n > 0 && isa(gate, PauliRotation)
        gm = PauliPropagation.symboltoint(PauliPropagation.paulitype(cache), gate.symbols, gate.qinds)
        trms = PauliPropagation.terms(PauliPropagation.mainsum(cache))
        a = 0
        @inbounds for ii in 1:n
            a += !PauliPropagation.commutes(trms[ii], gm)
        end
        tot_n += n; tot_a += a
    end
    p === nothing ? PauliPropagation.applymergetruncate!(gate, cache; min_abs_coeff=minc) :
                    PauliPropagation.applymergetruncate!(gate, cache, p; min_abs_coeff=minc)
end
```

Thread scaling was measured by launching one process per `julia --project=. -t N`, since
`max_tasks` is derived from `Threads.nthreads()` and is not a user-facing kwarg.

Caveat: this box has 24 logical cores (hyperthreaded) and had a VS Code Julia session
running throughout, so the 12- and 24-thread points are noisy. The trend — no meaningful
scaling past ~8 threads — is robust across runs; the individual timings are not.
