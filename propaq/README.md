# Propaq's techniques, measured against PauliPropagation.jl

Propaq (arXiv:2609.07730, `paper.pdf`) benchmarks PauliPropagation.jl at up to 100x slower than
itself. This directory takes its techniques apart one by one and measures each against the
library's fastest path, `Performance.propagate` over a `MultiPauliSum` (the path
`examples/advanced_performance.ipynb` ends on), on the paper's own circuits at sizes up to 35M
Pauli strings. Propaq's source (`propaq_repo/`, v0.1.5 at `0086ea1`) and its benchmark repository
(`propaq-benchmarks/`) are checked out alongside and were built and run here; `src/` holds two
prototypes built on the library's own types so that the two storage designs can be measured on the
same problems; `propaq-comparison.md` is the earlier code-reading study that motivated them. Some of
that study's conclusions did not survive measurement, and this file supersedes it where they differ.

Everything below was measured on an AMD Ryzen AI 7 350 (8 cores, 16 hardware threads, 48K L1d,
1MB L2 per core, 16MB L3, ~60 GB/s of DRAM bandwidth that two cores already saturate), Julia 1.12.5,
`multi-vector` at `24cce0b`, Propaq built from source with `target-cpu=native`. Best of two or three
runs, GC before each. Thread counts above 4 buy little on this machine and above 8 they lose, so the
8-thread numbers are its ceiling, not a trend.

```
propaq/
  src/cliffordframe.jl      Clifford deferral: a tableau frame over the library's gates
  src/IndexedPropagation.jl the transposed index and hash-table store, with the files below
  src/symplectic.jl         the symplectic view of a Pauli string, one word at a time
  src/transposedindex.jl    the bitmap columns and the sweep that marks branching terms
  src/termtable.jl          Pauli string -> position, open addressed
  src/propagationcache.jl   the single-zone cache: term vector, both indices, compaction
  src/gates.jl              the rotations and the noise channels
  src/multizone.jl          the same split over work zones, which is where the threads are
  bench/problems.jl         the circuits of the paper and of the notebook
  bench/benchmark.jl        every path of the library and the prototype over one problem
  bench/papercircuits.jl    the same over circuits saved by propaq-benchmarks, one JSON line per run
  bench/generatecircuits.py saves every circuit measured below in that JSON layout
  bench/exportcircuit.jl    saves a problem of problems.jl in that JSON layout, for Propaq
  bench/run_propaq.py       Propaq over those circuits, with its engine's own phase split
  bench/phases.jl           where the fused path's time goes, gate by gate, on a large sum
  bench/indexcost.jl        what the two indices cost to build and read, per term
  bench/precheck.jl         Propaq's emit precheck tried on the fused path
  bench/deferral.jl         Clifford deferral against the fused path
  test/runtests.jl          the storage prototype against the library, gate by gate
```

---

## The verdict

| Propaq technique | measured here | worth it? |
|---|---|---|
| Clifford deferral through a tableau | a TFIM circuit written as `cx rz cx`, as Qiskit hands it over, runs 4.5-4.7x faster with the frame and gives the same value to every digit | **yes, first**: self-contained, exact, and it makes imported circuits cost what native ones do |
| emit precheck, `\|c sin θ\|` below the cutoff never becomes a product | 5-18% faster, 22-32% more terms, ⟨Z⟩ moves in the fourth digit | **not without Propaq's rescue round**, and the rescue needs random access to the partner term |
| transposed index for selecting branching terms | marking is free; building the index is 11 ns/term against the 1.15 ns/term scan it replaces, and the sorted merge reorders the sum every gate | **only inside an append-only store**; it cannot be added to the sorted path |
| hash-table store, no sort and no rewrite | 2.0x slower than `fused` on one thread at 27M terms; level with `fused multi` from 16M terms up at 4-8 threads; 1.15-1.3x ahead below 6M terms at 8 threads; better thread scaling (3.9x vs 2.4x from 8 threads at 17M) | **not on this machine**; it is the right structure for many cores |
| position-list rows, O(weight) per term | 25 B per row at 36 qubits and 50 B at 100, against 16 B and 32 B for the library's integers | **no** on the paper's problems |
| decayed terms never removed | 10-14% more terms at the same cutoff, no gain in accuracy against a 1e-7 reference | **no** |
| hash partitions with outboxes, three barriers per gate | 4.1x from 8 threads, but from 2.2x further back; the zone-pair decomposition in `src/multizone.jl` scales as well without an outbox | **no**; the library's zones are the better decomposition |
| thread pinning | within 4% either way | **no** |

In one breath: implement Clifford deferral; leave the fused `MultiPauliSum` path as it is, because
neither the index nor the precheck can be grafted onto a sorted merge; and treat the no-rewrite
store as a question about the machine, not about the code. On eight cores it breaks even with the
fused path at the sizes this machine can hold, and its only measured advantage is that it keeps
scaling where the streaming path runs out of bandwidth. Propaq's Fig. 11 puts that scaling at
10-20x on 64 cores; nothing here can confirm or refute it, and the section at the end says how a
many-core node would.

What the comparison exposes on our side, though it is not a Propaq technique: at 27M terms the
library's peak is 98-137 B per term and Propaq's is 77-88, because the library holds a second copy
of the sum for the merge and a box per zone, not because of anything in Propaq's rows.

---

## How Fig. 1(a) gets its 100x

The paper's PauliPropagation.jl curve is `propagate(circuit, VectorPauliSum(psum), thetas)` under
`propaq-benchmarks/julia_env`, which pins **v0.7.3** (30 March 2026) on Julia 1.11.3, run with
`-t 64` on 64 EPYC 7H12 cores. Three things multiply.

**The path.** v0.7.3's vector path sorts the whole sum with `AK.sortperm!` after every gate and
has no `Performance` module and no `MultiPauliSum`. On this machine at 16 Trotter steps, 6.1M
terms: 56.1 s on one thread and 24.1 s on eight, against 6.2 s and 2.8 s for `fused` and
`fused multi` today. That is 8-9x on its own, and it is the whole of "which library was measured".

**64 threads on that path.** Every pass of it is an `AK.foreachindex` over 64 tasks, and there are
about ten passes per gate. The paper's curve reads 0.5 s at step 3 and 1.8 s at step 5, where the
sum holds hundreds of terms: 1.7 to 3.7 ms per gate of pure task overhead. Here the same path costs
0.19 ms per gate on one thread and 0.31 ms on eight, and on sixteen, which this box cannot run
without sharing cores, it collapses:

| threads | 8 steps, 42K terms | 12 steps, 687K | 14 steps, 2.2M |
|---|---|---|---|
| 1 | 0.14 s | 3.99 s | 15.5 s |
| 4 | 0.26 s | 2.20 s | 7.9 s |
| 8 | 0.24 s | 1.98 s | 7.1 s |
| 16 | 15.4 s | 45.9 s | 64.0 s |

The 16-thread row is this machine's hardware threads fighting over eight cores and does not
transfer to a 64-core socket; the per-gate overhead does. At step 15 the paper reads about 50 s
where v0.7.3 on eight threads here interpolates to about 13 s, so the paper's machine and thread
count together cost the old path another 4x, part of which is a 2.6 GHz Zen 2 core against a 5 GHz
Zen 5 one.

**Propaq on 64 cores.** The paper's Propaq curve reads about 0.5 s at step 15, where Propaq on
eight threads here takes about 2.8 s: 64 cores give it 5-6x over this machine despite the slower
core, which agrees with the 10-20x from 1 to 64 threads its Fig. 11 claims. Propaq's per-term work
is a few random cache lines, so it keeps scaling where a streaming design hits the memory wall.

So at step 15 the old path is 4.6x behind Propaq on eight threads here, 13 s against 2.8 s, and
100x behind on their machine, because 64 threads cost the old path 4x and gain Propaq 5-6x. Against
the current library the 4.6x turns around: Propaq is behind at every size measured, 2.5x on one
thread and 1.4x on eight at 27M terms, in the tables below.

### The rest of their harness

Nothing else in `propaq-benchmarks/experiments/ising_trotter` tilts the Pauli comparison, but for
the record:

- The Julia runner warms up on a one-qubit circuit, whose integer type is `UInt8`; the first 6x6
  circuit therefore pays the `UInt128` compilation inside its timed region. It is the 1-step point
  only.
- The two packages truncate differently. The library removes a term the moment its coefficient
  falls below the cutoff; Propaq applies the cutoff to a product as it is emitted and never removes
  a term afterwards unless a noise layer triggers `reclaim`, so at 19 steps 4.5M of its 30.7M terms
  are below the cutoff it was given. The extra terms do not buy accuracy: at 14 steps the 1e-6
  values are 0.956410 (library) and 0.956395 (Propaq) against 0.956148 at 1e-7, and on the
  notebook's circuit 0.669683 and 0.669630 against 0.670477. Same nominal cutoff, same error,
  10-14% more terms for Propaq.
- Peak RSS for Julia includes the runtime, 0.5 GB here, and whatever the GC has not returned yet.
- The circuits go through Qiskit's transpiler at `optimization_level=1` before being saved, which
  interleaves the `rzz` and `rx` gates. Every backend gets the same gate list.
- Angle and observable conventions match: `rzz(θ)` is `exp(-iθ/2 ZZ)` on both sides, and the
  right-to-left Qiskit label lands the `Z` on the middle qubit.

---

## The measurements

### The paper's 6x6 TFIM, cutoff 1e-6

Fig. 1(a)'s circuit from the benchmark repo's own generator, run over the Trotter step counts the
memory of this machine allows. `fused` is `Performance.propagate` over a `VectorPauliSum`,
`fused multi` the same over a `MultiPauliSum` of 32 zones; `indexed` and `indexed multi` are the
storage prototype below with one and 32 zones. Propaq's term counts are its own, 10-14% higher.

One thread:

| steps | terms | v0.7.3 | `fused` | `indexed` | Propaq |
|---|---|---|---|---|---|
| 14 | 2,192,913 | 15.5 s | **1.87 s** | 2.92 s | 3.83 s |
| 16 | 6,126,993 | 56.1 s | **6.16 s** | 11.2 s | 14.3 s |
| 18 | 17,107,380 | | **19.6 s** | 37.3 s | 47.9 s |
| 19 | 26,947,179 | | **33.2 s** | 66.4 s | 83.8 s |

Eight threads:

| steps | terms | v0.7.3 | `fused` | `fused multi` | `indexed multi` | Propaq |
|---|---|---|---|---|---|---|
| 14 | 2,192,913 | 7.05 s | 1.22 s | 0.89 s | **0.68 s** | 1.82 s |
| 16 | 6,126,993 | 24.1 s | 3.84 s | 2.79 s | **2.43 s** | 4.28 s |
| 18 | 17,107,380 | | 11.6 s | 8.90 s | **7.96 s** | 11.7 s |
| 19 | 26,947,179 | | 19.3 s | **13.8 s** | 14.2 s | 19.6 s |

`fused multi` over 8 zones instead of 32 is within 5% at every size (14.2 s at 19 steps).

**Threads**, at 18 steps and 17M terms:

| threads | `fused multi` | `indexed multi` | Propaq |
|---|---|---|---|
| 1 | 21.4 s | 30.9 s | 47.9 s |
| 2 | 12.3 s | 17.6 s | 27.2 s |
| 4 | 9.66 s | 10.5 s | 15.5 s |
| 8 | 8.90 s | 7.96 s | 11.7 s |
| speedup at 4 / 8 | 2.2x / 2.4x | 2.9x / 3.9x | 3.1x / 4.1x |

Pinning the Julia threads (`JULIA_EXCLUSIVE=1`): 8.60 s and 8.00 s at eight threads. Propaq pins
by default and the earlier study found that turning it off moves nothing either.

### The notebook's circuit

`examples/advanced_performance.ipynb`'s tilted-field Ising circuit, `Rx`, `Rz` and `Rzz` layers on
the 6x6 grid with `ZZ` on the two middle qubits, at the paper's cutoff of 1e-6 rather than the
notebook's 2^-20, which is why 14 layers give 994K terms here and 1.04M there. Four threads:

| layers | terms | `fused multi` | `indexed multi` | Propaq |
|---|---|---|---|---|
| 14 | 994,492 | 0.58 s | **0.41 s** | 1.07 s |
| 20 | 15,697,380 | **11.2 s** | 11.2 s | 15.0 s |
| 22 | 35,054,014 | **25.2 s** | 30.0 s | 37.6 s |

Same picture as the paper's circuit: the hash store wins below a few million terms, meets the
streaming path in the tens of millions and falls behind at 35M.

### Memory at 27M terms

Peak RSS of one run above the Julia baseline of 0.5 GB, eight threads:

| path | peak | per term |
|---|---|---|
| `fused multi`, 32 zones | 3.69 GB | 128 B |
| `fused multi`, cache `resize!`d to 28M up front | 3.36 GB | 114 B |
| `fused`, cache `resize!`d up front | 2.96 GB | 98 B |
| `indexed multi` | 3.97 GB | 137 B |
| Propaq, 30.7M terms, one thread / eight | 2.36 GB / 2.69 GB | 77 B / 88 B |

The library's sum is 24 B per term at 36 qubits. What multiplies it is the auxiliary copy the merge
writes into (24 B), the box each zone of a `MultiPauliSum` branches into (up to 24 B), and the
1.5x growth steps of all three. Propaq's row is 25 B, wider than ours, plus 8 B of coefficient,
11 B of hash slots at a load of 0.7, 9 B of transposed index and 4 B of pending-branch slot; it has
no second copy of anything. On a machine where 30M terms is the limit, that is the one place Propaq
is materially ahead, and closing it is a change to the merge, not to the storage.

### Where the fused path's time goes

`bench/phases.jl` times the last Trotter step gate by gate with the three passes of the fused
vector path separated: the scan that tests every term and appends the products, the XOR sort of
the appended tail, and the merge that rewrites the head against it.

| | 17M terms, 1 thread | 17M terms, 8 threads | 27M terms, 1 thread |
|---|---|---|---|
| scan and append | 2.19 s, 30%, 1.64 ns/term | 1.60 s, 30%, 1.20 ns/term | 3.42 s, 28% |
| XOR sort of the tail | 1.82 s, 25%, 8.6 ns/product | 0.47 s, 9%, 2.2 ns/product | 3.33 s, 28% |
| merge and rewrite | 3.17 s, 44%, 2.38 ns/term | 3.26 s, 61%, 2.45 ns/term | 5.30 s, 44% |

16% of the terms branch in that step. 70% of the products are below the cutoff the moment they are
made, and 51% land on a term already in the sum. The merge does not speed up with threads at all:
its threaded version counts in a dry run before it writes, so it reads the sum twice, and the
bandwidth is gone by then. This is the wall the storage prototype was built to get around, and the
table also says what each Propaq idea can touch: the transposed index replaces the scan, at most
30%; the precheck shrinks the tail, at most the sort's 25%; only a store that does not rewrite gets
at the 44-61%.

### The two indices, built and read

`bench/indexcost.jl` on the 17M-term sum, one thread, one `ZZ` rotation in the middle of the grid:

| | time | per term | memory |
|---|---|---|---|
| fused scan, testing every term | 0.020 s | 1.15 ns | |
| transposed index, build | 0.188 s | 11.0 ns | 9 B/term |
| transposed index, mark | 0.000 s | 0.01 ns | |
| term table, build | 0.334 s | 19.5 ns | 15 B/term |

Marking is as free as the earlier study said. Building is not, and the sorted merge writes the sum
back in a new order after every rotation, so on the sorted path the index would have to be rebuilt
every gate at ten times the cost of the scan it replaces. It lives only next to a store that never
moves a term, which is the hash table -- the two ideas the earlier study called independent are
not.

### The precheck without the rescue

`bench/precheck.jl` drops every product whose coefficient is below the cutoff as it is made, as
Propaq's `EmitPrecheck` does, but without the `hold_back`/`claim` round that lets a dropped branch
land on its partner after all. `fused multi`, eight threads:

| steps | exact terms | precheck terms | exact ⟨Z⟩ | precheck ⟨Z⟩ | time |
|---|---|---|---|---|---|
| 14 | 2,192,918 | 2,669,846 (+22%) | 0.956410 | 0.956257 | 0.91 s → 0.74 s |
| 16 | 6,126,983 | 7,714,063 (+26%) | 0.938056 | 0.937772 | 2.77 s → 2.56 s |
| 18 | 17,107,365 | 22,573,972 (+32%) | 0.948410 | 0.948146 | 8.41 s → 8.00 s |

A rotation moves coefficient between a term and its partner in both directions; dropping the small
transfers leaves mass where the exact dynamics would have cancelled it, and the sum grows. Propaq
pays a third exchange round per gate to put those transfers back, which its counters put at 10% of
its run, and its term count stays in line. In a sorted merge the partner is not at hand when the
product is being decided, so the library cannot have the precheck without first having a table.

### Clifford deferral

`src/cliffordframe.jl` is Propaq's `CliffordTableau` over the library's gates: a `CliffordGate`
is composed into a frame of `2n` generator images and their inverses in O(n), each later
`PauliRotation` has its generator conjugated through the inverse frame before it runs, and the
frame is applied to the terms once at readout. `bench/deferral.jl` runs it against the plain fused
path, four threads, and the values agree to every printed digit:

| circuit | terms | Cliffords | `fused multi` | with the frame | native `rzz` circuit |
|---|---|---|---|---|---|
| 6x6 TFIM, 10 steps, as `cx rz cx` | 187,787 | 1,200 | 0.59 s | **0.13 s** | 0.05 s |
| 6x6 TFIM, 12 steps, as `cx rz cx` | 687,036 | 1,440 | 2.53 s | **0.54 s** | 0.24 s |
| paper's Clifford+T, p = 0.1, cutoff 1e-9 | 10 | 8,971 | 0.06 s | 0.08 s | |
| paper's Clifford+T, p = 0.25 | 2,529 | 8,938 | 0.11 s | 0.09 s | |

The `cx rz cx` form is what a Qiskit-transpiled circuit looks like when it reaches the library, and
on it every CNOT rewrites every term in place and resets the sorted prefix, so the next rotation's
merge is a full sort: 10x slower than the same circuit written with `rzz`. The frame removes most of
that; what is left over the native circuit is the prototype driving `propagate!` one gate at a time.
Propaq on the same two circuits takes 0.93 s and 2.06 s at four threads.

The paper's own Fig. 7 circuits, 64 qubits and 80 layers of random Cliffords with a T gate every
few layers, hold 10 and 2,529 terms at their 1e-9 cutoff, so the library's plain Clifford pass
finishes them in a tenth of a second and there is nothing to defer. Propaq needs 1.7 s for them at
four threads, three `rayon::broadcast` barriers per gate over 7,700 gates. The deferral kill switch
that experiment flips, `PROPAQ_DISABLE_CLIFFORD_DEFERRAL`, is not read by v0.1.5, so its "off"
number cannot be reproduced from this checkout.

What a library version has to respect, as Propaq's does: a weight cutoff sees the stored term, not
its image, so deferral is off whenever `max_weight` is set and a deferred gate changes weight;
noise channels and custom truncations that read the Pauli string see the stored term too, so a
frame has to be flushed before them or they have to be conjugated through it.

---

## The storage prototype

`IndexedPropagation` is the storage architecture Propaq credits to `monoprop`, on the library's
own types and gate machinery. It agrees with `propagate` term for term and coefficient for
coefficient over `PauliRotation`, `ImaginaryPauliRotation` and `PauliNoise` under `min_abs_coeff`
and `max_weight` (`test/runtests.jl`); every other gate throws, because a
Clifford gate rewrites Pauli strings in place and invalidates both indices.

```julia
include("propaq/src/IndexedPropagation.jl")
using .IndexedPropagation

psum = IndexedPropagation.propagate(circuit, VectorPauliSum(observable), thetas; min_abs_coeff=1e-6)
psum = IndexedPropagation.propagate(circuit, VectorPauliSum(observable), thetas; n_zones=32, min_abs_coeff=1e-6)
```

### The symplectic view (`symplectic.jl`)

The library packs a Pauli into two bits: `01` for X, `10` for Y, `11` for Z. Rewriting that pair as
the symplectic pair `(x, z)` -- `x` set for X and Y, `z` set for Y and Z -- puts `x` on the even bit
and `z` on the odd bit of the same qubit, and makes anticommutation the parity of
`sym(t) & pairswap(sym(g))`. Both bits of a qubit live in the same 64-bit word, so the rewrite is

```julia
symplecticword(w) = ((w ⊻ (w >> 1)) & 0x5555555555555555) | (w & 0xaaaaaaaaaaaaaaaa)
```

applied word by word, and nothing ever shifts a whole Pauli string. `columnsof` names the bit
positions whose parity decides anticommutation with a generator: an X or a Z on a qubit contributes
one, a Y two. In the library's own basis a Z would contribute two, which doubles the work of every
ZZ rotation.

### The transposed index (`transposedindex.jl`)

`TransposedIndex` holds `2 * nqubits` bitmap columns of one bit per term, and marking the branching
terms of a gate is the XOR of the generator's columns over `nterms / 64` words: two columns for a ZZ
rotation, one for an X rotation, about 0.25 bytes read per term against the 24 of the scan. The
index is append-only; a rotation never rewrites a Pauli string it keeps, so only the terms appended
since the last gate are added, at one bit per non-identity Pauli and one more per Y or Z.

### The term table (`termtable.jl`)

`TermTable` is an open-addressed table from Pauli string to its position in the term vector: one
word per slot, position in the low half and a hash tag in the high half, linear probing, doubled at
a load factor of 0.7. The tag settles almost every mismatch without reading the Pauli string. This
replaces the sort and merge: a product is looked up once and either added onto the term already
there or appended.

### Pairs (`gates.jl`)

A rotation with generator `g` sends `t` to `t ⊻ g` and `t ⊻ g` back to `t`, and the two either
both branch or neither does. So one probe settles both:

```
c_t     <- c_t cos θ - c_{t⊻g} sin θ s
c_{t⊻g} <- c_{t⊻g} cos θ + c_t sin θ s
```

with a single sign `s`, opposite for the two products of a real rotation and equal for an imaginary
one. This halves the probes and makes the gate exact against the library: a product below the
cutoff whose partner exists still shifts the partner, and only the shifted coefficient is judged --
the effect Propaq gets from its `hold_back`/`claim` round, here for free. A pair is handled at
whichever of the two walks reaches it first and the other half is noted in a `handled` bitmap;
noting it in the marks instead would make the next position depend on the probe that just landed
and leave the memory system with one outstanding miss at a time.

### Truncation

A truncated term has its coefficient zeroed in place, because removing it would move every term
behind it. Zeroed terms are skipped, can be revived by a later product, and are cleared out by
`compact!` once a quarter of the sum is dead, which rebuilds both indices.

### Zones (`multizone.jl`)

`MultiIndexedPauliPropagationCache` splits the sum with the library's own `ZoneMap`. Because the
zone assignment is linear over GF(2), a rotation sends every term of a zone into one and the same
other zone, so it decomposes into independent zone **pairs** that read and write nothing but their
own two zones: no outbox and no routing pass, where the library's `MultiPauliSum` needs both and
Propaq routes every product through an outbox and three barriers. A gate with zone bits gives
`n_zones / 2` units of work, so the zone count should be at least twice the thread count.

### What it costs

The 27M-term run on one thread takes 66.4 s over 1.2 billion marked terms (Propaq's counter for the
same circuit), 55 ns per marked term, where the fused path spends 1.6 ns per term it streams past.
That is the whole trade: the store touches one term in six and pays the cache misses of doing so --
its own Pauli string and coefficient, the table slot, and the partner's Pauli string and
coefficient, none of them in cache once the sum outgrows L3. The earlier round of this work
measured two things that look like they should fix that and do not: a second cursor running ahead
over the marks to prefetch the table slots, at any depth from 4 to 384, costs more in duplicated
hashing than it recovers; and splitting the sum into 16 to 512 zones changes nothing on one thread,
because a zone is touched exactly once per gate and its lines are cold every time. Those two were
not re-run here.

The tables above put the crossover with the streaming path at a few hundred thousand terms on one
thread and at ten to twenty million on four to eight, on this machine. The design earns its keep on
threads: its per-gate work is a few random cache lines per branching term, so it scales past the
bandwidth wall that flattens the streaming path at two to four cores, which is what the paper's
Fig. 11 shows on 64 of them and what cannot be measured here.

---

## Reproduction

```bash
# every circuit above, in the layout of propaq-benchmarks (pip install qiskit~=2.4.1 numpy)
PYTHONPATH=propaq/propaq-benchmarks/propaq-benchmarks/src python3 propaq/bench/generatecircuits.py circuits/
julia --project=. propaq/bench/exportcircuit.jl tilted 20 circuits/tilted_steps20.json

# the library and the prototype, one JSON record per run; one process per path for a clean peak RSS
ZONES=32 REPEATS=2 julia --project=. -t8 propaq/bench/papercircuits.jl fused fusedmulti indexedmulti -- circuits/6x6_steps19.json

# Propaq, built from source: pip install maturin qiskit~=2.4.1; cd propaq_repo; RUSTFLAGS="-C target-cpu=native" maturin develop --release
python propaq/bench/run_propaq.py 1,8 circuits/6x6_steps19.json

# v0.7.3, the path the paper measured, in an environment pinning it
julia --project=<env with PauliPropagation v0.7.3 and JSON> -t8 propaq/bench/papercircuits.jl default -- circuits/6x6_steps14.json

# the phase split, the index cost, the precheck and the deferral
julia --project=. -t1 propaq/bench/phases.jl circuits/6x6_steps18.json
julia --project=. -t1 propaq/bench/indexcost.jl circuits/6x6_steps18.json
julia --project=. -t8 propaq/bench/precheck.jl circuits/6x6_steps16.json
julia --project=. -t4 propaq/bench/deferral.jl circuits/6x6cx_steps12.json
```

`generatecircuits.py` reproduces the benchmark repo's own circuits byte for byte; the `cx rz cx`
ones are its 6x6 problem passed through `qiskit.transpile(basis_gates=["cx", "rz", "rx", "h"],
optimization_level=1)`, and the Clifford+T ones are `build_circuit(64, 80, p, 123)` of its
Clifford deferral experiment. Propaq's own Python packages install with `pip install -e
propaq/propaq-benchmarks/propaq-benchmarks` if the `PYTHONPATH` route is not wanted.

## What a many-core node can settle

Every number above tops out at eight threads, and the one open question is the one this machine
cannot answer: whether the no-rewrite store's scaling continues where the fused path's stops. The
runs that decide it, on the paper's 6x6 circuit at 18 and 19 steps (17M and 27M terms; 20 steps is
43M and about 7 GB for the fused path), each path in its own process:

```bash
for t in 1 2 4 8 16 32 64; do
  ZONES=$((2 * t > 32 ? 2 * t : 32)) REPEATS=2 julia --project=. -t$t propaq/bench/papercircuits.jl fusedmulti indexedmulti -- \
    circuits/6x6_steps8.json circuits/6x6_steps18.json circuits/6x6_steps19.json
done
python propaq/bench/run_propaq.py 1,2,4,8,16,32,64 circuits/6x6_steps18.json circuits/6x6_steps19.json
```

Three things to keep in mind reading them. A rotation with zone bits hands the zone-pair
decomposition `n_zones / 2` units of work, so the prototype needs at least twice as many zones as
threads, and `fused multi` is within 5% between 8 and 32 zones here, so the same count serves both.
`JULIA_EXCLUSIVE=1` pins the Julia threads, which did nothing on eight cores but is the obvious
thing to try across sockets, as is `numactl --membind` for a run that fits in one NUMA node.
Propaq's `rayon::broadcast` barriers cost 0.7 ms per gate at eight threads under WSL2 and should
not on a bare-metal node; its `scan_s`/`absorb_s`/`claims_s` in the JSON records say where its
time went. If the prototype's curve keeps bending down past 16 threads and `fused multi`'s does not,
the store is worth building properly, starting from `src/multizone.jl`; if both flatten, it is not.

The v0.7.3 curve of the paper can be re-taken with `papercircuits.jl default` under an environment
that pins that release (`Pkg.add(url="https://github.com/SparqleSim/PauliPropagation.jl",
rev="v0.7.3")` plus `JSON`; the benchmark repo's own `julia_env` pins Julia 1.11.3 and lacks JSON),
which on 64 real cores would show how much of the paper's curve is the 64-task passes and how much
the core.
