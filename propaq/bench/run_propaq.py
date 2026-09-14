"""
Runs Propaq's Pauli propagator over the circuits that `propaq-benchmarks` saves as JSON, the way
the benchmark repo's own runner does, and prints one JSON record per run with the engine's phase
split from its log and the process's peak RSS so far.

    pip install propaq
    python3 propaq/bench/run_propaq.py 1,8 circuits/6x6_steps14.json ...

`CUTOFF` and `PIN` (thread pinning, on by default as in Propaq) come from the environment.
"""

import json
import os
import resource
import sys
import tempfile
import time

from propaq import Logger
from propaq.circuits import PauliCircuit
from propaq.datatypes import PauliTermSum
from propaq.propagators import PauliPropagator
from propaq.truncation import TruncationPolicy
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

CUTOFF = float(os.environ.get("CUTOFF", "1e-6"))
PIN = os.environ.get("PIN", "1") == "1"
REPEATS = 3


def load_problem(path):
    with open(path) as f:
        d = json.load(f)
    qc = QuantumCircuit(d["n_qubits"])
    for g in d["gates"]:
        getattr(qc, g["name"])(*([g["angle"]] if "angle" in g else []), *g["qubits"])
    circuit = PauliCircuit.from_qiskit(qc)
    obs = SparsePauliOp(d["observable"]["paulis"], coeffs=d["observable"]["coeffs"])
    return circuit, PauliTermSum.from_sparse_pauli_op(obs), d["params"]["steps"]


def phases(logfile):
    with open(logfile) as f:
        records = [json.loads(line) for line in f]
    return next(r for r in records if r["event"] == "engine_phases")


def run(path, threads):
    circuit, obs, steps = load_problem(path)
    logfile = os.path.join(tempfile.gettempdir(), f"propaq_{os.getpid()}.jsonl")
    prop = PauliPropagator(None, TruncationPolicy(coeff_cutoff=CUTOFF), n_threads=threads,
                           logger=Logger(logfile, 1_000_000), pin_threads=PIN)
    best = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = prop.expectation_value(obs, circuit, initial_state=0)
        elapsed = time.perf_counter() - t0
        best = elapsed if best is None else min(best, elapsed)
    p = phases(logfile)
    os.remove(logfile)
    return {
        "path": "propaq", "file": os.path.basename(path), "steps": steps, "threads": threads,
        "wall_s": round(best, 4), "n_terms": result.n_terms[-1],
        "terms_below_cutoff": result.terms_below_cutoff,
        "expectation_value": result.expectation_value,
        "scan_s": p["scan_s"], "absorb_s": p["absorb_s"], "claims_s": p["claims_s"],
        "visited": p["visited"], "emitted": p["emitted"], "declined": p["declined"],
        "exchange_hits": p["exchange_hits"], "inline_positions": p["inline_positions"],
        "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
    }


if __name__ == "__main__":
    threads = [int(t) for t in sys.argv[1].split(",")]
    for path in sys.argv[2:]:
        for t in threads:
            print(json.dumps(run(path, t)), flush=True)
