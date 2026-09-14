"""
Saves every circuit the README measures, in the JSON layout of `propaq-benchmarks`, so that
`papercircuits.jl`, `run_propaq.py` and the other scripts here can run over the same gate lists.

    pip install qiskit~=2.4.1 numpy
    PYTHONPATH=propaq/propaq-benchmarks/propaq-benchmarks/src python3 propaq/bench/generatecircuits.py circuits/

The 6x6 and 10x10 TFIM circuits are the benchmark repo's own `ising_trotter_problem`, passed
through Qiskit's transpiler as the paper's are. The `6x6cx` circuits are the 6x6 ones with `rzz`
decomposed into `cx rz cx`, which is how a transpiled circuit reaches the library. The Clifford+T
circuits are those of the repo's Clifford deferral experiment. The notebook's tilted-field circuit
is exported from Julia by `exportcircuit.jl` instead.
"""

import os
import random
import sys

from propaq_benchmarks import problems_qubit
from propaq_benchmarks.circuit_ir import qiskit_to_ir
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

TFIM_6X6_STEPS = (8, 10, 12, 14, 16, 17, 18, 19, 20)
TFIM_10X10_STEPS = (10, 12, 14)
CX_STEPS = (10, 12)
CLIFFORD_T_DENSITIES = (0.1, 0.25)


def clifford_t_circuit(n_qubits, total_layers, p, seed):
    rng = random.Random(seed)
    qc = QuantumCircuit(n_qubits)
    for _ in range(total_layers):
        for q in range(n_qubits):
            getattr(qc, rng.choice(["h", "s", "x", "sdg"]))(q)
        qubits = list(range(n_qubits))
        rng.shuffle(qubits)
        for i in range(0, n_qubits - 1, 2):
            qc.cx(qubits[i], qubits[i + 1])
        if rng.random() < p:
            qc.t(rng.randrange(n_qubits))
    return qc


def save(ir, path):
    ir.save(path)
    print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")


def main(out):
    os.makedirs(out, exist_ok=True)
    for nx, ny, steps in [(6, 6, s) for s in TFIM_6X6_STEPS] + [(10, 10, s) for s in TFIM_10X10_STEPS]:
        save(problems_qubit.ising_trotter_problem(nx=nx, ny=ny, steps=steps), f"{out}/{nx}x{ny}_steps{steps}.json")

    for steps in CX_STEPS:
        ir = problems_qubit.ising_trotter_problem(nx=6, ny=6, steps=steps)
        qc = transpile(ir.to_qiskit(), basis_gates=["cx", "rz", "rx", "h"], optimization_level=1, seed_transpiler=0)
        save(qiskit_to_ir(qc, ir.observable.to_sparse_pauli_op(), "ising_trotter_cx", dict(ir.params), canonicalize_circuit=False),
             f"{out}/6x6cx_steps{steps}.json")

    observable = SparsePauliOp("Z" + "I" * 63)
    for p in CLIFFORD_T_DENSITIES:
        qc = clifford_t_circuit(64, 80, p, 123)
        params = {"n_qubits": 64, "total_layers": 80, "p": p, "seed": 123, "coeff_cutoff": 1e-9, "steps": 1}
        save(qiskit_to_ir(qc, observable, "clifford_deferral", params, canonicalize_circuit=False), f"{out}/clifford_p{p}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "circuits")
