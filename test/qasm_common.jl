# Shared helpers for QASM tests.

using PauliPropagation
using PauliPropagation.OpenQASMInterface
import PauliPropagation: TransferMapGate

function _readqasm_program(qasm_content::AbstractString)
    filepath = tempname() * ".qasm"
    write(filepath, qasm_content)
    try
        return readqasm(filepath)
    finally
        rm(filepath; force=true)
    end
end

function _readqasm_single_gate(nq::Int, gate_line::AbstractString)
    qasm_content = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[$nq];
    $gate_line
    """
    return _readqasm_program(qasm_content)
end

function _default_observables(nq::Int)
    observables = PauliString[]
    for q in 1:nq, sym in (:X, :Z)
        push!(observables, PauliString(nq, sym, q))
    end
    if nq >= 2
        push!(observables, PauliString(nq, [:X, :Y], [1, 2]))
    end
    if nq >= 3
        push!(observables, PauliString(nq, [:Z, :X, :Y], [1, 2, 3]))
    end
    return observables
end

function _assert_qasm_propagation_matches_transfermap(
    nq::Int,
    gate_line::AbstractString;
    observables=_default_observables(nq),
)
    nq_parsed, circuit, thetas = _readqasm_single_gate(nq, gate_line)
    @test nq_parsed == nq
    ref_gate = TransferMapGate(totransfermap(nq_parsed, circuit, thetas), collect(1:nq_parsed))
    for obs in observables
        psum_qasm = propagate(circuit, obs, thetas; min_abs_coeff=0)
        psum_ref = propagate([ref_gate], obs; min_abs_coeff=0)
        @test psum_qasm == psum_ref
    end
end
