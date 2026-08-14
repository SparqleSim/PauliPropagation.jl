# Shared helpers for QASM tests.

using Test
using PauliPropagation
using PauliPropagation.OpenQASMInterface

"""
    _qasm_program_string(nq, gate_body) -> String

Build a minimal OpenQASM 2.0 program with one `qreg` and the given gate body.
"""
function _qasm_program_string(nq::Int, gate_body::AbstractString)
    return """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[$nq];
    $gate_body
    """
end

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
    return _readqasm_program(_qasm_program_string(nq, gate_line))
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

function _pp_overlap_with_zero(circuit, thetas, obs; min_abs_coeff=0)
    return overlapwithzero(propagate(circuit, obs, thetas; min_abs_coeff))
end
