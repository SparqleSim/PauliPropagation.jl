# Shared Yao comparison helpers for QASM tests.
# Include only from Yao-backed test files (Yao is a test-only dependency).

using Test
using PauliPropagation
using YaoBlocks
using YaoArrayRegister: zero_state, expect, apply!

include("qasm_common.jl")

"""
    _yao_unitary_circuit(task_or_circuit)

Return a unitary Yao circuit. Accepts a `SimulationTask` (uses `.circuit`) or a block.
Measurement blocks are stripped when present.
"""
function _yao_unitary_circuit(task_or_circuit)
    circ = hasproperty(task_or_circuit, :circuit) ? task_or_circuit.circuit : task_or_circuit
    if circ isa ChainBlock
        kept = filter(b -> !(b isa Measure), collect(subblocks(circ)))
        return chain(nqubits(circ), kept...)
    end
    return circ
end

function _yao_qasm_ext_available()
    return Base.get_extension(YaoBlocks, :OpenQASMExt) !== nothing
end

function _parse_qasm_with_yao(qasm_content::AbstractString)
    _yao_qasm_ext_available() || error(
        "YaoBlocks.OpenQASMExt is not loaded. " *
        "Released YaoBlocks may not ship OpenQASM support yet; " *
        "use `_assert_qasm_pp_matches_yao_circuit` with a hand-built Yao block.",
    )
    return parseblock(String(qasm_content))
end

function _yao_overlap_with_zero(nq::Int, yao_circuit, obs; prep_yao=nothing)
    reg = zero_state(nq)
    if prep_yao !== nothing
        apply!(reg, prep_yao)
    end
    apply!(reg, yao_circuit)
    return real(expect(paulipropagation2yao(obs), reg))
end

"""
    _assert_pp_matches_yao_circuit(nq, circuit_pp, thetas_pp, yao_circuit; kwargs...)

Compare PauliPropagation propagation against Yao simulation for the same observables
on `|0…0⟩` (optionally after a shared state-preparation probe).
"""
function _assert_pp_matches_yao_circuit(
    nq::Int,
    circuit_pp,
    thetas_pp,
    yao_circuit;
    observables=_default_observables(nq),
    prep_pp=Any[],
    prep_thetas=Float64[],
    prep_yao=nothing,
    atol=1e-10,
    min_abs_coeff=0,
)
    full_pp = vcat(prep_pp, collect(circuit_pp))
    full_thetas = vcat(prep_thetas, collect(thetas_pp))
    for obs in observables
        pp_val = _pp_overlap_with_zero(full_pp, full_thetas, obs; min_abs_coeff)
        yao_val = _yao_overlap_with_zero(nq, yao_circuit, obs; prep_yao)
        @test isapprox(pp_val, yao_val; atol)
    end
end

"""
    _assert_qasm_pp_matches_yao_circuit(nq, gate_line, yao_circuit; kwargs...)

Parse `gate_line` with PauliPropagation and compare against a hand-built Yao circuit.
This is the primary Yao oracle for released Yao/YaoBlocks (and for gates Yao cannot parse).
"""
function _assert_qasm_pp_matches_yao_circuit(
    nq::Int,
    gate_line::AbstractString,
    yao_circuit;
    observables=_default_observables(nq),
    prep_pp=Any[],
    prep_thetas=Float64[],
    prep_yao=nothing,
    atol=1e-10,
    min_abs_coeff=0,
)
    nq_parsed, circuit_pp, thetas_pp = _readqasm_single_gate(nq, gate_line)
    @test nq_parsed == nq
    _assert_pp_matches_yao_circuit(
        nq,
        circuit_pp,
        thetas_pp,
        yao_circuit;
        observables,
        prep_pp,
        prep_thetas,
        prep_yao,
        atol,
        min_abs_coeff,
    )
end

"""
    _assert_qasm_pp_matches_yao_parseblock(nq, gate_line; kwargs...)

Parse the same QASM independently with PauliPropagation and Yao `parseblock`, then
compare expectation values.

Requires a YaoBlocks build that ships `OpenQASMExt`. Prefer
`_assert_qasm_pp_matches_yao_circuit` when that extension is unavailable.
"""
function _assert_qasm_pp_matches_yao_parseblock(
    nq::Int,
    gate_line::AbstractString;
    observables=_default_observables(nq),
    prep_pp=Any[],
    prep_thetas=Float64[],
    prep_yao=nothing,
    atol=1e-10,
    min_abs_coeff=0,
)
    qasm = _qasm_program_string(nq, gate_line)
    nq_parsed, circuit_pp, thetas_pp = _readqasm_program(qasm)
    @test nq_parsed == nq
    yao_circuit = _yao_unitary_circuit(_parse_qasm_with_yao(qasm))
    _assert_pp_matches_yao_circuit(
        nq,
        circuit_pp,
        thetas_pp,
        yao_circuit;
        observables,
        prep_pp,
        prep_thetas,
        prep_yao,
        atol,
        min_abs_coeff,
    )
end
