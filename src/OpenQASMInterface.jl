module OpenQASMInterface

using OpenQASM
# This line imports the necessary types from the main PauliPropagation module
using ..PauliPropagation: CliffordGate, PauliRotation

export readqasm

# --- Helper functions for Value-Based Dispatch ---

# Fallback for any gate not explicitly defined
function _translate_gate(gate_name_val::Val{G}, qubits, cargs) where {G}
    gate_name = Symbol(gate_name_val) # Extract symbol from Val type
    error("Unsupported translation for QASM gate: '$(G)'. This operation is not implemented.")
end

# -- Clifford Gates (return gate, nothing for theta) --
_translate_gate(::Val{:h}, qubits, cargs) = (CliffordGate(:H, qubits[1]), nothing)
_translate_gate(::Val{:x}, qubits, cargs) = (CliffordGate(:X, qubits[1]), nothing)
_translate_gate(::Val{:cx}, qubits, cargs) = (CliffordGate(:CNOT, qubits), nothing)
# Add other supported Cliffords like :y, :z, :cz here if needed.

# -- Pauli Rotations (return gate, and the parsed theta value) --
function _translate_gate(::Val{:rx}, qubits, cargs)
    param_token = cargs[1]
    theta = parse(Float64, param_token.str)
    return (PauliRotation(:X, qubits[1]), theta)
end

function _translate_gate(::Val{:ry}, qubits, cargs)
    param_token = cargs[1]
    theta = parse(Float64, param_token.str)
    return (PauliRotation(:Y, qubits[1]), theta)
end

function _translate_gate(::Val{:rz}, qubits, cargs)
    param_token = cargs[1]
    theta = parse(Float64, param_token.str)
    return (PauliRotation(:Z, qubits[1]), theta)
end


"""
    readqasm(filepath::String) -> (Int, Vector, Vector)

Parses an OpenQASM 2.0 file and returns the number of qubits, 
a circuit compatible with PauliPropagation.jl, and a vector of gate parameters (thetas).
"""
function readqasm(filepath::String)
    qasm_content = read(filepath, String)
    parsed_program = OpenQASM.parse(qasm_content)

    statements = parsed_program.prog

    # Find the qreg declaration to determine the number of qubits
    nq = 0
    qreg_decl_idx = findfirst(x -> x isa OpenQASM.Types.RegDecl, statements)
    if qreg_decl_idx !== nothing
        size_token = statements[qreg_decl_idx].size
        nq = parse(Int, size_token.str)
    else
        error("QASM file does not contain a qubit register ('qreg') declaration.")
    end

    circuit = []
    thetas = []

    # Loop through every instruction in the extracted list
    for instruction in statements
        if !(instruction isa OpenQASM.Types.Instruction)
            continue
        end

        gate_name_str = instruction.name
        qubits = [parse(Int, q.address.str) + 1 for q in instruction.qargs]
        cargs = instruction.cargs

        # Convert the gate name string to a Val{Symbol} to enable multiple dispatch
        gate_name_val = Val(Symbol(gate_name_str))

        # Dispatch to the correct helper method based on the gate name
        gate_obj, theta = _translate_gate(gate_name_val, qubits, cargs)

        push!(circuit, gate_obj)
        if !isnothing(theta)
            push!(thetas, theta)
        end
    end

    return nq, circuit, thetas
end

end # end of module