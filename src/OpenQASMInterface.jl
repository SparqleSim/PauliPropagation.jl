module OpenQASMInterface

using OpenQASM
# This line imports the necessary types from the main PauliPropagation module
using ..PauliPropagation: CliffordGate, PauliRotation

export readqasm

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
        
        gate_name = instruction.name # This is a String
        
        qubits = [parse(Int, q.address.str) + 1 for q in instruction.qargs]

        # --- Gate Translation Logic ---
        
        # Compare gate_name to Strings, not Symbols.
        if gate_name == "h"
            push!(circuit, CliffordGate(:H, qubits[1]))
        elseif gate_name == "x"
            push!(circuit, CliffordGate(:X, qubits[1]))
        elseif gate_name == "cx"
            push!(circuit, CliffordGate(:CNOT, qubits))
        
        # Handle Pauli Rotations (with parameters)
        elseif gate_name == "rx"
            param_token = instruction.cargs[1]
            push!(circuit, PauliRotation(:X, qubits[1]))
            push!(thetas, parse(Float64, param_token.str)) # FINAL FIX
        elseif gate_name == "ry"
            param_token = instruction.cargs[1]
            push!(circuit, PauliRotation(:Y, qubits[1]))
            push!(thetas, parse(Float64, param_token.str)) # FINAL FIX
        elseif gate_name == "rz"
            param_token = instruction.cargs[1]
            push!(circuit, PauliRotation(:Z, qubits[1]))
            push!(thetas, parse(Float64, param_token.str)) # FINAL FIX
        
        else
            error("Unsupported QASM gate: '$(gate_name)'. This operation is not implemented.")
        end
    end

    return nq, circuit, thetas
end

end # end of module