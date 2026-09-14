###
##
# Saves one of the problems of `problems.jl` in the JSON layout of `propaq-benchmarks`, so that
# `run_propaq.py` and `papercircuits.jl` can run Propaq and the library over it.
#
#   julia --project=. propaq/bench/exportcircuit.jl tilted 20 circuits/tilted_steps20.json
##
###

using PauliPropagation
using JSON

include(joinpath(@__DIR__, "problems.jl"))

function export_problem(name::String, nsteps::Int, path::String)
    if name == "tilted"
        circuit, thetas = tiltedising(6, 6, nsteps)
        obs = tiltedisingobservable(6, 6)
    elseif name == "tfim2d100"
        circuit, thetas = tfim2d(10, 10, nsteps)
        obs = tfim2dobservable(10, 10)
    else
        throw(ArgumentError("unknown problem $name"))
    end
    nq = nqubits(obs)

    gates = Dict{String,Any}[]
    theta_it = Iterators.Stateful(thetas)
    for gate in circuit
        gate isa PauliRotation || throw(ArgumentError("only Pauli rotations are exported"))
        name = "r" * lowercase(join(String.(gate.symbols)))
        push!(gates, Dict("name" => name, "qubits" => gate.qinds .- 1, "angle" => popfirst!(theta_it)))
    end

    # Qiskit labels read right to left
    label = fill('I', nq)
    for (q, pauli) in enumerate(obs.term |> x -> [getpauli(x, k) for k in 1:nq])
        pauli == 0 && continue
        label[nq-q+1] = ('X', 'Y', 'Z')[pauli]
    end

    d = Dict("problem" => name, "n_qubits" => nq, "params" => Dict("steps" => nsteps), "initial_state" => 0,
        "gates" => gates, "observable" => Dict("paulis" => [String(label)], "coeffs" => [1.0]))
    open(path, "w") do io
        JSON.print(io, d)
    end
    println("saved $path ($nq qubits, $(length(gates)) gates)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    export_problem(ARGS[1], parse(Int, ARGS[2]), ARGS[3])
end
