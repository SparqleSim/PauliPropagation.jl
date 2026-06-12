module PauliPropagationYao

using PauliPropagation
const PP = PauliPropagation
using YaoBlocks
using YaoBlocks.ConstGate: I2, X, Y, Z, H, S, T

const _symbol_yao_map = Dict{Symbol, Any}(
    :I => I2,
    :X => X,
    :Y => Y,
    :Z => Z,
    :H => H,
    :S => S,
    :T => T,
    :SX => Rx(π / 2),
    :SY => Ry(π / 2),
)

function _symbol_to_yao(sym::Symbol)
    haskey(_symbol_yao_map, sym) ||
        throw(ArgumentError("Unsupported Pauli symbol for Yao conversion: $sym"))
    return _symbol_yao_map[sym]
end

function _clifford_to_yao!(c::ChainBlock, ::Val{:CNOT}, qinds)
    length(qinds) == 2 ||
        throw(ArgumentError("Controlled gates should have exactly 2 qubits"))
    ctrl_loc, target_loc = qinds
    push!(c, control(c.n, ctrl_loc, target_loc => X))
    return c
end

function _clifford_to_yao!(c::ChainBlock, ::Val{:CZ}, qinds)
    length(qinds) == 2 ||
        throw(ArgumentError("Controlled gates should have exactly 2 qubits"))
    ctrl_loc, target_loc = qinds
    push!(c, control(c.n, ctrl_loc, target_loc => Z))
    return c
end

function _clifford_to_yao!(c::ChainBlock, ::Val{:ZZpihalf}, qinds)
    length(qinds) == 2 ||
        throw(ArgumentError("ZZpihalf gate should have exactly 2 qubits"))
    push!(c, put(c.n, (qinds...,) => rot(kron(Z, Z), π / 2)))
    return c
end

function _clifford_to_yao!(c::ChainBlock, ::Val{:SWAP}, qinds)
    length(qinds) == 2 ||
        throw(ArgumentError("SWAP gate should have exactly 2 qubits"))
    push!(c, swap(c.n, qinds...))
    return c
end

function _clifford_to_yao!(c::ChainBlock, sym::Val{S}, qinds) where {S}
    push!(c, put(c.n, (qinds...,) => _symbol_to_yao(S)))
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.FrozenGate)
    _pauli_to_yao_gate!(c, g.gate, g.parameter)
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.CliffordGate)
    _clifford_to_yao!(c, Val(g.symbol), g.qinds)
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.PauliRotation, θ::Number)
    ops = [_symbol_to_yao(s) for s in g.symbols]
    push!(c, put(c.n, (g.qinds...,) => rot(kron(ops...), θ)))
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.DepolarizingNoise, p::Number)
    length(g.qind) == 1 ||
        throw(ArgumentError("Depolarizing noise should be applied to a single qubit"))
    push!(c, put(c.n, (g.qind...,) => quantum_channel(DepolarizingError(1, p))))
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.PauliXNoise, p::Number)
    push!(c, put(c.n, (g.qind...,) => MixedUnitaryChannel(PauliError(p / 2, 0.0, 0.0))))
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.PauliYNoise, p::Number)
    push!(c, put(c.n, (g.qind...,) => MixedUnitaryChannel(PauliError(0.0, p / 2, 0.0))))
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.PauliZNoise, p::Number)
    push!(c, put(c.n, (g.qind...,) => MixedUnitaryChannel(PauliError(0.0, 0.0, p / 2))))
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.AmplitudeDampingNoise, γ::Number)
    push!(c, quantum_channel(AmplitudeDampingError(γ)))
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.ParametrizedGate, ::Number)
    error("Unsupported parametrized gate for Yao conversion: $(typeof(g))")
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.Gate)
    error("Unsupported gate type for Yao conversion: $(typeof(g))")
end

@inline _scale_if_needed(coeff, base) = isone(coeff) ? base : coeff * base

@inline _pauli_gate(p::Integer) = p == 1 ? X : p == 2 ? Y : Z

"""
    pauli_term_to_yao(n::Int, term::Integer, coeff=1)

Build a Yao observable block from an integer Pauli string on `n` qubits and scalar `coeff`.

Uses `put` for weight-1 strings and `kron` for higher weight (Yao's recommended form for
Pauli observables; avoids `ChainBlock` products of commuting `PutBlock`s).
"""
function pauli_term_to_yao(n::Int, term::Integer, coeff=1)
    if term == 0
        return _scale_if_needed(coeff, put(n, 1 => I2))
    end

    w = PP.countweight(term)
    if w == 1
        @inbounds for i in 1:n
            p = PP.getpauli(term, i)
            if p != 0
                return _scale_if_needed(coeff, put(n, i => _pauli_gate(p)))
            end
        end
        error("invalid weight-1 Pauli term")
    end

    pairs = Vector{Pair{Int, YaoBlocks.ConstGate.PauliGate}}(undef, w)
    j = 0
    @inbounds for i in 1:n
        p = PP.getpauli(term, i)
        if p != 0
            j += 1
            pairs[j] = i => _pauli_gate(p)
        end
    end
    return _scale_if_needed(coeff, kron(n, pairs...))
end

"""
    paulipropagation2yao(pstr::PauliString)

Convert a `PauliString` to a Yao observable (`PutBlock`, `KronBlock`, or `Scale` thereof).
"""
function paulipropagation2yao(pstr::PP.PauliString)
    return pauli_term_to_yao(pstr.nqubits, pstr.term, pstr.coeff)
end

"""
    paulipropagation2yao(psum::AbstractPauliSum)

Convert an `AbstractPauliSum` to a Yao observable (`Add` of Pauli terms, possibly scaled).
"""
function paulipropagation2yao(psum::PP.AbstractPauliSum)
    m = length(psum)
    m == 0 && throw(ArgumentError("Cannot convert empty Pauli sum to Yao observable."))
    n = PP.nqubits(psum)
    if m == 1
        pauli, coeff = only(zip(PP.paulis(psum), PP.coefficients(psum)))
        return pauli_term_to_yao(n, pauli, coeff)
    end
    blocks = Vector{AbstractBlock{2}}(undef, m)
    i = 0
    @inbounds for (pauli, coeff) in zip(PP.paulis(psum), PP.coefficients(psum))
        i += 1
        blocks[i] = pauli_term_to_yao(n, pauli, coeff)
    end
    return Add(n, blocks)
end

"""
    paulipropagation2yao(n::Integer, circ, thetas)

Convert a PauliPropagation circuit to a Yao `ChainBlock`.

`thetas` must have one entry per `ParametrizedGate` in `circ` (see `countparameters`), matching
`PropagationBase.propagate!`. `FrozenGate` is a `StaticGate` with a bundled parameter and does not
consume entries from `thetas`.
"""
function paulipropagation2yao(n::Integer, circ, thetas)
    nparams = PP.countparameters(circ)
    nparams == length(thetas) ||
        throw(ArgumentError(
            "The number of parameters must match the number of parametrized gates in the circuit. " *
            "countparameters(circ)=$nparams, length(thetas)=$(length(thetas))."
        ))
    thetas = collect(thetas)
    c = chain(Int(n))
    for g in circ
        if g isa PP.ParametrizedGate
            _pauli_to_yao_gate!(c, g, popfirst!(thetas))
        else
            _pauli_to_yao_gate!(c, g)
        end
    end
    return c
end

end
