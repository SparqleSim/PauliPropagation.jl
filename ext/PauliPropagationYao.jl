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

function _clifford_to_yao!(c::ChainBlock, sym::Val{S}, qinds) where {S}
    push!(c, put(c.n, (qinds...,) => _symbol_to_yao(S)))
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.CliffordGate, ::Nothing)
    _clifford_to_yao!(c, Val(g.symbol), g.qinds)
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.PauliRotation, parameter)
    ops = [_symbol_to_yao(s) for s in g.symbols]
    push!(c, put(c.n, (g.qinds...,) => rot(kron(ops...), parameter)))
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.DepolarizingNoise, parameter)
    length(g.qind) == 1 ||
        throw(ArgumentError("Depolarizing noise should be applied to a single qubit"))
    push!(c, put(c.n, (g.qind...,) => quantum_channel(DepolarizingError(1, parameter))))
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.PauliXNoise, parameter)
    push!(c, put(c.n, (g.qind...,) => MixedUnitaryChannel(PauliError(parameter / 2, 0.0, 0.0))))
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.PauliYNoise, parameter)
    push!(c, put(c.n, (g.qind...,) => MixedUnitaryChannel(PauliError(0.0, parameter / 2, 0.0))))
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.PauliZNoise, parameter)
    push!(c, put(c.n, (g.qind...,) => MixedUnitaryChannel(PauliError(0.0, 0.0, parameter / 2))))
    return c
end

function _pauli_to_yao_gate!(c::ChainBlock, g::PP.AmplitudeDampingNoise, parameter)
    push!(c, quantum_channel(AmplitudeDampingError(parameter)))
    return c
end

"""
    pauli_term_to_yao(n::Int, term::Integer, coeff=1)

Build a Yao observable block from an integer Pauli string on `n` qubits and scalar `coeff`.
"""
function pauli_term_to_yao(n::Int, term::Integer, coeff=1)
    syms = PP.inttosymbol(term, n)
    active = [(q, _symbol_to_yao(s)) for (q, s) in enumerate(syms) if s != :I]
    base = if isempty(active)
        put(n, 1 => I2)
    elseif length(active) == 1
        q, op = only(active)
        put(n, q => op)
    else
        kron(n, [q => op for (q, op) in active]...)
    end
    return isone(coeff) ? base : Scale(coeff, base)
end

"""
    paulipropagation2yao(pstr::PauliString)

Convert a `PauliString` to a Yao observable (`PutBlock`, `KronBlock`, or `Scale` thereof).
"""
function PauliPropagation.paulipropagation2yao(pstr::PP.PauliString)
    return pauli_term_to_yao(pstr.nqubits, pstr.term, pstr.coeff)
end

"""
    paulipropagation2yao(psum::AbstractPauliSum)

Convert an `AbstractPauliSum` to a Yao observable (`Add` of Pauli terms, possibly scaled).
"""
function PauliPropagation.paulipropagation2yao(psum::PP.AbstractPauliSum)
    length(psum) == 0 && throw(ArgumentError("Cannot convert empty Pauli sum to Yao observable."))
    n = PP.nqubits(psum)
    terms = zip(PP.paulis(psum), PP.coefficients(psum))
    if length(psum) == 1
        pauli, coeff = only(terms)
        return pauli_term_to_yao(n, pauli, coeff)
    end
    return sum(pauli_term_to_yao(n, pauli, coeff) for (pauli, coeff) in terms)
end

"""
    paulipropagation2yao(n::Integer, circ, thetas)

Convert a PauliPropagation circuit to a Yao `ChainBlock`.

`thetas` must have one entry per `ParametrizedGate` in `circ` (see `countparameters`).
`FrozenGate` entries carry their parameter and do not consume `thetas`.
"""
function PauliPropagation.paulipropagation2yao(n::Integer, circ, thetas)
    PP.countparameters(circ) == length(thetas) ||
        throw(ArgumentError(
            "Expected $(PP.countparameters(circ)) parameters, got $(length(thetas))."
        ))
    thetas = collect(thetas)
    c = chain(Int(n))
    for g in circ
        if g isa PP.FrozenGate
            _pauli_to_yao_gate!(c, g.gate, g.parameter)
        elseif g isa PP.CliffordGate
            _pauli_to_yao_gate!(c, g, nothing)
        elseif g isa PP.ParametrizedGate
            _pauli_to_yao_gate!(c, g, popfirst!(thetas))
        else
            error("Unsupported gate type for Yao conversion: $(typeof(g))")
        end
    end
    return c
end

end
