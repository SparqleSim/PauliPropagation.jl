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

@inline _scale_if_needed(coeff, base) = isone(coeff) ? base : Scale(coeff, base)

const _GATES_BY_PG = (X, Y, Z)

function _pauli_kron_base(n::Int, term::Integer, ::Val{0})
    return put(n, 1 => I2)
end

function _pauli_kron_base(n::Int, term::Integer, ::Val{1})
    @inbounds for i in 1:n
        p = PP.getpauli(term, i)
        if p == 1
            return put(n, i => X)
        elseif p == 2
            return put(n, i => Y)
        elseif p == 3
            return put(n, i => Z)
        end
    end
    error("invalid weight-1 Pauli term")
end

function _generated_kron_scan(W::Int)
    loc_decls = [:( $(Symbol("loc_", k)) = 0 ) for k in 1:W]
    pg_decls = [:( $(Symbol("pg_", k)) = 0 ) for k in 1:W]
    assign_exprs = [
        quote
            if j == $k
                $(Symbol("loc_", k)) = i
                $(Symbol("pg_", k)) = p
            end
        end for k in 1:W
    ]
    return quote
        $(loc_decls...)
        $(pg_decls...)
        j = 0
        @inbounds for i in 1:n
            p = PP.getpauli(term, i)
            p == 0 && continue
            j += 1
            $(assign_exprs...)
        end
    end
end

function _generated_kron_unrolled_return(W::Int)
    ex = :(error("invalid Pauli term with weight $W"))
    for combo in Iterators.product(ntuple(_ -> 1:3, W)...)
        pairs = [:( $(Symbol("loc_", k)) => $(_GATES_BY_PG[combo[k]])) for k in 1:W]
        cond = W == 1 ? :($(Symbol("pg_", 1)) == $(combo[1])) :
               Expr(:&&, [:( $(Symbol("pg_", k)) == $(combo[k])) for k in 1:W]...)
        ex = Expr(:if, cond, :(return kron(n, $(pairs...))), ex)
    end
    return ex
end

"""
    _pauli_kron_base(n, term, ::Val{W})

Build a type-stable `PutBlock` or `KronBlock` for an integer Pauli string with Hamming weight `W`.
No heap arrays are allocated; gate locations are collected in generated locals for `W ≥ 2`.
"""
@generated function _pauli_kron_base(n::Int, term::Integer, ::Val{2})
    scan = _generated_kron_scan(2)
    ret = _generated_kron_unrolled_return(2)
    return quote
        $scan
        return $ret
    end
end

@generated function _pauli_kron_base(n::Int, term::Integer, ::Val{W}) where {W}
    W >= 3 || return :(error("internal error: weight $W should use a dedicated method"))
    scan = _generated_kron_scan(W)
    gate_assigns = [
        quote
            if $(Symbol("pg_", k)) == 1
                $(Symbol("g_", k)) = X
            elseif $(Symbol("pg_", k)) == 2
                $(Symbol("g_", k)) = Y
            else
                $(Symbol("g_", k)) = Z
            end
        end for k in 1:W
    ]
    kron_pairs = [:( $(Symbol("loc_", k)) => $(Symbol("g_", k)) ) for k in 1:W]
    return quote
        $scan
        $(gate_assigns...)
        return kron(n, $(kron_pairs...))
    end
end

"""
    pauli_term_to_yao(n::Int, term::Integer, coeff=1)
    pauli_term_to_yao(n::Int, term::Integer, coeff, ::Val{W})

Build a Yao observable block from an integer Pauli string on `n` qubits and scalar `coeff`.

Pass `Val(W)` when the Hamming weight is known (e.g. fixed `max_weight` batches) for a
fully specialized `_pauli_kron_base` call. `W` must match `countweight(term)`.
"""
function pauli_term_to_yao(n::Int, term::Integer, coeff, ::Val{W}) where {W}
    PP.countweight(term) == W ||
        throw(ArgumentError(
            "Pauli term weight $(PP.countweight(term)) does not match Val{$W}."
        ))
    return _scale_if_needed(coeff, _pauli_kron_base(n, term, Val(W)))
end

function pauli_term_to_yao(n::Int, term::Integer, coeff=1)
    return pauli_term_to_yao(n, term, coeff, Val(PP.countweight(term)))
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
function PauliPropagation.paulipropagation2yao(n::Integer, circ, thetas)
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
