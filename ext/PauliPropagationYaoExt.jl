# ext/PauliPropagationYaoExt.jl
#
# Extension implementing paulipropagation2yao(). The reverse of Yao's yao2paulipropagation().

module PauliPropagationYaoExt

using PauliPropagation: PauliString, PauliSum, CliffordGate, TGate, PauliRotation,
                        getpauli, inttosymbol, topaulistrings
import PauliPropagation: paulipropagation2yao
using Yao: X, Y, Z, H,  Rx, Rz, Ry, I2, put, control, swap, chain, matblock, time_evolve, kron

# ---------------------------------------------------------------------------
# single-qubit Pauli symbol -> Yao single-qubit operator
# ---------------------------------------------------------------------------
_pauliop(s::Symbol) =
    s === :X ? X :
    s === :Y ? Y :
    s === :Z ? Z :
    s === :I ? I2 :
    error("paulipropagation2yao: unsupported Pauli symbol :$s")

# T and S aren't exported by Yao at top level, so build them explicitly.
const _TMAT = ComplexF64[1 0; 0 exp(im * π / 4)]   # T = diag(1, e^{iπ/4})
const _SMAT = ComplexF64[1 0; 0 im]                # S = diag(1, i)

# ---------------------------------------------------------------------------
# PauliPropagation gate -> Yao block, placed on n qubits
# ---------------------------------------------------------------------------
function _gate_to_yao(g::CliffordGate, n::Int)
    s, q = g.symbol, g.qinds
    s === :H    && return put(n, q[1] => H)
    s === :X    && return put(n, q[1] => X)
    s === :Y    && return put(n, q[1] => Y)
    s === :Z    && return put(n, q[1] => Z)
    s === :S    && return put(n, q[1] => matblock(_SMAT))
    s === :CNOT && return control(n, q[1], q[2] => X)
    s === :CZ   && return control(n, q[1], q[2] => Z)
    s === :SWAP && return swap(n, q[1], q[2])
    s === :SX   && return put(n, q[1] => Rx(π / 2))
    s === :SY   && return put(n, q[1] => Ry(π / 2))
    s === :ZZpihalf && return put(n, (q[1], q[2]) => time_evolve(kron(Z, Z), π / 4))
    error("paulipropagation2yao: unsupported CliffordGate :$s")
end

_gate_to_yao(g::TGate, n::Int) = put(n, g.qind => matblock(_TMAT))

# PauliRotation(P, θ) = exp(-iθP/2) = time_evolve(P, θ/2), placed on its qubits.
# symbols[i] acts on qinds[i]; put maps the kron factors onto the qubit tuple in order.
function _gate_to_yao(g::PauliRotation, n::Int, θ::Real)
    ops = [_pauliop(s) for s in g.symbols]
    P   = length(ops) == 1 ? ops[1] : kron(ops...)
    qs  = length(g.qinds) == 1 ? g.qinds[1] : Tuple(g.qinds)
    return put(n, qs => time_evolve(P, θ / 2))
end

# ---------------------------------------------------------------------------
# PauliString -> Yao observable  (tensor product of single-qubit Paulis × coeff)
# ---------------------------------------------------------------------------
function paulipropagation2yao(pstr::PauliString)
    n = pstr.nqubits
    blocks = []
    for q in 1:n
        sym = inttosymbol(getpauli(pstr.term, q))
        sym === :I || push!(blocks, put(n, q => _pauliop(sym)))
    end
    op = isempty(blocks)     ? put(n, 1 => I2) :
         length(blocks) == 1 ? only(blocks) :
                               chain(n, blocks...)
    return isone(pstr.coeff) ? op : pstr.coeff * op
end

# ---------------------------------------------------------------------------
# PauliSum -> Yao observable:  sum coeff * (Pauli term)
# ---------------------------------------------------------------------------
function paulipropagation2yao(psum::PauliSum)
    terms = topaulistrings(psum)
    isempty(terms) && return 0 * put(psum.nqubits, 1 => I2)
    return sum(paulipropagation2yao(p) for p in terms)
end

# ---------------------------------------------------------------------------
# static circuit -> Yao chain (no parametrised gates)
# ---------------------------------------------------------------------------
function paulipropagation2yao(circuit::AbstractVector, nqubits::Int)
    return chain(nqubits, (_gate_to_yao(g, nqubits) for g in circuit)...)
end

# ---------------------------------------------------------------------------
# parametrised circuit -> Yao chain
#   thetas are consumed by PauliRotation gates in circuit order, exactly as propagate does
# ---------------------------------------------------------------------------
function paulipropagation2yao(circuit::AbstractVector, nqubits::Int, θs::AbstractVector)
    n_rot = count(g -> g isa PauliRotation, circuit)
    length(θs) == n_rot ||
        throw(ArgumentError("expected $n_rot angle(s) for the PauliRotation gates, got $(length(θs))"))
    blocks = []
    i = 1
    for g in circuit
        if g isa PauliRotation
            push!(blocks, _gate_to_yao(g, nqubits, θs[i]))
            i += 1
        else
            push!(blocks, _gate_to_yao(g, nqubits))
        end
    end
    return chain(nqubits, blocks...)
end

end # module