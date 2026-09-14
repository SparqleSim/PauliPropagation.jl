###
##
# Clifford deferral: a Clifford gate is composed into a frame instead of being applied to every
# term, and every later rotation has its generator conjugated through the frame before it runs.
# The frame is applied to the terms once, at the end.
#
# The frame `f` is the map the deferred Clifford gates would have applied to each term, held as the
# images of the `X` and `Z` generators of every qubit, with their signs, together with the images
# under the inverse map. A rotation of the actual sum `f(S)` by generator `G` is the same as `f`
# applied to a rotation of `S` by `f⁻¹(G)`, so the stored sum is rotated by the conjugated
# generator instead. This is what Propaq's `CliffordTableau` does.
##
###

using PauliPropagation
using PauliPropagation: CliffordGate, PauliRotation, clifford_map, transposecliffordmap, pauliprod, getpauli, setpauli, symboltoint

"""
    CliffordFrame(TT, nqubits)

The identity frame over `nqubits` qubits, for Pauli strings of integer type `TT`. `images[2q-1]`
and `images[2q]` are the images of `X_q` and `Z_q` as `(pstr, sign)`, `inverses` the same under the
inverse map.
"""
struct CliffordFrame{TT}
    nqubits::Int
    images::Vector{Tuple{TT,Int}}
    inverses::Vector{Tuple{TT,Int}}
end

function CliffordFrame(::Type{TT}, nqubits::Int) where {TT}
    generators = [(setpauli(zero(TT), pauli, q), 1) for q in 1:nqubits for pauli in (:X, :Z)]
    return CliffordFrame{TT}(nqubits, generators, copy(generators))
end

isidentity(frame::CliffordFrame) = all(((pstr, sign),) -> sign == 1, frame.images) &&
    all(k -> frame.images[k][1] == frame.inverses[k][1], eachindex(frame.images))

# The image of `pstr` under the map with the given generator images: the product of the images of
# its factors, `Y = i X Z`, with the phase carried along. The phase of a Hermitian image is ±1.
function _mapthrough(rows::Vector{Tuple{TT,Int}}, pstr::TT, nqubits::Int) where {TT}
    out = zero(TT)
    phase = Complex(1.0)
    for q in 1:nqubits
        pauli = getpauli(pstr, q)
        pauli == 0 && continue
        if pauli == 1 || pauli == 2
            image, sign = rows[2q-1]
            out, s = pauliprod(out, image)
            phase *= s * sign
        end
        if pauli == 3 || pauli == 2
            image, sign = rows[2q]
            out, s = pauliprod(out, image)
            phase *= s * sign
        end
        pauli == 2 && (phase *= im)
    end
    abs(imag(phase)) < 1e-12 || error("the image of a Pauli string came out non-Hermitian")
    return out, Int(round(real(phase)))
end

"""
    mapterm(frame, pstr)

The image of `pstr` under the frame, as `(pstr, sign)`: what the deferred Clifford gates would have
made of the term.
"""
mapterm(frame::CliffordFrame{TT}, pstr::TT) where {TT} = _mapthrough(frame.images, pstr, frame.nqubits)

"""
    conjugategenerator(frame, generator)

The generator a rotation has to use on the stored sum so that it acts on the actual sum as
`generator` would: the image under the inverse frame, as `(pstr, sign)`.
"""
conjugategenerator(frame::CliffordFrame{TT}, generator::TT) where {TT} = _mapthrough(frame.inverses, generator, frame.nqubits)

"""
    compose!(frame, gate::CliffordGate)

Defer `gate`: the frame becomes `gate` applied after everything it already holds, so the images
go through the gate's own map and the inverse images through the transposed one.
"""
function compose!(frame::CliffordFrame{TT}, gate::CliffordGate) where {TT}
    forward = clifford_map[gate.symbol]
    backward = transposecliffordmap(forward)

    for k in eachindex(frame.images)
        pstr, sign = frame.images[k]
        (new_pstr, new_sign) = _applyclifford(gate, pstr, forward)
        frame.images[k] = (new_pstr, sign * new_sign)
    end

    # f_new⁻¹ = f⁻¹ ∘ c⁻¹: take each generator through the inverse gate, then through the old inverse
    old_inverses = copy(frame.inverses)
    for q in 1:frame.nqubits, (j, pauli) in enumerate((:X, :Z))
        k = 2q - 2 + j
        pstr, sign = _applyclifford(gate, setpauli(zero(TT), pauli, q), backward)
        image, image_sign = _mapthrough(old_inverses, pstr, frame.nqubits)
        frame.inverses[k] = (image, sign * image_sign)
    end

    return frame
end

# the library's own single-term Clifford application, through the given lookup map
function _applyclifford(gate::CliffordGate, pstr::TT, lookup_map) where {TT}
    ((new_pstr, sign),) = PauliPropagation.PropagationBase.apply(gate, pstr, 1, lookup_map)
    return new_pstr, sign
end
