###
##
# Pauli gates applied to a MultiPauliSum. A gate that moves every term takes the generic
# `applytoallzones!`, and a gate that branches by a fixed mask takes `applyxorbranch!`.
##
###

PropagationBase.staysinzone(::PauliNoise) = true


### Gates that move every term

PropagationBase.applytoall!(gate, prop_cache::MultiPauliPropagationCache, args...; kwargs...) =
    applytoallzones!(gate, prop_cache, args...; kwargs...)

# these three gates carry their own `applytoall!` for any `AbstractPauliPropagationCache`, which is
# neither more nor less specific than the method above
function PropagationBase.applytoall!(gate::CliffordGate, prop_cache::MultiPauliPropagationCache; kwargs...)
    _check_qind_range(nqubits(prop_cache), gate.qinds)
    return applytoallzones!(gate, prop_cache, clifford_map[gate.symbol]; kwargs...)
end

PropagationBase.applytoall!(gate::TGate, prop_cache::MultiPauliPropagationCache; kwargs...) =
    applytoall!(PauliRotation(:Z, gate.qind), prop_cache, π / 4; kwargs...)

PropagationBase.applytoall!(gate::FrozenGate, prop_cache::MultiPauliPropagationCache; kwargs...) =
    applytoall!(gate.gate, prop_cache, gate.parameter; kwargs...)


### Gates that branch by a fixed mask

"""
    applytoall!(gate::PauliRotation, prop_cache::MultiPauliPropagationCache, theta; kwargs...)

Every zone rescales the Pauli strings that branch and parks the Pauli strings they make in its outbox.
"""
function PropagationBase.applytoall!(gate::PauliRotation, prop_cache::MultiPauliPropagationCache, theta; kwargs...)
    _check_qind_range(nqubits(prop_cache), gate.qinds)

    gate_mask = symboltoint(paulitype(prop_cache), gate.symbols, gate.qinds)
    cos_val = cos(theta)
    sin_val = sin(theta)

    return applyxorbranch!(prop_cache, gate_mask; kwargs...) do pstr, coeff
        commutes(gate_mask, pstr) && return nothing

        _, sign = paulirotationproduct(gate_mask, pstr)
        return (coeff * cos_val, coeff * sin_val * sign, true)
    end
end

"""
    applytoall!(gate::ImaginaryPauliRotation, prop_cache::MultiPauliPropagationCache, tau; kwargs...)

Like the `PauliRotation` overload, except that imaginary Pauli rotations branch upon commutation.
"""
function PropagationBase.applytoall!(gate::ImaginaryPauliRotation, prop_cache::MultiPauliPropagationCache, tau; kwargs...)
    _check_qind_range(nqubits(prop_cache), gate.qinds)

    gate_mask = symboltoint(paulitype(prop_cache), gate.symbols, gate.qinds)
    cosh_val = cosh(tau)
    sinh_val = sinh(tau)

    return applyxorbranch!(prop_cache, gate_mask; kwargs...) do pstr, coeff
        commutes(gate_mask, pstr) || return nothing

        _, sign = paulirotationproduct(gate_mask, pstr)
        return (coeff * cosh_val, coeff * sinh_val * sign, true)
    end
end

"""
    applytoall!(gate::AmplitudeDampingNoise, prop_cache::MultiPauliPropagationCache, gamma; kwargs...)

Every zone rescales its Pauli strings and parks the ones that its Z Paulis damp into with their owners.
"""
function PropagationBase.applytoall!(gate::AmplitudeDampingNoise, prop_cache::MultiPauliPropagationCache, gamma; kwargs...)
    _check_qind_range(nqubits(prop_cache), gate.qind)
    _check_noise_strength(AmplitudeDampingNoise, gamma)

    qind = gate.qind
    z_mask = symboltoint(paulitype(prop_cache), :Z, qind)
    damp_val = sqrt(1 - gamma)

    return applyxorbranch!(prop_cache, z_mask; kwargs...) do pstr, coeff
        pauli = getpauli(pstr, qind)
        pauli == 0 && return nothing

        # Z damps into the identity on that site, X and Y are only rescaled
        pauli == 3 && return (coeff * (1 - gamma), coeff * gamma, true)
        return (coeff * damp_val, coeff, false)
    end
end
