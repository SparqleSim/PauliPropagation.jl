###
##
# This file contains specialized functions for some of our gates.
# We overload `applytoall!()` to reduce unnecessarily moving Pauli strings between `psum` and `aux_psum`.
# This usually also fixes potential type-instabilities in `apply()`.
# Both functions can be overloaded if needed.
##
###

### PAULI GATES
"""
    applytoall!(gate::PauliRotation, theta, psum, aux_psum; kwargs...)

Overload of `applytoall!` for `PauliRotation` gates and a propagating `PauliSum`.
It fixes the type-instability of the `apply()` function and reduces moving Pauli strings between `psum` and `aux_psum`.
`psum` and `aux_psum` are merged later.
"""
function PropagationBase.applytoall!(gate::PauliRotation, prop_cache::PauliPropagationCache, theta; kwargs...)
    # unpack the pauli sums
    psum = mainsum(prop_cache)
    aux_psum = auxsum(prop_cache)

    # get the bitmask of the Pauli generator
    # this allows for faster operations
    gate_mask = symboltoint(paulitype(prop_cache), gate.symbols, gate.qinds)

    # pre-compute the sine and cosine values because the are used for every Pauli string that does not commute with the gate
    cos_val = cos(theta)
    sin_val = sin(theta)
    # loop over all Pauli strings and their coefficients in the Pauli sum
    for (pstr, coeff) in psum

        if commutes(gate_mask, pstr)
            # if the gate commutes with the pauli string, do nothing
            continue
        end

        # else we know the gate will split the Pauli string into two
        coeff1 = coeff * cos_val
        new_pstr, sign = paulirotationproduct(gate_mask, pstr)
        coeff2 = coeff * sin_val * sign

        # set the coefficient of the original Pauli string
        set!(psum, pstr, coeff1)

        # set the coefficient of the new Pauli string in the aux_psum
        # we can set the coefficient because PauliRotations create non-overlapping new Pauli strings
        set!(aux_psum, new_pstr, coeff2)
    end

    return prop_cache
end


function paulirotationproduct(gate::PauliRotation, pstr::TT) where TT
    masked_gate = _tomaskedpaulirotation(gate, TT)
    return paulirotationproduct(masked_gate, pstr)
end

# TODO: completely remove MaskedPauliRotation
function paulirotationproduct(gate::MaskedPauliRotation, pstr::TT) where TT
    return paulirotationproduct(gate.generator_mask, pstr)
end

function paulirotationproduct(gate_mask::TT, pstr::TT) where TT
    new_pstr = PauliPropagation._bitpaulimultiply(gate_mask, pstr)

    # this counts the exponent of the imaginary unit in the new Pauli string
    im_count = PauliPropagation._calculatesignexponent(gate_mask, pstr)

    # now, instead of computing im^im_count followed by another im factor from the gate rules,
    # we do this in one step via a cheeky trick:
    sign = (im_count & 2) - 1
    # this is equivalent to sign = real( im * im^im_count)

    return new_pstr, sign
end

### Imaginary Pauli Rotation
"""
    applymergetruncate!(gate::ImaginaryPauliRotation, prop_cache::AbstractPauliPropagationCache, tau; normalize_coeffs=true, kwargs...)

Overload of `applymergetruncate!` for `ImaginaryPauliRotation` gates and a propagating `PauliSum`.
Applies the gate, merges the resulting Pauli sum, and truncates it.
If `normalize_coeffs=true`, the resulting Pauli sum is normalized by the coefficient of the identity Pauli string after merging.
This is useful for numerical stability when evolving states in the Schrödinger picture.
"""
function PropagationBase.applymergetruncate!(gate::ImaginaryPauliRotation, prop_cache::AbstractPauliPropagationCache, tau; normalize_coeffs=true, kwargs...)
    # normal application
    applytoall!(gate, prop_cache, tau; kwargs...)

    # normal merging
    merge!(prop_cache; kwargs...)

    # This gate assumes we are working in the Schrödinger picture evolving states
    # we normalize by the coefficient of the identity Pauli string
    # this is beneficial for numerical stability and if absolute coefficient truncation is used
    # example failure modes are if the coefficient is zero, of if it is supposed to be a number other than 1
    # these can be avoided by setting `normalize_coeffs=false`
    if normalize_coeffs
        # "getmergedcoeff" because we know there are no duplictates.
        # for array storage, the identity term will also be right at the beginning
        mult!(prop_cache, 1 / getmergedcoeff(mainsum(prop_cache), 0))
    end

    # normal truncation
    truncate!(prop_cache; kwargs...)

    return
end

function PauliPropagation.applytoall!(gate::ImaginaryPauliRotation, prop_cache::PauliPropagationCache, tau; kwargs...)
    # unpack the pauli sums
    psum = mainsum(prop_cache)
    aux_psum = auxsum(prop_cache)

    # get the bitmask of the Pauli generator
    # this allows for faster operations
    gate_mask = symboltoint(paulitype(prop_cache), gate.symbols, gate.qinds)

    # pre-compute the sinh and cosh values because they are used for every Pauli string that does not commute with the gate
    cosh_val = cosh(tau)
    sinh_val = sinh(tau)
    # loop over all Pauli strings and their coefficients in the Pauli sum
    for (pstr, coeff) in psum

        if !commutes(gate_mask, pstr)
            # imaginary Pauli rotations branch upon commutation
            continue
        end

        coeff1 = coeff * cosh_val
        # because of the imaginary time, we have take a normal product here
        # given the commutation, the sign is always real
        new_pstr, sign = imaginarypaulirotationproduct(gate_mask, pstr)
        coeff2 = coeff * sinh_val * sign

        # set the coefficient of the original Pauli string
        set!(psum, pstr, coeff1)

        # set the coefficient of the new Pauli string in the aux_psum
        # we can set the coefficient because PauliRotations create non-overlapping new Pauli strings
        set!(aux_psum, new_pstr, coeff2)
    end

    return
end

function imaginarypaulirotationproduct(gate_mask::TT, pstr::TT) where TT
    new_pstr = PauliPropagation._bitpaulimultiply(gate_mask, pstr)

    # this counts the exponent of the imaginary unit in the new Pauli string
    im_count = PauliPropagation._calculatesignexponent(gate_mask, pstr)

    # now, instead of computing real(im^im_count),
    # we do this in one step via a cheeky trick:
    sign = 1 - (im_count & 2)
    # this is equivalent to sign = real(im^im_count)
    return new_pstr, sign
end


### Clifford gates

"""
    applytoall!(gate::CliffordGate, prop_cache::AbstractPauliPropagationCache; kwargs...)

Overload of `applytoall!` for `CliffordGate`s with a propagating `PauliSum`.
Provides the Clifford lookup map to the default `applytoall!`, and `apply` functions.
"""
function PropagationBase.applytoall!(gate::CliffordGate, prop_cache::AbstractPauliPropagationCache, ; kwargs...)
    # greedy overload for Clifford gates
    # this is the most concrete function for them, but with an additional arg it will go into the generic applytoall!
    # there the apply function will receive the lookup map directly
    lookup_map = clifford_map[gate.symbol]

    applytoall!(gate, prop_cache, lookup_map; kwargs...)

    return
end

function PropagationBase.apply(gate::CliffordGate, pstr, coeff, lookup_map; kwargs...)
    # the lookup array carries the new Paulis + sign for every occuring old Pauli combination

    qinds = gate.qinds

    # this integer carries the active Paulis on its bits
    lookup_int = getpauli(pstr, qinds)

    # this integer can be used to index into the array returning the new Paulis
    # +1 because Julia is 1-indexed and lookup_int is 0-indexed
    partial_pstr, sign = lookup_map[lookup_int+1]

    # insert the bits of the new Pauli into the old Pauli
    pstr = setpauli(pstr, partial_pstr, qinds)

    coeff *= sign

    # always a length-1 tuple, which will be compiled away
    return ((pstr, coeff),)
end

### Pauli Noise
"""
    applytoall!(gate::PauliNoise, prop_cache::PauliPropagationCache, p; kwargs...)

Overload of `applytoall!` for `PauliNoise` gates with noise strength `p` and a propagating `PauliSum`.
"""
function PropagationBase.applytoall!(gate::PauliNoise, prop_cache::PauliPropagationCache, p; kwargs...)
    # unpack the main pauli sum, aux is not needed
    psum = mainsum(prop_cache)

    # check that the noise strength is in the correct range
    _check_noise_strength(PauliNoise, p)

    # loop over all Pauli strings and their coefficients in the Pauli sum
    for (pstr, coeff) in psum

        # the Pauli on the site that the noise acts on
        pauli = getpauli(pstr, gate.qind)

        # `isdamped` is defined in noisechannels.jl for each Pauli noise channel
        # I Paulis are never damped, but the others vary
        if !isdamped(gate, pauli)
            continue
        end

        new_coeff = coeff * (1 - p)
        # change the coefficient in psum, don't move anything to aux_psum
        set!(psum, pstr, new_coeff)
    end

    return prop_cache
end

### Amplitude Damping Noise
"""
    applytoall!(gate::AmplitudeDampingNoise, prop_cache::PauliPropagationCache, gamma; kwargs...)

Overload of `applytoall!` for `AmplitudeDampingNoise` gates and a propagating `PauliSum`.
"""
function PropagationBase.applytoall!(gate::AmplitudeDampingNoise, prop_cache::PauliPropagationCache, gamma; kwargs...)
    # unpack the pauli sums
    psum = mainsum(prop_cache)
    aux_psum = auxsum(prop_cache)

    # check that the noise strength is in the correct range
    _check_noise_strength(AmplitudeDampingNoise, gamma)

    # loop over all Pauli strings and their coefficients in the Pauli sum
    for (pstr, coeff) in psum
        pauli = getpauli(pstr, gate.qind)
        if pauli == 0
            # Pauli is I, so the gate does not do anything
            continue

        elseif pauli == 1 || pauli == 2
            # Pauli is X or Y, so the gate will give a sqrt(1-gamma) prefactor
            new_coeff = sqrt(1 - gamma) * coeff
            # set the coefficient of the Pauli string in the psum to the new coefficient
            set!(psum, pstr, new_coeff)

        else
            # Pauli is Z, so the gate will split the Pauli string 

            # else we know the gate will split th Pauli string into two
            new_pstr = setpauli(pstr, 0, gate.qind)
            coeff1 = (1 - gamma) * coeff
            coeff2 = gamma * coeff

            # set the coefficient of the original Pauli string
            set!(psum, pstr, coeff1)

            # add the coefficient of the new Pauli string in the aux_psum
            add!(aux_psum, new_pstr, coeff2)

        end
    end

    return prop_cache
end

## T Gate
"""
    applytoall!(gate::TGate, prop_cache::AbstractPauliPropagationCache; kwargs...)

Overload of `applytoall!()` for `TGate(qind)` and a propagating `PauliSum`.
Redirects to a `PauliRotation(:Z, qind)` with angle π/4.
"""
function PropagationBase.applytoall!(gate::TGate, prop_cache::AbstractPauliPropagationCache; kwargs...)
    return applytoall!(PauliRotation(:Z, gate.qind), prop_cache, π / 4; kwargs...)
end

## TransferMapGate
"""
    apply(gate::TransferMapGate, pstr, coeff)

Apply a `TransferMapGate` to an integer Pauli string and its coefficient.
The outcomes are determined by the `transfer_map` of the gate.
"""
function PropagationBase.apply(gate::TransferMapGate, pstr, coeff; kwargs...)
    # the Paulis packed into the integer are used to index into the transfer map
    pauli_int = getpauli(pstr, gate.qinds)
    pstrs_and_factors = gate.transfer_map[pauli_int+1]
    # the new pstrs are the new Paulis that need to be set and the coefficients need to be multiplied with the factors
    return Tuple((setpauli(pstr, new_pstr, gate.qinds), coeff * factor) for (new_pstr, factor) in pstrs_and_factors)
end

### Frozen Gates
"""
    applytoall!(gate::FrozenGate, thetas, psum, aux_psum; kwargs...)

Overload of `applytoall!` for `FrozenGate`s. Re-directs to `applytoall!` for the wrapped `FrozenGate.gate` with the frozen parameter.
"""
function PropagationBase.applytoall!(gate::FrozenGate, prop_cache::AbstractPauliPropagationCache; kwargs...)
    return applytoall!(gate.gate, prop_cache, gate.parameter; kwargs...)
end
