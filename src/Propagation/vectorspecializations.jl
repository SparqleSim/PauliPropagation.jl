"""
    applytoall!(gate::PauliRotation, prop_cache::VectorPauliPropagationCache, theta; kwargs...)

Overload of `applytoall!` for `PauliRotation` gates and a propagating `VectorPauliSum`.
"""
function PropagationBase.applytoall!(gate::PauliRotation, prop_cache::VectorPauliPropagationCache, theta; kwargs...)

    # TODO: design this function in a way that it can be the default for branching gates. 
    # Think of U3 or amplitude damping 

    if prop_cache.active_size == 0
        return prop_cache
    end

    n_old = prop_cache.active_size

    # get the mask out because because the gate cannot be in the function when using GPU
    gate_mask = symboltoint(nqubits(prop_cache), gate.symbols, gate.qinds)

    # this needs to be in a separate function because variable names cannot be duplicated (WOW)
    # _flaganticommuting!(prop_cache, gate_mask)
    anticommutesfunc(trm) = !commutes(trm, gate_mask)
    flagterms!(anticommutesfunc, prop_cache)

    # this runs a cumsum over the flags to get the indices
    flagstoindices!(prop_cache)

    # the final index is the number of new terms
    n_noncommutes = lastactiveindex(prop_cache)

    # slit off into the same array
    n_new = n_old + n_noncommutes

    # potential resize factor
    resize_factor = 1.5
    if capacity(prop_cache) < n_new
        resize!(prop_cache, round(Int, n_new * resize_factor))
    end

    # does the branching logic
    _applypaulirotation!(prop_cache, gate_mask, theta)

    return prop_cache
end

function _applypaulirotation!(prop_cache::VectorPauliPropagationCache, gate_mask::TT, theta) where {TT}

    # pre-compute the sine and cosine values because the are used for every Pauli string that does not commute with the gate
    cos_val = cos(theta)
    sin_val = sin(theta)

    n = activesize(prop_cache)
    n_new = n + lastactiveindex(prop_cache)

    active_terms = activeterms(prop_cache)

    # full-length terms so we can write new terms at the end
    terms = paulis(mainsum(prop_cache))
    coeffs = coefficients(mainsum(prop_cache))
    @assert length(terms) >= n_new "VectorPauliPropagationCache terms array is not large enough to hold new terms."
    @assert length(coeffs) >= n_new "VectorPauliPropagationCache coeffs array is not large enough to hold new coeffs."

    flags = activeflags(prop_cache)
    indices = activeindices(prop_cache)

    # TODO: modularize this into something like "two-branching pattern"
    AK.foreachindex(active_terms) do ii
        # here it anticommutes
        if flags[ii]
            term = terms[ii]
            coeff = coeffs[ii]

            coeff1 = coeff * cos_val
            new_term, sign = paulirotationproduct(gate_mask, term)
            coeff2 = coeff * sin_val * sign

            coeffs[ii] = coeff1

            terms[n+indices[ii]] = new_term
            coeffs[n+indices[ii]] = coeff2
        end
    end

    # we now have n_new possibly douplicate Pauli strings in the array
    setactivesize!(prop_cache, n_new)

    return
end

### Imaginary Pauli Rotation
"""
    applytoall!(gate::ImaginaryPauliRotation, prop_cache::VectorPauliPropagationCache, tau; kwargs...)

Overload of `applytoall!` for `ImaginaryPauliRotation` gates and a propagating `VectorPauliSum`.
"""
function PropagationBase.applytoall!(gate::ImaginaryPauliRotation, prop_cache::VectorPauliPropagationCache, tau; kwargs...)

    # TODO: design this function in a way that it can be the default for branching gates. 
    # Think of U3 or amplitude damping 

    if prop_cache.active_size == 0
        return prop_cache
    end

    n_old = prop_cache.active_size

    # get the mask out because because the gate cannot be in the function when using GPU
    gate_mask::paulitype(prop_cache) = symboltoint(nqubits(prop_cache), gate.symbols, gate.qinds)

    # Imaginary Rotation branches on commutation, not anticommutation
    commutesfunc(trm) = commutes(trm, gate_mask)
    flagterms!(commutesfunc, prop_cache)

    # this runs a cumsum over the flags to get the indices
    flagstoindices!(prop_cache)

    # the final index is the number of new terms
    n_noncommutes = lastactiveindex(prop_cache)

    # slit off into the same array
    n_new = n_old + n_noncommutes

    # potential resize factor
    resize_factor = 1.5
    if capacity(prop_cache) < n_new
        resize!(prop_cache, round(Int, n_new * resize_factor))
    end

    # does the branching logic
    _applyimaginarypaulirotation!(prop_cache, gate_mask, tau)


    return prop_cache
end

function _applyimaginarypaulirotation!(prop_cache::VectorPauliPropagationCache, gate_mask::TT, tau) where {TT}

    # pre-compute the sine and cosine values because the are used for every Pauli string that does not commute with the gate
    cosh_val = cosh(tau)
    sinh_val = sinh(tau)

    n = activesize(prop_cache)
    n_new = n + lastactiveindex(prop_cache)

    active_terms = activeterms(prop_cache)

    # full-length terms so we can write new terms at the end
    terms = paulis(mainsum(prop_cache))
    coeffs = coefficients(mainsum(prop_cache))
    @assert length(terms) >= n_new "VectorPauliPropagationCache terms array is not large enough to hold new terms."
    @assert length(coeffs) >= n_new "VectorPauliPropagationCache coeffs array is not large enough to hold new coeffs."

    flags = activeflags(prop_cache)
    indices = activeindices(prop_cache)

    AK.foreachindex(active_terms) do ii
        # branching upon commutation
        if flags[ii]
            term = terms[ii]
            coeff = coeffs[ii]

            coeff1 = coeff * cosh_val
            new_term, sign = imaginarypaulirotationproduct(gate_mask, term)
            coeff2 = coeff * sinh_val * sign

            coeffs[ii] = coeff1

            terms[n+indices[ii]] = new_term
            coeffs[n+indices[ii]] = coeff2
        end
    end

    # we now have n_new possibly douplicate Pauli strings in the array
    setactivesize!(prop_cache, n_new)

    return
end

### Clifford gates
"""
    applytoall!(gate::CliffordGate, prop_cache::VectorPauliPropagationCache; kwargs...)
    
Overload of `applytoall!` for `CliffordGate`s and a propagating `VectorPauliSum`.
"""
function PropagationBase.applytoall!(gate::CliffordGate, prop_cache::VectorPauliPropagationCache; kwargs...)
    # TODO: This needs to be reworked for GPU support

    lookup_map = clifford_map[gate.symbol]

    # everything is done in place
    terms_view = activeterms(prop_cache)
    coeffs_view = activecoeffs(prop_cache)
    @assert length(terms_view) == length(coeffs_view)
    AK.foreachindex(terms_view) do ii
        term = terms_view[ii]
        coeff = coeffs_view[ii]

        # apply here returns a length-1 tuple
        term, coeff = only(apply(gate, term, coeff, lookup_map))

        # inbounds is safe here because we assert equal lengths
        terms_view[ii] = term
        coeffs_view[ii] = coeff
    end

    return prop_cache
end

requiresmerging(::CliffordGate) = false

##########################################

"""
    applytoall!(gate::PauliNoise, prop_cache::VectorPauliPropagationCache, p; kwargs...)

Overload of `applytoall!` for `PauliNoise` gates with noise strength `p` and a propagating `VectorPauliSum`.
"""
function PropagationBase.applytoall!(gate::PauliNoise, prop_cache::VectorPauliPropagationCache, p; kwargs...)

    # check that the noise strength is in the correct range
    _check_noise_strength(PauliNoise, p)

    # everything is done in place
    terms_view = activeterms(prop_cache)
    coeffs_view = activecoeffs(prop_cache)
    @assert length(terms_view) == length(coeffs_view)
    AK.foreachindex(terms_view) do ii
        pstr = terms_view[ii]
        coeff = coeffs_view[ii]

        pauli = getpauli(pstr, gate.qind)

        if isdamped(gate, pauli)
            coeffs_view[ii] = coeff * (1 - p)
        end
    end

    return prop_cache
end

requiresmerging(::PauliNoise) = false
