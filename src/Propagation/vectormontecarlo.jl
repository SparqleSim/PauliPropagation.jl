###
##
# mcapplytoall! specializations for the VectorPauliSum type, the Monte Carlo analogue of
# applytoall! in vectorspecializations.jl: instead of branching a term into two, randomly
# keep one branch, reweighted so that the result stays unbiased in expectation.
##
###

"""
    mcapplytoall!(gate, psum::VectorPauliSum, [param]; squared=false, thread=true, kwargs...)

1st-level function below `mcsample!` that stochastically applies one `gate` to every term in `psum`,
in place. This is the Monte Carlo analogue of `applytoall!`: instead of branching a term into two,
it randomly keeps one branch, reweighted to remain unbiased. Must be overloaded for each custom gate type.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function PropagationBase.mcapplytoall!(gate::CliffordGate, psum::VectorPauliSum; squared=false, thread::Bool=true, kwargs...)
    # Clifford gates act deterministically even in Monte Carlo mode.
    lookup_map = clifford_map[gate.symbol]

    power = squared ? 2 : 1

    term_vec = paulis(psum)
    coeff_vec = coefficients(psum)
    AK.foreachindex(term_vec; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do ii
        term = term_vec[ii]
        coeff = coeff_vec[ii]

        new_term, sign = only(apply(gate, term, one(coeff), lookup_map))

        new_coeff = coeff * sign^power

        term_vec[ii] = new_term
        coeff_vec[ii] = new_coeff
    end

    return psum
end


function PropagationBase.mcapplytoall!(gate::PauliRotation, psum::VectorPauliSum, theta; squared=false, thread::Bool=true, kwargs...)
    power = squared ? 2 : 1

    sin_val = sin(theta)
    abs_sin = abs(sin_val)^power
    sin_sign = sign(sin_val)^power

    cos_val = cos(theta)
    abs_cos = abs(cos_val)^power
    cos_sign = sign(cos_val)^power

    # >= 1 for power=1 and = 1 for power=2
    normalization = abs_sin + abs_cos

    # probability of branching off
    p = abs_sin / normalization

    gate_mask = symboltoint(paulitype(psum), gate.symbols, gate.qinds)

    term_vec = paulis(psum)
    coeff_vec = coefficients(psum)
    AK.foreachindex(term_vec; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do ii
        term = term_vec[ii]
        coeff = coeff_vec[ii]

        if !commutes(gate_mask, term)
            if rand() < p
                # Apply sine branch
                new_term, prod_sign = paulirotationproduct(gate_mask, term)
                new_coeff = coeff * normalization * sin_sign * prod_sign^power
                # Update in place
                term_vec[ii] = new_term
                coeff_vec[ii] = new_coeff
            else
                # Apply cos branch
                new_coeff = coeff * normalization * cos_sign
                # Update in place
                coeff_vec[ii] = new_coeff
            end
        end
    end

    return psum
end


function PropagationBase.mcapplytoall!(gate::FrozenGate, psum::VectorPauliSum; kwargs...)
    return PropagationBase.mcapplytoall!(gate.gate, psum, gate.parameter; kwargs...)
end
