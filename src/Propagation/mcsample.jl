###
##
# The Monte Carlo counterparts of the gates in specializations.jl, written once for every Pauli sum:
# instead of branching a term into two, a gate randomly keeps one branch, reweighted so that the
# result stays unbiased in expectation. Every term has a single output, so each gate is a transform
# for `map!`.
##
###

"""
    mcapplytoall!(gate, psum::AbstractPauliSum, [param]; squared=false, thread=true, kwargs...)

1st-level function below `mcsample!` that stochastically applies one `gate` to every term in `psum`,
in place. This is the Monte Carlo analogue of `applytoall!`: instead of branching a term into two,
it randomly keeps one branch, reweighted to remain unbiased. Must be overloaded for each custom gate type.
`thread=false` disables multithreading in every function on the `VectorPauliSum` backend that can multithread.
"""
function PropagationBase.mcapplytoall!(gate::CliffordGate, psum::AbstractPauliSum; squared::Bool=false, thread::Bool=true, kwargs...)
    _check_qind_range(nqubits(psum), gate.qinds)

    lookup_map = clifford_map[gate.symbol]

    # a Clifford gate is deterministic, and sampling by squared coefficients squares its sign away
    function permute(pstr, coeff)
        new_pstr, signed_coeff = only(apply(gate, pstr, coeff, lookup_map))
        return (new_pstr, squared ? coeff : signed_coeff)
    end

    return map!(permute, psum; thread)
end

function PropagationBase.mcapplytoall!(gate::PauliRotation, psum::AbstractPauliSum, theta; squared::Bool=false, thread::Bool=true, kwargs...)
    _check_qind_range(nqubits(psum), gate.qinds)

    gate_mask = symboltoint(paulitype(psum), gate.symbols, gate.qinds)
    power = squared ? 2 : 1
    sin_val = sin(theta)
    cos_val = cos(theta)

    # a branch is kept with probability proportional to |sin|^power or |cos|^power, and carries
    # the normalization so that the sum stays unbiased
    normalization = abs(sin_val)^power + abs(cos_val)^power
    p_sin = abs(sin_val)^power / normalization
    sin_weight = normalization * sign(sin_val)^power
    cos_weight = normalization * sign(cos_val)^power

    function sample(pstr, coeff)
        commutes(gate_mask, pstr) && return (pstr, coeff)
        rand() < p_sin || return (pstr, _cosbranch(coeff, cos_weight))

        new_pstr, prod_sign = paulirotationproduct(gate_mask, pstr)
        return (new_pstr, _sinbranch(coeff, sin_weight * prod_sign^power))
    end

    return map!(sample, psum; thread)
end

function PropagationBase.mcapplytoall!(gate::FrozenGate, psum::AbstractPauliSum; kwargs...)
    return PropagationBase.mcapplytoall!(gate.gate, psum, gate.parameter; kwargs...)
end

# the coefficient of the branch a term keeps; path properties also count the branch
_cosbranch(coeff, weight) = coeff * weight
_sinbranch(coeff, weight) = coeff * weight
