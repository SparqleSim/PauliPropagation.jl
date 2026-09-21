###
##
# The gates of the library, written once for every propagation cache. A gate that branches by a fixed
# Pauli string is a rule for `xorbranch!`, one that only rescales coefficients a function for
# `mapcoeffsbypair!`, and one with a single output per input a transform for `map!`.
# The storage decides how each of them runs.
##
###

### Pauli rotations

"""
    applytoall!(gate::PauliRotation, prop_cache::AbstractPauliPropagationCache, theta; thread=true, kwargs...)

Overload of `applytoall!` for `PauliRotation` gates.
A Pauli string that anticommutes with the generator keeps a factor of cos(θ) and branches into its product with the generator,
which gets a factor of sin(θ).
"""
function PropagationBase.applytoall!(gate::PauliRotation, prop_cache::AbstractPauliPropagationCache, theta; thread::Bool=true, kwargs...)
    _check_qind_range(nqubits(prop_cache), gate.qinds)

    gate_mask = _branchmask(gate, prop_cache)
    cos_val = cos(theta)
    sin_val = sin(theta)

    function rotate(pstr, coeff)
        if commutes(gate_mask, pstr)
            return Unchanged()
        else
            _, sign = paulirotationproduct(gate_mask, pstr)
            return Branch(coeff * cos_val, coeff * sin_val * sign)
        end
    end

    return xorbranch!(rotate, prop_cache, gate_mask; thread)
end

"""
    applymergetruncate!(gate::PauliRotation, prop_cache::AbstractPauliPropagationCache, theta; thread=true, kwargs...)

Overload of `applymergetruncate!` for `PauliRotation` gates.
Applies the gate, merges the Pauli strings it branched into through `xormerge!`, and truncates.
"""
function PropagationBase.applymergetruncate!(gate::PauliRotation, prop_cache::AbstractPauliPropagationCache, theta; kwargs...)
    return _applyxormergetruncate!(gate, prop_cache, theta; kwargs...)
end

function paulirotationproduct(gate::PauliRotation, pstr::TT) where TT
    gate_mask = symboltoint(TT, gate.symbols, gate.qinds)
    return paulirotationproduct(gate_mask, pstr)
end

function paulirotationproduct(gate_mask::TT, pstr::TT) where TT
    new_pstr = _bitpaulimultiply(gate_mask, pstr)

    # this counts the exponent of the imaginary unit in the new Pauli string
    im_count = _calculatesignexponent(gate_mask, pstr)

    # now, instead of computing im^im_count followed by another im factor from the gate rules,
    # we do this in one step via a cheeky trick:
    sign = (im_count & 2) - 1
    # this is equivalent to sign = real( im * im^im_count)

    return new_pstr, sign
end

### Imaginary Pauli rotations

"""
    applymergetruncate!(gate::ImaginaryPauliRotation, prop_cache::AbstractPauliPropagationCache, tau; normalize_coeffs=true, kwargs...)

Overload of `applymergetruncate!` for `ImaginaryPauliRotation` gates and a propagating `PauliSum`.
Applies the gate, merges the resulting Pauli sum, and truncates it.
If `normalize_coeffs=true`, the resulting Pauli sum is normalized by the coefficient of the identity Pauli string after merging.
This is useful for numerical stability when evolving states in the Schrödinger picture.
"""
function PropagationBase.applymergetruncate!(gate::ImaginaryPauliRotation, prop_cache::AbstractPauliPropagationCache, tau; normalize_coeffs=true, thread::Bool=true, kwargs...)
    applytoall!(gate, prop_cache, tau; thread)
    xormerge!(prop_cache, _branchmask(gate, prop_cache); thread)

    # This gate assumes we are working in the Schrödinger picture evolving states
    # we normalize by the coefficient of the identity Pauli string
    # this is beneficial for numerical stability and if absolute coefficient truncation is used
    # example failure modes are if the coefficient is zero, of if it is supposed to be a number other than 1
    # these can be avoided by setting `normalize_coeffs=false`
    if normalize_coeffs
        # getcoeff is fast here even for VectorPauliSum
        # because we just merged and can do sorted search.
        mult!(prop_cache, 1 / getcoeff(activesum(prop_cache), 0))
    end

    truncate!(prop_cache; thread, kwargs...)
    return
end

"""
    applytoall!(gate::ImaginaryPauliRotation, prop_cache::AbstractPauliPropagationCache, tau; thread=true, kwargs...)

Like the `PauliRotation` method, except that an imaginary Pauli rotation branches the Pauli strings that commute with its generator,
with factors of cosh(τ) and sinh(τ).
"""
function PropagationBase.applytoall!(gate::ImaginaryPauliRotation, prop_cache::AbstractPauliPropagationCache, tau; thread::Bool=true, kwargs...)
    _check_qind_range(nqubits(prop_cache), gate.qinds)

    gate_mask = _branchmask(gate, prop_cache)
    cosh_val = cosh(tau)
    sinh_val = sinh(tau)

    # the sign of paulirotationproduct is also the minus sign in
    # e^{-τ/2 P} Q e^{-τ/2 P} = cosh(τ) Q - sinh(τ) PQ for commuting P and Q
    function rotate(pstr, coeff)
        if commutes(gate_mask, pstr)
            _, sign = paulirotationproduct(gate_mask, pstr)
            return Branch(coeff * cosh_val, coeff * sinh_val * sign)
        else
            return Unchanged()
        end
    end

    return xorbranch!(rotate, prop_cache, gate_mask; thread)
end

### Clifford gates

"""
    applytoall!(gate::CliffordGate, prop_cache::AbstractPauliPropagationCache; thread=true, kwargs...)

Apply a Clifford gate in place to a propagation cache. Clifford gates have exactly one output per
input, so the pair transformation is handled by `map!`.
"""
function PropagationBase.applytoall!(gate::CliffordGate, prop_cache::AbstractPauliPropagationCache; thread::Bool=true, kwargs...)
    _check_qind_range(nqubits(prop_cache), gate.qinds)

    lookup_map = clifford_map[gate.symbol]
    transform(term, coefficient) = only(apply(gate, term, coefficient, lookup_map))

    return map!(transform, prop_cache; thread)
end

# a Clifford gate maps distinct Pauli strings to distinct Pauli strings
PropagationBase.requiresmerging(::CliffordGate, ::AbstractPauliPropagationCache) = false

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

### Pauli noise

"""
    applytoall!(gate::PauliNoise, prop_cache::AbstractPauliPropagationCache, lambda; thread=true, kwargs...)

Overload of `applytoall!` for `PauliNoise` gates with noise strength `lambda`.
The Pauli strings that `isdamped` selects are damped by a factor of `1 - lambda`.
"""
function PropagationBase.applytoall!(gate::PauliNoise, prop_cache::AbstractPauliPropagationCache, lambda; thread::Bool=true, kwargs...)
    _check_qind_range(nqubits(prop_cache), gate.qind)
    _check_noise_strength(PauliNoise, lambda)

    qind = gate.qind
    damp_val = 1 - lambda
    damp(pstr, coeff) = isdamped(gate, getpauli(pstr, qind)) ? coeff * damp_val : coeff
    return mapcoeffsbypair!(damp, prop_cache; thread)
end

"""
    applymergetruncate!(gate::PauliNoise, prop_cache::AbstractPauliPropagationCache, lambda; kwargs...)

Apply `PauliNoise` and truncate in one walk whenever the truncation threshold does not depend on
the post-gate maximum coefficient. The gate never creates duplicate terms, so no merge is needed.
"""
function PropagationBase.applymergetruncate!(gate::PauliNoise, prop_cache::AbstractPauliPropagationCache, lambda;
    thread::Bool=true, min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf,
    max_sins::Real=Inf, min_rel_coeff=nothing, customtruncfunc=nothing, kwargs...)

    _check_qind_range(nqubits(prop_cache), gate.qind)
    _check_noise_strength(PauliNoise, lambda)

    # A relative threshold needs the maximum after damping, so it necessarily remains two passes.
    if !isnothing(min_rel_coeff)
        applytoall!(gate, prop_cache, lambda; thread)
        return truncate!(prop_cache;
            thread, min_abs_coeff, max_weight, max_freq, max_sins, min_rel_coeff, customtruncfunc)
    end

    qind = gate.qind
    damp_val = 1 - lambda
    damp(pstr, coeff) = isdamped(gate, getpauli(pstr, qind)) ? coeff * damp_val : coeff
    truncfunc = buildtruncfunc(prop_cache;
        min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread)
    return mapandtruncate!(damp, truncfunc, prop_cache; thread)
end

PropagationBase.requiresmerging(::PauliNoise, ::AbstractPauliPropagationCache) = false

### Amplitude damping noise

"""
    applytoall!(gate::AmplitudeDampingNoise, prop_cache::AbstractPauliPropagationCache, gamma; thread=true, kwargs...)

Overload of `applytoall!` for `AmplitudeDampingNoise` gates with noise strength `gamma`.
On the damped qubit, X and Y are damped by a factor of sqrt(1 - gamma),
and Z keeps a factor of 1 - gamma and branches into the identity with a factor of gamma.
"""
function PropagationBase.applytoall!(gate::AmplitudeDampingNoise, prop_cache::AbstractPauliPropagationCache, gamma; thread::Bool=true, kwargs...)
    _check_qind_range(nqubits(prop_cache), gate.qind)
    _check_noise_strength(AmplitudeDampingNoise, gamma)

    qind = gate.qind
    damp_val = sqrt(1 - gamma)

    function damp(pstr, coeff)
        pauli = getpauli(pstr, qind)
        if pauli == 0
            return Unchanged()
        elseif pauli == 3
            return Branch((1 - gamma) * coeff, gamma * coeff)
        else
            return Kept(damp_val * coeff)
        end
    end

    return xorbranch!(damp, prop_cache, _branchmask(gate, prop_cache); thread)
end

"""
    applymergetruncate!(gate::AmplitudeDampingNoise, prop_cache::AbstractPauliPropagationCache, gamma; thread=true, kwargs...)

Overload of `applymergetruncate!` for `AmplitudeDampingNoise` gates.
Applies the gate, merges the Pauli strings it branched into through `xormerge!`, and truncates.
"""
function PropagationBase.applymergetruncate!(gate::AmplitudeDampingNoise, prop_cache::AbstractPauliPropagationCache, gamma; kwargs...)
    return _applyxormergetruncate!(gate, prop_cache, gamma; kwargs...)
end

### T gate

"""
    applymergetruncate!(gate::TGate, prop_cache::AbstractPauliPropagationCache; kwargs...)

Apply a `TGate(qind)` through the top-level implementation of a `PauliRotation(:Z, qind)` with angle π/4.
"""
function PropagationBase.applymergetruncate!(gate::TGate, prop_cache::AbstractPauliPropagationCache; kwargs...)
    return applymergetruncate!(PauliRotation(:Z, gate.qind), prop_cache, π / 4; kwargs...)
end

"""
    applytoall!(gate::TGate, prop_cache::AbstractPauliPropagationCache; kwargs...)

Apply a `TGate(qind)` as a `PauliRotation(:Z, qind)` with angle π/4.
"""
function PropagationBase.applytoall!(gate::TGate, prop_cache::AbstractPauliPropagationCache; kwargs...)
    return applytoall!(PauliRotation(:Z, gate.qind), prop_cache, π / 4; kwargs...)
end

### Transfer map gates

"""
    apply(gate::TransferMapGate, pstr, coeff)

Apply a `TransferMapGate` to an integer Pauli string and its coefficient.
The outcomes are determined by the `transfer_map` of the gate.
"""
function PropagationBase.apply(gate::TransferMapGate, pstr, coeff; kwargs...)
    # the Paulis packed into the integer are used to index into the transfer map
    pauli_int = _transfermapindex(gate, pstr)
    pstrs_and_factors = gate.shifted_transfer_map[pauli_int]
    # the new pstrs are the new Paulis that need to be set and the coefficients need to be multiplied with the factors
    return ((_applytransfermap(pstr, shifted_pstr, gate.qind_mask), coeff * factor) for (shifted_pstr, factor) in pstrs_and_factors)
end

@inline _transfermapindex(gate::TransferMapGate{TM,STM,TMask,true}, pstr) where {TM,STM,TMask} = getpauli(pstr, gate.qind_start, gate.qind_stop)
@inline _transfermapindex(gate::TransferMapGate{TM,STM,TMask,false}, pstr) where {TM,STM,TMask} = getpauli(pstr, gate.qinds)

@inline function _applytransfermap(pstr::TT, shifted_pstr, qind_mask) where {TT<:PauliStringType}
    mask = TT(qind_mask)
    return (pstr & ~mask) | TT(shifted_pstr)
end

### Frozen gates

"""
    applymergetruncate!(gate::FrozenGate, prop_cache::AbstractPauliPropagationCache; kwargs...)

Apply a `FrozenGate` through the top-level implementation of its wrapped gate, with its frozen parameter.
"""
function PropagationBase.applymergetruncate!(gate::FrozenGate, prop_cache::AbstractPauliPropagationCache; kwargs...)
    return applymergetruncate!(gate.gate, prop_cache, gate.parameter; kwargs...)
end

"""
    applytoall!(gate::FrozenGate, prop_cache::AbstractPauliPropagationCache; kwargs...)

Apply a `FrozenGate` through the `applytoall!` implementation of its wrapped gate, with its frozen parameter.
"""
function PropagationBase.applytoall!(gate::FrozenGate, prop_cache::AbstractPauliPropagationCache; kwargs...)
    return applytoall!(gate.gate, prop_cache, gate.parameter; kwargs...)
end

### Gates that branch by a fixed Pauli string

# the Pauli string a gate branches by, as the mask of its `xorbranch!`
_branchmask(gate::Union{PauliRotation,ImaginaryPauliRotation}, prop_cache) = symboltoint(paulitype(prop_cache), gate.symbols, gate.qinds)

# Z ⊻ Z is the identity on the damped qubit
_branchmask(gate::AmplitudeDampingNoise, prop_cache) = symboltoint(paulitype(prop_cache), :Z, gate.qind)

# `applymergetruncate!` for a gate whose `applytoall!` is an `xorbranch!` by `_branchmask`.
# `xormergeandtruncate!` combines the merge and all per-term truncation criteria in one pass.
function _applyxormergetruncate!(gate, prop_cache::AbstractPauliPropagationCache, args...;
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf,
    min_rel_coeff=nothing, customtruncfunc=nothing, thread::Bool=true, kwargs...)

    applytoall!(gate, prop_cache, args...; thread)
    mask = _branchmask(gate, prop_cache)

    # A relative threshold is based on the largest merged coefficient and thus cannot be
    # evaluated while the merged output is written.
    if !isnothing(min_rel_coeff)
        xormerge!(prop_cache, mask; thread)
        truncate!(prop_cache;
            min_abs_coeff, max_weight, max_freq, max_sins, min_rel_coeff, customtruncfunc,
            thread, kwargs...)
        return
    end

    truncfunc = buildtruncfunc(prop_cache;
        min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread
    )

    xormergeandtruncate!(truncfunc, prop_cache, mask; thread)
    return
end
