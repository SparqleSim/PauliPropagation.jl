###
##
# Variants of `applymergetruncate!` for the rotation gates on a VectorPauliSum, and on a MultiPauliSum,
# that truncate during gate application: the coefficient truncations are paid in the merge of the
# branch, and a product too heavy to ever be kept is never made.
# This may yield slightly different results compared to normal functionality.
##
###

const FusedCache = Union{PauliPropagation.VectorPauliPropagationCache,PauliPropagation.MultiPauliPropagationCache}

"""
    applymergetruncate!(gate::PauliRotation, prop_cache, theta; fused::Bool=false, kwargs...)

Fused overload of `applymergetruncate!` for `PauliRotation` -- see file header.
Only used when `fused=true`; otherwise falls through (via `invoke`) to default behavior.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.PauliRotation, prop_cache::FusedCache, theta;
    fused::Bool=false,
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing,
    thread::Bool=true, kwargs...)

    if !fused
        return _invokedefault(gate, prop_cache, theta;
            min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
    end

    _checkunusedkwargs(kwargs)
    return _fusedrotation!(gate, prop_cache, cos(theta), sin(theta), false;
        min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread)
end

"""
    applymergetruncate!(gate::ImaginaryPauliRotation, prop_cache, tau; fused::Bool=false, normalize_coeffs=true, kwargs...)

Fused overload of `applymergetruncate!` for `ImaginaryPauliRotation` -- see file header.
Only used when `fused=true`; otherwise falls through (via `invoke`) to default behavior.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.ImaginaryPauliRotation, prop_cache::FusedCache, tau;
    fused::Bool=false, normalize_coeffs::Bool=true,
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing,
    thread::Bool=true, kwargs...)

    if !fused
        return _invokedefault(gate, prop_cache, tau;
            normalize_coeffs, min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
    end

    _checkunusedkwargs(kwargs)
    _fusedrotation!(gate, prop_cache, cosh(tau), sinh(tau), true;
        min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread)

    # an empty sum has no identity coefficient to normalize by
    if normalize_coeffs && !isempty(prop_cache)
        mult!(prop_cache, 1 / getcoeff(activesum(prop_cache), 0))
    end

    return prop_cache
end

# Both rotations branch by the gate's Pauli string: a `PauliRotation` the terms that anticommute
# with it, an `ImaginaryPauliRotation` the ones that commute.
function _fusedrotation!(gate, prop_cache, kept_val, new_val, on_commuting::Bool;
    min_abs_coeff::Real, max_weight::Real, max_freq::Real, max_sins::Real, customtruncfunc, thread::Bool)

    PauliPropagation._check_qind_range(nqubits(prop_cache), gate.qinds)
    if isempty(prop_cache)
        return prop_cache
    end

    mask = symboltoint(paulitype(prop_cache), gate.symbols, gate.qinds)
    rule = LocalRotationRule(_gatemask(mask, _localterms(prop_cache)), kept_val, new_val, on_commuting)
    capped_rule = isinf(max_weight) ? rule : WeightCapped(rule, mask, max_weight)

    truncfunc(pstr, coeff) = _coefftruncfunc(pstr, coeff; min_abs_coeff, max_freq, max_sins, customtruncfunc)

    xorbranch!(capped_rule, prop_cache, mask; thread)
    return xormergeandtruncate!(truncfunc, prop_cache, mask; thread)
end

# the terms whose array type decides which local read applies; every zone holds the same kind
_localterms(prop_cache::PauliPropagation.VectorPauliPropagationCache) = terms(mainsum(prop_cache))
_localterms(prop_cache::PauliPropagation.MultiPauliPropagationCache) = terms(first(zones(prop_cache)))


### The rules

"""
    LocalRotationRule(gate_mask, kept_val, new_val, on_commuting)

The rule of a rotation for `xorbranch!`, deciding from the bytes or words that `gate_mask` touches when it is a `ByteMask` or a `WordMask`,
and from the whole Pauli string otherwise.
A term that commutes with the gate branches when `on_commuting`, and one that anticommutes otherwise.
"""
struct LocalRotationRule{M,C}
    gate_mask::M
    kept_val::C
    new_val::C
    on_commuting::Bool
end

# the array kernels come with an index, and read through the bytes
@inline function PropagationBase.ruleat(rule::LocalRotationRule, terms, coefficients, ii::Int)
    bytes = _bytesof(terms, rule.gate_mask)
    if _gatecommutes(rule.gate_mask, terms, bytes, ii) == rule.on_commuting
        sign = _gatesign(rule.gate_mask, terms, bytes, ii)
        coeff = @inbounds coefficients[ii]
        return Branch(coeff * rule.kept_val, coeff * rule.new_val * sign)
    else
        return Unchanged()
    end
end

# every other storage comes with the term
@inline function (rule::LocalRotationRule)(pstr, coeff)
    mask = _plainmask(rule.gate_mask)
    if commutes(mask, pstr) == rule.on_commuting
        _, sign = PauliPropagation.paulirotationproduct(mask, pstr)
        return Branch(coeff * rule.kept_val, coeff * rule.new_val * sign)
    else
        return Unchanged()
    end
end

"""
    WeightCapped(rule, mask, max_weight)

A rule of `xorbranch!` whose new terms above `max_weight` are not made.
They could never be kept, and this way they are never written or sorted.
"""
struct WeightCapped{R,TT,W<:Real}
    rule::R
    mask::TT
    max_weight::W
end

@inline function (capped::WeightCapped)(pstr, coeff)
    branched = capped.rule(pstr, coeff)
    return _capweight(capped, branched, pstr)
end

@inline function PropagationBase.ruleat(capped::WeightCapped, terms, coefficients, ii::Int)
    branched = PropagationBase.ruleat(capped.rule, terms, coefficients, ii)
    if branched isa Branch
        return _capweight(capped, branched, (@inbounds terms[ii]))
    else
        return branched
    end
end

# a term whose new term is too heavy only keeps its own coefficient
@inline function _capweight(capped::WeightCapped, branched, pstr)
    if branched isa Branch && _truncateweight(pstr ⊻ capped.mask, capped.max_weight)
        return Kept(branched.kept)
    else
        return branched
    end
end
