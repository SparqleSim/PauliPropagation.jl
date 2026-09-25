###
##
# Variants of `applymergetruncate!` for the rotation gates on a VectorPauliSum, and on a MultiPauliSum,
# that branch and merge in one call by the library's own rule: the rule is asked only about the limbs
# the gate acts on, a product too heavy to ever be kept is never made, and the truncations are paid
# in the merge of the branch, which trusts the order the branch left instead of checking it.
##
###

const FusedCache = Union{PauliPropagation.VectorPauliPropagationCache,PauliPropagation.MultiPauliPropagationCache}

"""
    applymergetruncate!(gate::PauliRotation, prop_cache, theta; fused::Bool=false, kwargs...)

Fused overload of `applymergetruncate!` for `PauliRotation` -- see file header.
Only used when `fused=true`; otherwise falls through (via `invoke`) to default behavior.
"""
@inline function PauliPropagation.applymergetruncate!(gate::PauliPropagation.PauliRotation, prop_cache::FusedCache, theta;
    fused::Bool=false,
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing,
    thread::Bool=true, kwargs...)

    if !fused
        return _invokedefault(gate, prop_cache, theta;
            min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
    end

    _checkunusedkwargs(kwargs)
    return _fusedrotation!(gate, prop_cache, theta;
        min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread)
end

"""
    applymergetruncate!(gate::ImaginaryPauliRotation, prop_cache, tau; fused::Bool=false, normalize_coeffs=true, kwargs...)

Fused overload of `applymergetruncate!` for `ImaginaryPauliRotation` -- see file header.
Only used when `fused=true`; otherwise falls through (via `invoke`) to default behavior.
"""
@inline function PauliPropagation.applymergetruncate!(gate::PauliPropagation.ImaginaryPauliRotation, prop_cache::FusedCache, tau;
    fused::Bool=false, normalize_coeffs::Bool=true,
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing,
    thread::Bool=true, kwargs...)

    if !fused
        return _invokedefault(gate, prop_cache, tau;
            normalize_coeffs, min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
    end

    _checkunusedkwargs(kwargs)
    _fusedrotation!(gate, prop_cache, tau;
        min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread)

    # an empty sum has no identity coefficient to normalize by
    if normalize_coeffs && !isempty(prop_cache)
        mult!(prop_cache, 1 / getcoeff(activesum(prop_cache), 0); thread)
    end

    return prop_cache
end

# Both rotations branch by the library's rule of the gate, capped so that no new term above `max_weight` is made.
function _fusedrotation!(gate, prop_cache, param;
    min_abs_coeff::Real, max_weight::Real, max_freq::Real, max_sins::Real, customtruncfunc, thread::Bool)

    mask = PauliPropagation._branchmask(gate, prop_cache)
    rule = PauliPropagation._branchrule(gate, prop_cache, param)
    capped_rule = isinf(max_weight) ? rule : WeightCapped(rule, mask, max_weight)
    truncfunc = buildtruncfunc(prop_cache; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread)

    return xorbranchmergeandtruncate!(capped_rule, truncfunc, prop_cache, mask; thread)
end

"""
    WeightCapped(rule, mask, max_weight)

A rule of `xorbranch!` whose new terms above `max_weight` are not made.
They could never be kept, and this way they are never written or sorted.
They are weighed on the whole Pauli string, also when `rule` is asked about only some of its limbs.
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
    if branched isa Branch && PauliPropagation.truncateweight(pstr ⊻ capped.mask, capped.max_weight)
        return Kept(branched.kept)
    else
        return branched
    end
end
