###
##
# Variants of `applymergetruncate!` for MultiPauliSum that run the fused vector application inside
# every zone: a zone writes what it branches straight into the box its owner takes delivery of, and
# the truncations that read the coefficient are paid in the merge that follows.
#
# Fusing needs the zones to be array-backed, and the branching gates need the zone assignment to be
# linear, which is to say a power-of-two number of zones. Anything else falls through to the default.
##
###

# a single box per zone only exists where a gate that branches by a fixed mask permutes the zones
_fusedzones(prop_cache::PauliPropagation.MultiPauliPropagationCache) =
    _fusedzonestorage(prop_cache) && isxorlinear(prop_cache)

_fusedzonestorage(prop_cache::PauliPropagation.MultiPauliPropagationCache) =
    zonestorage(prop_cache) isa PropagationBase.ArrayStorage


### The branching rotation gates

"""
    applymergetruncate!(gate::PauliRotation, prop_cache::MultiPauliPropagationCache, theta; fused::Bool=false, kwargs...)

Fused overload of `applymergetruncate!` for `PauliRotation` over work zones -- see file header. Only
used when `fused=true` and the zones can take it; otherwise falls through (via `invoke`) to default
behavior.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.PauliRotation, prop_cache::PauliPropagation.MultiPauliPropagationCache, theta;
    fused::Bool=false,
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing,
    thread::Bool=true, kwargs...)

    if !fused || !_fusedzones(prop_cache)
        return invoke(PauliPropagation.applymergetruncate!,
            Tuple{PauliPropagation.PauliRotation,PauliPropagation.AbstractPauliPropagationCache,typeof(theta)},
            gate, prop_cache, theta;
            min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
    end

    return _fusedzonerotation!(gate, prop_cache, cos(theta), sin(theta), Val(:PauliRotation);
        min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
end

"""
    applymergetruncate!(gate::ImaginaryPauliRotation, prop_cache::MultiPauliPropagationCache, tau; fused::Bool=false, normalize_coeffs=true, kwargs...)

Fused overload of `applymergetruncate!` for `ImaginaryPauliRotation` over work zones. Shares its
branch-and-write core with the fused `PauliRotation` overload.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.ImaginaryPauliRotation, prop_cache::PauliPropagation.MultiPauliPropagationCache, tau;
    fused::Bool=false, normalize_coeffs::Bool=true,
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing,
    thread::Bool=true, kwargs...)

    if !fused || !_fusedzones(prop_cache)
        return invoke(PauliPropagation.applymergetruncate!,
            Tuple{PauliPropagation.ImaginaryPauliRotation,PauliPropagation.AbstractPauliPropagationCache,typeof(tau)},
            gate, prop_cache, tau;
            normalize_coeffs, min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
    end

    PauliPropagation._check_qind_range(PauliPropagation.nqubits(prop_cache), gate.qinds)

    # an empty sum has no identity coefficient to normalize by, so return ahead of that too
    if length(prop_cache) == 0
        return prop_cache
    end

    _fusedzonerotation!(gate, prop_cache, cosh(tau), sinh(tau), Val(:ImaginaryPauliRotation);
        min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)

    # This gate assumes we are working in the Schrödinger picture evolving states
    # we normalize by the coefficient of the identity Pauli string for numerical stability
    # the identity sits in one zone alone, and every zone is then scaled by what it says
    if normalize_coeffs
        identity_pstr = PauliPropagation.identitypauli(PauliPropagation.paulitype(prop_cache))
        scale = 1 / PauliPropagation.getcoeff(PauliPropagation.activesum(prop_cache), identity_pstr)

        for zonecache in zonecaches(prop_cache)
            PauliPropagation.activecoeffs(zonecache) .*= scale
        end
    end

    return prop_cache
end

# Shared core: every zone branches into its box, and the merge that takes delivery truncates.
function _fusedzonerotation!(gate, prop_cache, kept_val, new_val, gatetype::Val;
    min_abs_coeff::Real, max_weight::Real, max_freq::Real, max_sins::Real, customtruncfunc,
    thread::Bool, kwargs...)

    PauliPropagation._check_qind_range(PauliPropagation.nqubits(prop_cache), gate.qinds)

    if length(prop_cache) == 0
        return prop_cache
    end

    plain_mask = PauliPropagation.symboltoint(PauliPropagation.paulitype(prop_cache), gate.symbols, gate.qinds)

    # every zone holds the same kind of array, so which local path applies is decided once
    gate_mask = _gatemask(plain_mask, terms(first(zones(prop_cache))))

    truncfunc(pstr, coeff) = _coefftruncfunc(pstr, coeff; min_abs_coeff, max_freq, max_sins, customtruncfunc)

    applyxorbranchzones!(prop_cache, plain_mask; thread, truncfunc, kwargs...) do zonecache, box
        _fusedbranchzone!(zonecache, box, gate_mask, kept_val, new_val, max_weight, gatetype)
    end

    return prop_cache
end

# One zone, single-threaded: scale what branches and write the products into the box, in parent order
# so the owner can sort them by XOR passes.
function _fusedbranchzone!(zonecache, box, gate_mask, kept_val, new_val, max_weight, gatetype::Val)
    n_old = activesize(zonecache)
    n_old == 0 && return nothing

    box_terms, box_coeffs = terms(box), coefficients(box)

    # a term branches at most once, so the zone's own size bounds what the box has to hold
    if length(box_terms) < n_old
        resize!(box_terms, n_old)
        resize!(box_coeffs, n_old)
    end

    zone_terms, zone_coeffs, _, _ = PropagationBase._mainauxarrays(zonecache)
    n_new = _fusedbranchwrite!(box_terms, box_coeffs, 1, zone_terms, zone_coeffs, 1, n_old,
        gate_mask, kept_val, new_val, max_weight, gatetype, Val(true))

    resize!(box_terms, n_new)
    resize!(box_coeffs, n_new)

    return nothing
end


### Pauli Noise

"""
    applymergetruncate!(gate::PauliNoise, prop_cache::MultiPauliPropagationCache, lambda; fused::Bool=false, kwargs...)

Fused overload of `applymergetruncate!` for `PauliNoise` over work zones. The gate leaves every Pauli
string in the zone that owns it, so each zone runs the fused application of the sum it carries and
nothing travels.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.PauliNoise, prop_cache::PauliPropagation.MultiPauliPropagationCache, lambda;
    fused::Bool=false, thread::Bool=true, kwargs...)

    if !fused || !_fusedzonestorage(prop_cache)
        return invoke(PauliPropagation.applymergetruncate!,
            Tuple{PauliPropagation.PauliNoise,PauliPropagation.AbstractPauliPropagationCache,typeof(lambda)},
            gate, prop_cache, lambda; thread, kwargs...)
    end

    PropagationBase._eachzone(prop_cache, thread) do zone_id
        PauliPropagation.applymergetruncate!(gate, zonecaches(prop_cache)[zone_id], lambda;
            fused=true, thread=false, kwargs...)
    end

    return PropagationBase._syncsums!(prop_cache)
end
