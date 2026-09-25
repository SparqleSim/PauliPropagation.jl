module Performance
# This is a module for performance-oriented optimizations that
# 1) may eventually become the default, and/or
# 2) inherently change the output (slightly) for performance benefits
#
# Opt in via `Performance.propagate` and `Performance.propagate!`.

using PauliPropagation
using PauliPropagation.PropagationBase

using AcceleratedKernels
const AK = AcceleratedKernels


# Truncation shared by the fused gate applications
include("./truncation.jl")


# Byte-local masks for one- and two-qubit gates on wide Pauli strings
include("./bytemasks.jl")


# PauliSum overload for PauliRotation
include("./fused_dict.jl")


# VectorPauliSum and MultiPauliSum overloads for the rotation gates
include("./fused_rotations.jl")


"""
    propagate(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)

Like `PauliPropagation.propagate`, but defaults `fused=true` to apply the rotation gates with this module's fused variants of `applymergetruncate!`.
Every other gate takes the library's `applymergetruncate!`, and `fused=false` gives byte-identical default results.
With `fused=true`, coefficients that track path properties and the `max_freq` and `max_sins` truncations are not supported.
"""
function propagate(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)
    if fused
        return _fusedpropagate!(_fusedapplymergetruncate!, circuit, deepcopy(thing), thetas; kwargs...)
    else
        return PauliPropagation.propagate(circuit, thing, thetas; kwargs...)
    end
end

"""
    propagate!(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)

In-place counterpart of `propagate`. See `propagate` for details.
"""
function propagate!(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)
    if fused
        return _fusedpropagate!(_fusedapplymergetruncate!, circuit, thing, thetas; kwargs...)
    else
        return PauliPropagation.propagate!(circuit, thing, thetas; kwargs...)
    end
end

"""
    mcpropagate(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)

Like `PauliPropagation.mcpropagate`, but defaults `fused=true` to apply the rotation gates with this module's fused variants of `applymergetruncate!`.
See `propagate` for details.
"""
function mcpropagate(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)
    if fused
        return _fusedmcpropagate(circuit, thing, thetas; kwargs...)
    else
        return PauliPropagation.mcpropagate(circuit, thing, thetas; kwargs...)
    end
end

"""
    mcpropagate!(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)

In-place counterpart of `mcpropagate`. See `mcpropagate` for details.
"""
function mcpropagate!(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)
    if fused
        return _fusedpropagate!(_fusedapplymergetruncateresample!, circuit, thing, thetas; kwargs...)
    else
        return PauliPropagation.mcpropagate!(circuit, thing, thetas; kwargs...)
    end
end

# refused in place, as by the library, since Monte Carlo propagation runs on a VectorPauliSum
mcpropagate!(circuit, thing::Union{PauliString,PauliSum}, thetas=nothing; kwargs...) =
    PauliPropagation.mcpropagate!(circuit, thing, thetas; kwargs...)

_fusedmcpropagate(circuit, pstr::PauliString, thetas; kwargs...) =
    _fusedpropagate!(_fusedapplymergetruncateresample!, circuit, VectorPauliSum(pstr), thetas; kwargs...)

_fusedmcpropagate(circuit, psum::PauliSum, thetas; kwargs...) =
    PauliSum(_fusedpropagate!(_fusedapplymergetruncateresample!, circuit, VectorPauliSum(psum), thetas; kwargs...))

_fusedmcpropagate(circuit, thing, thetas; kwargs...) =
    _fusedpropagate!(_fusedapplymergetruncateresample!, circuit, deepcopy(thing), thetas; kwargs...)

# `PauliPropagation.propagate!` with `step` applying each gate
_fusedpropagate!(step::F, circuit, pstr::PauliString, thetas; kwargs...) where {F} =
    _fusedpropagate!(step, circuit, PauliSum(pstr), thetas; kwargs...)

function _fusedpropagate!(step::F, circuit, term_sum::AbstractTermSum, thetas; kwargs...) where {F}
    prop_cache = _fusedpropagate!(step, circuit, PropagationCache(term_sum), thetas; kwargs...)
    return extractsum!(prop_cache, term_sum)
end

function _fusedpropagate!(step::F, circuit, prop_cache::AbstractPauliPropagationCache, thetas;
    heisenberg::Bool=true, max_freq::Real=Inf, max_sins::Real=Inf, kwargs...) where {F}

    # the fused rotations scale coefficients without recording the path they took
    if coefftype(prop_cache) <: PathProperties || !isinf(max_freq) || !isinf(max_sins)
        throw(ArgumentError(
            "The fused propagation of `Performance` does not support `PathProperties` coefficients or the `max_freq` and `max_sins` truncations. " *
            "Pass `fused=false` to propagate with the library's gates."))
    end

    circuit, thetas = PauliPropagation._preparecircuit(circuit, thetas, heisenberg)
    return PropagationBase._propagate!(step, circuit, prop_cache, thetas; kwargs...)
end

"""
    _fusedapplymergetruncate!(gate, prop_cache, args...; kwargs...)

Apply one gate in the fused propagation.
This is the fused variant of `applymergetruncate!` where this module defines one for the gate and cache, and the library's `applymergetruncate!` otherwise.
"""
_fusedapplymergetruncate!(gate, prop_cache, args...; kwargs...) =
    applymergetruncate!(gate, prop_cache, args...; kwargs...)

# a frozen gate and a T gate reach the fused variant of the rotation they apply
_fusedapplymergetruncate!(gate::FrozenGate, prop_cache; kwargs...) =
    _fusedapplymergetruncate!(gate.gate, prop_cache, gate.parameter; kwargs...)

_fusedapplymergetruncate!(gate::TGate, prop_cache; kwargs...) =
    _fusedapplymergetruncate!(PauliRotation(:Z, gate.qind), prop_cache, π / 4; kwargs...)

# `PauliPropagation.applymergetruncateresample!` with the fused step applying the gate
function _fusedapplymergetruncateresample!(gate, prop_cache, args...; max_size::Real,
    resampling_size::Integer=round(Int, max_size / 2), resample_func=nothing, thread::Bool=true, kwargs...)

    _fusedapplymergetruncate!(gate, prop_cache, args...; thread, kwargs...)

    if length(prop_cache) > max_size
        resample!(prop_cache, resampling_size; resample_func, thread, kwargs...)
    end

    return prop_cache
end

end
