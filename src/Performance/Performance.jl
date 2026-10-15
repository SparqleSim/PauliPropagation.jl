module Performance
# This is a module for performance-oriented optimizations that
# 1) may eventually become the default, and/or
# 2) inherently change the output (slightly) for performance benefits
#
# Opt in via `Performance.propagate` and `Performance.propagate!`.

using PauliPropagation
using PauliPropagation.PropagationBase
# exported from Base only since Julia 1.11
using Base: IdSet

using AcceleratedKernels
const AK = AcceleratedKernels


# Dispatch shared by the fused gate applications
include("./dispatch.jl")


# PauliSum overload for PauliRotation
include("./fused_dict.jl")


# VectorPauliSum and MultiPauliSum overloads for the rotation gates
include("./fused_rotations.jl")


# Layers of commuting gates where they are faster than their gates one by one
include("./layers.jl")


"""
    propagate(circuit, thing, thetas=nothing; fused::Bool=true, layers::Bool=fused, kwargs...)

Like `PauliPropagation.propagate`, but defaults `fused=true` to use this module's fused
`applymergetruncate!` overloads. Pass `fused=false` for byte-identical default results.
With `layers=true`, consecutive commuting gates are gathered into layers with `tolayers` where there are enough of them for a layer to be faster than its gates one by one on the type of `thing`.
Layers that are already in `circuit` are kept as they are.
"""
function propagate(circuit, thing, thetas=nothing; fused::Bool=true, layers::Bool=fused, kwargs...)
    if layers
        circuit, thetas = _tolayers(circuit, thetas, thing)
    end
    return PauliPropagation.propagate(circuit, thing, thetas; fused, kwargs...)
end

"""
    propagate!(circuit, thing, thetas=nothing; fused::Bool=true, layers::Bool=fused, kwargs...)

In-place counterpart of `propagate`. See `propagate` for details.
"""
function propagate!(circuit, thing, thetas=nothing; fused::Bool=true, layers::Bool=fused, kwargs...)
    if layers
        circuit, thetas = _tolayers(circuit, thetas, thing)
    end
    return PauliPropagation.propagate!(circuit, thing, thetas; fused, kwargs...)
end

"""
    mcpropagate(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)

Like `PauliPropagation.mcpropagate`, but defaults `fused=true` to use this module's fused
`applymergetruncate!` overloads. Pass `fused=false` for byte-identical default results.
"""
function mcpropagate(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)
    return PauliPropagation.mcpropagate(circuit, thing, thetas; fused, kwargs...)
end

"""
    mcpropagate!(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)

In-place counterpart of `mcpropagate`. See `mcpropagate` for details.
"""
function mcpropagate!(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)
    return PauliPropagation.mcpropagate!(circuit, thing, thetas; fused, kwargs...)
end

end
