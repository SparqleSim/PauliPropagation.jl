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


# VectorPauliSum overloads for the rotation gates and PauliNoise
include("./fused_vector.jl")


# MultiPauliSum overloads that run the fused vector application inside every zone
include("./fused_multi.jl")


"""
    propagate(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)

Like `PauliPropagation.propagate`, but defaults `fused=true` to use this module's fused
`applymergetruncate!` overloads. Pass `fused=false` for byte-identical default results.
"""
function propagate(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)
    return PauliPropagation.propagate(circuit, thing, thetas; fused, kwargs...)
end

"""
    propagate!(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)

In-place counterpart of `propagate`. See `propagate` for details.
"""
function propagate!(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)
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
