###
##
# Pauli propagation over a term vector that carries a transposed bit index and a term table.
#
# The design follows propaq (arXiv:2609.07730), which credits it to monoprop: select the branching
# terms from a transposed view of the sum instead of testing every Pauli string, and place the
# products through a hash table instead of sorting and merging the whole sum after every gate. Both
# passes then cost the branching fraction of the sum -- between a twentieth and a sixth of it on the
# circuits in bench/ -- rather than all of it.
##
###

module IndexedPropagation

using PauliPropagation
using PauliPropagation.PropagationBase

export IndexedPauliPropagationCache, MultiIndexedPauliPropagationCache, nliveterms, compact!

include("symplectic.jl")
include("transposedindex.jl")
include("termtable.jl")
include("propagationcache.jl")
include("gates.jl")
include("multizone.jl")

"""
    propagate(circuit, psum, thetas=nothing; kwargs...)

Like `PauliPropagation.propagate`, but propagating through an `IndexedPauliPropagationCache`, or a
`MultiIndexedPauliPropagationCache` over `n_zones` zones when that is given. Returns a
`VectorPauliSum`.
"""
function propagate(circuit, psum, thetas=nothing; n_zones::Union{Nothing,Integer}=nothing, kwargs...)
    cache = n_zones === nothing ? IndexedPauliPropagationCache(psum) : MultiIndexedPauliPropagationCache(psum, n_zones)
    PauliPropagation.propagate!(circuit, cache, thetas; kwargs...)
    return VectorPauliSum(cache)
end

end
