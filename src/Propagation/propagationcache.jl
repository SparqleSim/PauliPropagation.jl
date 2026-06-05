
###
##
# The default propagation cache creates a tuple of a PauliSum and an auxillary PauliSum using the similar() function.
##
###

# TODO: More helpful utilities for this Pauli abstract type
abstract type AbstractPauliPropagationCache <: AbstractPropagationCache end

nqubits(prop_cache::AbstractPauliPropagationCache) = nqubits(mainsum(prop_cache))
PropagationBase.nsites(prop_cache::AbstractPauliPropagationCache) = nqubits(prop_cache)

paulis(prop_cache::AbstractPauliPropagationCache) = terms(prop_cache)
paulitype(prop_cache::AbstractPauliPropagationCache) = termtype(prop_cache)


# fallback for Pauli sums
function PropagationBase.PropagationCache(psum::AbstractPauliSum)
    return PauliPropagationCache(psum)
end


PropagationBase.mainsum(prop_cache::AbstractPauliPropagationCache) = prop_cache.psum
PropagationBase.auxsum(prop_cache::AbstractPauliPropagationCache) = prop_cache.aux_psum

# assumes the fields are called "psum" and "aux_psum"
function PropagationBase.setmainsum!(prop_cache::AbstractPauliPropagationCache, new_mainsum)
    prop_cache.psum = new_mainsum
    return prop_cache
end

function PropagationBase.setauxsum!(prop_cache::AbstractPauliPropagationCache, new_auxsum)
    prop_cache.aux_psum = new_auxsum
    return prop_cache
end


## The default propagation cache 
mutable struct PauliPropagationCache{TS} <: AbstractPauliPropagationCache
    psum::TS
    aux_psum::TS
end

function PauliPropagationCache(psum)
    aux_psum = similar(psum)
    return PauliPropagationCache(psum, aux_psum)
end

function PropagationBase.activesum(prop_cache::PauliPropagationCache)
    return mainsum(prop_cache)
end

###
##
# A VectorPropagationCache carries two VectorPauliSums and flags and indices for propagation.
# It is used inside propagate() to avoid reallocations.
##
###

mutable struct VectorPauliPropagationCache{VPS<:VectorPauliSum,VB,VI} <: AbstractPauliPropagationCache
    psum::VPS
    aux_psum::VPS
    # TODO: are flags ever needed? Can we use indices alone?
    flags::VB
    indices::VI

    # we will over-allocate the arrays and keep track of the non-empty size
    active_size::Int
end

# An overload for generality
function PropagationBase.PropagationCache(vecpsum::VectorPauliSum)
    return VectorPauliPropagationCache(vecpsum)
end

function VectorPauliPropagationCache(vecpsum::VectorPauliSum{VT,VC}) where {VT,VC}
    aux_vecpsum = similar(vecpsum)
    flags = similar(paulis(vecpsum), Bool)
    indices = similar(paulis(vecpsum), Int)
    return VectorPauliPropagationCache(vecpsum, aux_vecpsum, flags, indices, length(vecpsum))
end

VectorPauliPropagationCache(pstr::PauliString) = VectorPauliPropagationCache(VectorPauliSum(pstr))

function VectorPauliPropagationCache(vpsum::PauliSum)
    return VectorPauliPropagationCache(VectorPauliSum(vpsum))
end

function PropagationBase.activesum(prop_cache::VectorPauliPropagationCache)
    return VectorPauliSum(nqubits(prop_cache), activeterms(prop_cache), activecoeffs(prop_cache))
end

# convert back
function VectorPauliSum(prop_cache::VectorPauliPropagationCache)
    vecpsum = deepcopy(mainsum(prop_cache))
    resize!(vecpsum, activesize(prop_cache))
    return vecpsum
end

function PauliSum(prop_cache::VectorPauliPropagationCache)
    merge!(prop_cache)
    return PauliSum(nqubits(prop_cache), Dict(zip(activeterms(prop_cache), activecoeffs(prop_cache))))
end

PropagationBase.activesize(prop_cache::VectorPauliPropagationCache) = prop_cache.active_size
PropagationBase.setactivesize!(prop_cache::VectorPauliPropagationCache, new_size::Int) = (prop_cache.active_size = new_size; prop_cache)

PropagationBase.indices(prop_cache::VectorPauliPropagationCache) = prop_cache.indices
PropagationBase.flags(prop_cache::VectorPauliPropagationCache) = prop_cache.flags


function Base.show(io::IO, prop_cache::VectorPauliPropagationCache)
    println(io, "VectorPropagationCache with $(prop_cache.active_size) terms:")
    for i in 1:prop_cache.active_size
        if i > 20
            println(io, "  ...")
            break
        end
        pauli_string = inttostring(prop_cache.psum.terms[i], prop_cache.psum.nqubits)
        if length(pauli_string) > 20
            pauli_string = pauli_string[1:20] * "..."
        end
        println(io, prop_cache.psum.coeffs[i], " * $(pauli_string)")
    end
end


function Base.resize!(prop_cache::VectorPauliPropagationCache, n_new::Int)
    resize!(prop_cache.psum, n_new)
    resize!(prop_cache.aux_psum, n_new)
    resize!(prop_cache.flags, n_new)
    resize!(prop_cache.indices, n_new)
    return prop_cache
end
