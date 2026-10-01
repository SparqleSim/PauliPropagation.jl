
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

function PropagationBase.PropagationCache(psum::PauliString)
    return PauliPropagationCache(PauliSum(psum))
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

    # the array kernels index all four arrays up to the active size without bounds checks, and
    # write at positions they accumulate in `indices`, which must not wrap
    function VectorPauliPropagationCache(psum::VPS, aux_psum::VPS, flags::VB, indices::VI, active_size::Int) where {VPS<:VectorPauliSum,VB,VI}
        n = length(psum)
        if !(length(aux_psum) == length(flags) == length(indices) == n)
            throw(ArgumentError("psum, aux_psum, flags and indices must have the same length."))
        end
        if eltype(indices) !== Int
            throw(ArgumentError("indices must hold Int, got $(eltype(indices))."))
        end
        if !(0 <= active_size <= n)
            throw(ArgumentError("active_size must be between 0 and the length of psum, got $active_size for $n."))
        end
        return new{VPS,VB,VI}(psum, aux_psum, flags, indices, active_size)
    end
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
    n_sorted = sortedprefix(mainsum(prop_cache))
    active_size = activesize(prop_cache)
    # we can only assume something went wront here. Reset to 0.
    n_sorted = n_sorted > active_size ? 0 : n_sorted
    return VectorPauliSum(nqubits(prop_cache), activeterms(prop_cache), activecoeffs(prop_cache), n_sorted)
end

# convert back
function VectorPauliSum(prop_cache::VectorPauliPropagationCache)
    vecpsum = deepcopy(mainsum(prop_cache))
    resize!(vecpsum, activesize(prop_cache))
    return vecpsum
end

function PauliSum(prop_cache::VectorPauliPropagationCache; thread::Bool=true)
    merge!(prop_cache; thread)
    return PauliSum(nqubits(prop_cache), Dict(zip(activeterms(prop_cache), activecoeffs(prop_cache))))
end

PropagationBase.activesize(prop_cache::VectorPauliPropagationCache) = prop_cache.active_size

function PropagationBase.setactivesize!(prop_cache::VectorPauliPropagationCache, new_size::Int)
    if !(0 <= new_size <= capacity(prop_cache))
        throw(ArgumentError("active size must be between 0 and the capacity, got $new_size for $(capacity(prop_cache))."))
    end
    prop_cache.active_size = new_size
    return prop_cache
end

PropagationBase.indices(prop_cache::VectorPauliPropagationCache) = prop_cache.indices
PropagationBase.flags(prop_cache::VectorPauliPropagationCache) = prop_cache.flags

# the sums grow their own arrays, and the cache its flags and indices
function Base.resize!(prop_cache::VectorPauliPropagationCache, n::Int)
    resize!(mainsum(prop_cache), n)
    resize!(auxsum(prop_cache), n)
    prop_cache.flags = PropagationBase._resizearray(prop_cache.flags, n)
    prop_cache.indices = PropagationBase._resizearray(prop_cache.indices, n)
    setactivesize!(prop_cache, min(activesize(prop_cache), n))
    return prop_cache
end


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
