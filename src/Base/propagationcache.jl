###
##
# Propagation caches carry the main term sum and auxiliary data structures for efficient propagation.
##
###

abstract type AbstractPropagationCache end

## The interface to hook into with each library
PropagationCache(thing::AbstractTermSum) = _thrownotimplemented(thing, :PropagationCache)

"""
    mainsum(prop_cache::AbstractPropagationCache)

Returns the current main term sum from the propagation cache.
It may be over-allocated and filled with trash terms.
Use `extractsum!(prop_cache)` to get a properly-sized and valid term sum.
"""
mainsum(prop_cache::AbstractPropagationCache) = _thrownotimplemented(prop_cache, :mainsum)

"""
    auxsum(prop_cache::AbstractPropagationCache)

Returns the current auxiliary term sum from the propagation cache.
It may be over-allocated and filled with trash terms.
"""
auxsum(prop_cache::AbstractPropagationCache) = _thrownotimplemented(prop_cache, :auxsum)

setmainsum!(prop_cache::AbstractPropagationCache, new_mainsum) = _thrownotimplemented(prop_cache, :setmainsum!)
setauxsum!(prop_cache::AbstractPropagationCache, new_auxsum) = _thrownotimplemented(prop_cache, :setauxsum!)

terms(prop_cache::AbstractPropagationCache) = _terms(StorageType(prop_cache), prop_cache)
_terms(::DictStorage, prop_cache::AbstractPropagationCache) = terms(mainsum(prop_cache))
_terms(::ArrayStorage, prop_cache::AbstractPropagationCache) = activeterms(prop_cache)

coefficients(prop_cache::AbstractPropagationCache) = _coefficients(StorageType(prop_cache), prop_cache)
_coefficients(::DictStorage, prop_cache::AbstractPropagationCache) = coefficients(mainsum(prop_cache))
_coefficients(::ArrayStorage, prop_cache::AbstractPropagationCache) = activecoeffs(prop_cache)

function termtype(prop_cache::AbstractPropagationCache)
    mainTT = termtype(mainsum(prop_cache))
    auxTT = termtype(auxsum(prop_cache))
    if mainTT != auxTT
        throw(ErrorException("Term types of mainsum(prop_cache) and auxsum(prop_cache) do not match."))
    end
    return mainTT
end

function coefftype(prop_cache::AbstractPropagationCache)
    mainCT = coefftype(mainsum(prop_cache))
    auxCT = coefftype(auxsum(prop_cache))
    if mainCT != auxCT
        throw(ErrorException("Coefficient types of mainsum(prop_cache) and auxsum(prop_cache) do not match."))
    end
    return mainCT
end

function numcoefftype(prop_cache::AbstractPropagationCache)
    mainNCT = numcoefftype(mainsum(prop_cache))
    auxNCT = numcoefftype(auxsum(prop_cache))
    if mainNCT != auxNCT
        throw(ErrorException("Numeric coefficient types of mainsum(prop_cache) and auxsum(prop_cache) do not match."))
    end
    return mainNCT
end


# An optional interface for returning the mainsum when it is safe to return it as a whole or as a view.
# Should leave the sum linked to the cache, and the cache unperturbed.
# This is concrete type dependent.
function activesum(prop_cache::AbstractPropagationCache)
    _thrownotimplemented(prop_cache, :activesum)
end

"""
    swapsums!(prop_cache::AbstractPropagationCache)

Swaps the mainsum and auxsum pointers on the propagation cache.
"""
function swapsums!(prop_cache::AbstractPropagationCache)
    temp = mainsum(prop_cache)
    setmainsum!(prop_cache, auxsum(prop_cache))
    setauxsum!(prop_cache, temp)
    return prop_cache
end

"""
    copyswapsums!(prop_cache::AbstractPropagationCache)
    
Copies the active contents of the mainsum into the auxsum and then calls swapsums!(prop_cache).
This is useful for when the current auxsum is meant to carry the final result.
"""
function copyswapsums!(prop_cache::AbstractPropagationCache)
    # instead of just swapping auxsum(prop_cache) and mainsum(prop_cache),
    # we need to copy the mainsum into the aux sum and then swap the sums
    copy!(auxsum(prop_cache), mainsum(prop_cache))
    swapsums!(prop_cache)
    return prop_cache
end

StorageType(prop_cache::AbstractPropagationCache) = StorageType(mainsum(prop_cache))

nsites(prop_cache::AbstractPropagationCache) = nsites(mainsum(prop_cache))

function Base.length(prop_cache::AbstractPropagationCache)
    return _length(StorageType(prop_cache), prop_cache)
end
function _length(::DictStorage, prop_cache::AbstractPropagationCache)
    return length(mainsum(prop_cache))
end

function _length(::ArrayStorage, prop_cache::AbstractPropagationCache)
    return activesize(prop_cache)
end

Base.isempty(prop_cache::AbstractPropagationCache) = length(prop_cache) == 0

function capacity(prop_cache::AbstractPropagationCache)
    return _capacity(StorageType(prop_cache), prop_cache)
end

function _capacity(::DictStorage, prop_cache::AbstractPropagationCache)
    return length(storage(mainsum(prop_cache)).slots)
end

function _capacity(::ArrayStorage, prop_cache::AbstractPropagationCache)
    return length(mainsum(prop_cache))
end

## Interface for vector-based propagation caches

activesize(prop_cache::AbstractPropagationCache) = _thrownotimplemented(prop_cache, :activesize)
setactivesize!(prop_cache::AbstractPropagationCache, new_size::Int) = _thrownotimplemented(prop_cache, :setactivesize!)


activeterms(prop_cache::AbstractPropagationCache) = view(terms(mainsum(prop_cache)), 1:activesize(prop_cache))
activecoeffs(prop_cache::AbstractPropagationCache) = view(coefficients(mainsum(prop_cache)), 1:activesize(prop_cache))
activeauxterms(prop_cache::AbstractPropagationCache) = view(terms(auxsum(prop_cache)), 1:activesize(prop_cache))
activeauxcoeffs(prop_cache::AbstractPropagationCache) = view(coefficients(auxsum(prop_cache)), 1:activesize(prop_cache))

flags(prop_cache::AbstractPropagationCache) = _thrownotimplemented(prop_cache, :flags)
indices(prop_cache::AbstractPropagationCache) = _thrownotimplemented(prop_cache, :indices)
activeflags(prop_cache::AbstractPropagationCache) = view(flags(prop_cache), 1:activesize(prop_cache))
activeindices(prop_cache::AbstractPropagationCache) = view(indices(prop_cache), 1:activesize(prop_cache))
lastactiveindex(prop_cache::AbstractPropagationCache) = activeindices(prop_cache)[end]


function mult!(prop_cache::AbstractPropagationCache, scalar::Number)
    mult!(mainsum(prop_cache), scalar)
    return prop_cache
end

function Base.resize!(prop_cache::AbstractPropagationCache, new_size::Int)
    _thrownotimplemented(prop_cache, :resize!)
end

## Back-conversions 

# effectively a out-of-place version of extractsum!()
function (::Type{TS})(prop_cache::AbstractPropagationCache) where TS<:AbstractTermSum
    return convert(TS, extractsum!(deepcopy(prop_cache)))
end

"""
    extractsum!(prop_cache::AbstractPropagationCache, which_sum::AbstractTermSum)

Extracts the indicated sum `which_sum` from the propagation cache. 
This resizes the cache to the active size and returns the indicated sum.
Further manipulating prop_cache or the returned sum may invalidate the other.
If the indicated sum is the auxsum, active contents will be copied over from mainsum.
If neither `mainsum(prop_cache)` nor `auxsum(prop_cache)` is indicated, an error is thrown. 
"""
function extractsum!(prop_cache::AbstractPropagationCache, which_sum::AbstractTermSum)
    if which_sum !== mainsum(prop_cache) && which_sum !== auxsum(prop_cache)
        throw(ArgumentError("Indicated sum is not part of the propagation cache."))
    end

    # if the indicated sum is not the mainsum, it is the auxsum
    # copy contents over and extract 
    if which_sum !== mainsum(prop_cache)
        copyswapsums!(prop_cache)
    end

    return extractsum!(prop_cache)
end

"""
    extractsum!(prop_cache::AbstractPropagationCache)

Extracts the mainsum from the propagation cache. 
This resizes the cache to the active size and returns the mainsum.
Further manipulating prop_cache or the returned sum may invalidate the other.
"""
function extractsum!(prop_cache::AbstractPropagationCache)
    return _extractsum!(StorageType(prop_cache), prop_cache)
end


_extractsum!(::DictStorage, prop_cache::AbstractPropagationCache) = mainsum(prop_cache)

function _extractsum!(::ArrayStorage, prop_cache::AbstractPropagationCache)
    # resize the entire cache to retain validity
    resize!(prop_cache, activesize(prop_cache))
    return mainsum(prop_cache)
end

function _extractsum!(::Type{ST}, ::PC) where {ST<:StorageType,PC<:AbstractPropagationCache}
    throw(ErrorException("extractsum!(::$(ST), ::$(PC)) not implemented."))
end