### symmetries.jl
##
# This file contains functions to merge Pauli strings by symmetries.
# Currently it supports the following symmetries:
# 1. Translational symmetry in 1D and 2D.
# 2. Reflection symmetry in 1D and 2D.
# 3. Permutation symmetry, i.e. all-to-all connectivity.
# The underlying bit manipulations live in `symmetry_utils.jl`.
##
###


"""
    symmetrymerge(mapfunc, psum::AbstractPauliSum) -> AbstractPauliSum

Merge equivalent Pauli strings in `psum` under a symmetry mapping.
Each Pauli string is transformed using `mapfunc(pstr)` to its canonical
representative, and identical representatives are combined.

# Arguments
- `mapfunc`: A callable mapping each integer Pauli string (`PauliStringType`) 
  to its canonical representative. It must be constant on symmetry orbits, 
  i.e. equivalent Pauli strings must map to the same integer.
- `psum`: A `PauliSum` or `VectorPauliSum` containing Pauli strings and coefficients.

# Returns
A new Pauli sum of the same type where symmetric terms have been merged.

# Example
```julia
psum = PauliSum(6)
add!(psum, :Z, 3)
add!(psum, :Z, 6)
# merge under 1D translations, the same as `translationmerge(psum)`
symmetrymerge(pstr -> _translatetolowestinteger(pstr, nqubits(psum)), psum)
```
"""
function symmetrymerge(mapfunc::F, psum::PauliSum) where F
    merged_psum = similar(psum)
    # TODO: make this work for a `mapfunc` that also modifies the coefficient
    for (pstr, coeff) in psum
        pstr = mapfunc(pstr)
        add!(merged_psum, pstr, coeff)
    end

    return merged_psum
end

# `PropagationCache` wraps `psum` without copying it, so merging through a cache built
# directly from `psum` would rewrite the caller's terms and coefficients in place.
# Copy first to keep this out-of-place method non-mutating.
function symmetrymerge(mapfunc::F, psum::VectorPauliSum) where F
    return symmetrymerge!(mapfunc, deepcopy(psum))
end

"""
    symmetrymerge!(mapfunc, psum::VectorPauliSum)
    symmetrymerge!(mapfunc, prop_cache::VectorPauliPropagationCache)

In-place version of [`symmetrymerge`](@ref) for vector-backed Pauli sums.
Returns the merged `psum` or `prop_cache`.
"""
function symmetrymerge!(mapfunc::F, psum::VectorPauliSum) where F
    cache = PropagationCache(psum)
    symmetrymerge!(mapfunc, cache)
    return extractsum!(cache, psum)
end

# `mapfunc` is left unrestricted (no `<:Function`) so that any callable,
# not only `Function`s, can be used as a mapper.
function symmetrymerge!(mapfunc::F, prop_cache::VectorPauliPropagationCache) where F
    AK.map!(mapfunc, activeterms(prop_cache), activeterms(prop_cache))
    merge!(prop_cache)
    return prop_cache
end


## Translational symmetry

"""
    translationmerge(psum::AbstractPauliSum)

Merge Pauli strings related by translations of a periodic 1D chain.

# Example
```julia
psum = PauliSum(6)
add!(psum, :Z, 3)
add!(psum, :Z, 6)
translationmerge(psum)
>>> PauliSum(nqubits: 6, 1 Pauli term: 
 2.0 * ZIIIII
)
```
"""
translationmerge(psum::AbstractPauliSum) = symmetrymerge(_translationmapper(psum), psum)

"""
    translationmerge!(psum::Union{VectorPauliSum, VectorPauliPropagationCache})

In-place version of [`translationmerge`](@ref) for a periodic 1D chain.
"""
translationmerge!(psum) = symmetrymerge!(_translationmapper(psum), psum)

function _translationmapper(psum)
    nq = nqubits(psum)
    return pstr -> _translatetolowestinteger(pstr, nq)
end

"""
    translationmerge(psum::AbstractPauliSum, nx::Integer, ny::Integer)

Merge Pauli strings related by translations of a periodic `nx` x `ny` grid.
Sites are numbered row by row, site `(x, y)` being qubit `(y - 1) * nx + x`,
consistent with `rectangletopology`.

# Example
```julia
psum = PauliSum(6)
add!(psum, :Z, 3)
add!(psum, :Z, 6)
translationmerge(psum, 3, 2)
>>> PauliSum(nqubits: 6, 1 Pauli term: 
 2.0 * ZIIIII
)
```
"""
function translationmerge(psum::AbstractPauliSum, nx::Integer, ny::Integer)
    return symmetrymerge(_translationmapper(psum, nx, ny), psum)
end

"""
    translationmerge!(psum::Union{VectorPauliSum, VectorPauliPropagationCache}, nx::Integer, ny::Integer)

In-place version of [`translationmerge`](@ref) for a periodic `nx` x `ny` grid.
"""
function translationmerge!(psum, nx::Integer, ny::Integer)
    return symmetrymerge!(_translationmapper(psum, nx, ny), psum)
end

function _translationmapper(psum, nx::Integer, ny::Integer)
    _checkgridsize(psum, nx, ny)

    # precompute masks once to accelerate shifting
    main_mask, wrap_mask = _computeshiftleftmasks(paulitype(psum), nx, ny)

    return pstr -> _translatetolowestinteger(pstr, nx, ny, main_mask, wrap_mask)
end


## Reflection symmetry

"""
    reflectionmerge(psum::AbstractPauliSum)

Merge Pauli strings related by reflection of a 1D chain, 
i.e. by reversing the order of the qubits.

# Example
```julia
psum = PauliSum(6)
add!(psum, :Z, 1)
add!(psum, :Z, 6)
reflectionmerge(psum)
>>> PauliSum(nqubits: 6, 1 Pauli term: 
 2.0 * ZIIIII
)
```
"""
reflectionmerge(psum::AbstractPauliSum) = symmetrymerge(_reflectionmapper(psum), psum)

"""
    reflectionmerge!(psum::Union{VectorPauliSum, VectorPauliPropagationCache})

In-place version of [`reflectionmerge`](@ref) for a 1D chain.
"""
reflectionmerge!(psum) = symmetrymerge!(_reflectionmapper(psum), psum)

_reflectionmapper(psum) = _lowestpermutationmapper((_chainreflection(nqubits(psum)),))

"""
    reflectionmerge(psum::AbstractPauliSum, nx::Integer, ny::Integer; axes=(:x, :y))

Merge Pauli strings related by reflections of an `nx` x `ny` grid.
Sites are numbered row by row, site `(x, y)` being qubit `(y - 1) * nx + x`,
consistent with `rectangletopology`.

`axes` selects the mirror symmetries of the system:
`:x` reflects the x coordinate (`x -> nx - x + 1`), `:y` reflects the y coordinate.
By default both are used, which merges under the full point group of the 
rectangle (both mirrors and their product, the rotation by 180 degrees).
Pass `axes=:x` or `axes=:y` for systems that are symmetric under only one mirror.

# Example
```julia
psum = PauliSum(6)
add!(psum, :Z, 1)
add!(psum, :Z, 3)
reflectionmerge(psum, 3, 2)
>>> PauliSum(nqubits: 6, 1 Pauli term: 
 2.0 * ZIIIII
)
```
"""
function reflectionmerge(psum::AbstractPauliSum, nx::Integer, ny::Integer; axes=(:x, :y))
    return symmetrymerge(_reflectionmapper(psum, nx, ny, axes), psum)
end

"""
    reflectionmerge!(psum::Union{VectorPauliSum, VectorPauliPropagationCache}, nx::Integer, ny::Integer; axes=(:x, :y))

In-place version of [`reflectionmerge`](@ref) for an `nx` x `ny` grid.
"""
function reflectionmerge!(psum, nx::Integer, ny::Integer; axes=(:x, :y))
    return symmetrymerge!(_reflectionmapper(psum, nx, ny, axes), psum)
end

function _reflectionmapper(psum, nx::Integer, ny::Integer, axes)
    _checkgridsize(psum, nx, ny)
    return _lowestpermutationmapper(_gridreflections(axes, nx, ny))
end


## Permutation symmetry

"""
    permutationmerge(psum::AbstractPauliSum)

Merge Pauli strings related by any permutation of the qubits, 
as in a system with all-to-all connectivity.
Two Pauli strings are equivalent if they contain the same number of X, Y and Z, 
and the representative of each class is the sorted string `X...X Y...Y Z...Z I...I`.

This is the largest symmetry group of the qubits, containing in particular 
translations and reflections.

# Example
```julia
psum = PauliSum(4)
add!(psum, [:Z, :X], [1, 4])
add!(psum, [:X, :Z], [2, 3])
permutationmerge(psum)
>>> PauliSum(nqubits: 4, 1 Pauli term: 
 2.0 * XZII
)
```
"""
permutationmerge(psum::AbstractPauliSum) = symmetrymerge(_permutationcanonicalform, psum)

"""
    permutationmerge!(psum::Union{VectorPauliSum, VectorPauliPropagationCache})

In-place version of [`permutationmerge`](@ref).
"""
permutationmerge!(psum) = symmetrymerge!(_permutationcanonicalform, psum)
