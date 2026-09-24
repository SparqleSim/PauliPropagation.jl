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
    symmetrymerge(mapfunc, psum::AbstractPauliSum; thread=true)
    symmetrymerge(mapfunc, prop_cache::AbstractPauliPropagationCache; thread=true)

Merge equivalent Pauli strings under a symmetry mapping, returning a copy.
Each Pauli string is transformed using `mapfunc(pstr)` to its canonical representative,
and identical representatives are combined.
`mapfunc` must be constant on symmetry orbits, i.e. equivalent Pauli strings must map to the same integer.
On array-backed sums, `thread=false` turns off multithreading, the same as in `propagate`.

# Example
```julia
psum = PauliSum(6)
add!(psum, :Z, 3)
add!(psum, :Z, 6)
# merge under 1D translations, the same as `translationmerge(psum)`
symmetrymerge(pstr -> _translatetolowestinteger(pstr, nqubits(psum)), psum)
```
"""
symmetrymerge(mapfunc::F, thing::Union{AbstractPauliSum,AbstractPauliPropagationCache}; thread::Bool=true) where {F} =
    symmetrymerge!(mapfunc, deepcopy(thing); thread)

"""
    symmetrymerge!(mapfunc, psum::AbstractPauliSum; thread=true)
    symmetrymerge!(mapfunc, prop_cache::AbstractPauliPropagationCache; thread=true)

In-place version of [`symmetrymerge`](@ref).
Returns the merged `psum` or `prop_cache`, which is the same object that was passed in.
"""
function symmetrymerge!(mapfunc::F, psum::AbstractPauliSum; thread::Bool=true) where {F}
    prop_cache = PropagationCache(psum)
    symmetrymerge!(mapfunc, prop_cache; thread)
    return extractsum!(prop_cache, psum)
end

function symmetrymerge!(mapfunc::F, prop_cache::AbstractPauliPropagationCache; thread::Bool=true) where {F}
    mapterms!(mapfunc, prop_cache; thread)
    return merge!(prop_cache; thread)
end


## Translational symmetry

"""
    translationmerge(psum::AbstractPauliSum; thread=true)
    translationmerge(prop_cache::AbstractPauliPropagationCache; thread=true)

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
translationmerge(thing::Union{AbstractPauliSum,AbstractPauliPropagationCache}; thread::Bool=true) =
    symmetrymerge(_translationmapper(thing), thing; thread)

"""
    translationmerge!(psum::AbstractPauliSum; thread=true)
    translationmerge!(prop_cache::AbstractPauliPropagationCache; thread=true)

In-place version of [`translationmerge`](@ref) for a periodic 1D chain.
"""
translationmerge!(thing; thread::Bool=true) = symmetrymerge!(_translationmapper(thing), thing; thread)

function _translationmapper(thing)
    nq = nqubits(thing)
    return pstr -> _translatetolowestinteger(pstr, nq)
end


"""
    translationmerge(psum::AbstractPauliSum, nx::Integer, ny::Integer; thread=true)
    translationmerge(prop_cache::AbstractPauliPropagationCache, nx::Integer, ny::Integer; thread=true)

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
function translationmerge(thing::Union{AbstractPauliSum,AbstractPauliPropagationCache}, nx::Integer, ny::Integer; thread::Bool=true)
    return symmetrymerge(_translationmapper(thing, nx, ny), thing; thread)
end

"""
    translationmerge!(psum::AbstractPauliSum, nx::Integer, ny::Integer; thread=true)
    translationmerge!(prop_cache::AbstractPauliPropagationCache, nx::Integer, ny::Integer; thread=true)

In-place version of [`translationmerge`](@ref) for a periodic `nx` x `ny` grid.
"""
function translationmerge!(thing, nx::Integer, ny::Integer; thread::Bool=true)
    return symmetrymerge!(_translationmapper(thing, nx, ny), thing; thread)
end

# builds and returns the canonicalization function; the merge itself
# happens in `symmetrymerge`/`symmetrymerge!`
function _translationmapper(thing, nx::Integer, ny::Integer)
    _checkgridsize(thing, nx, ny)

    # precompute masks once to accelerate shifting
    main_mask, wrap_mask = _computeshiftleftmasks(paulitype(thing), nx, ny)

    return pstr -> _translatetolowestinteger(pstr, nx, ny, main_mask, wrap_mask)
end


## Reflection symmetry

"""
    reflectionmerge(psum::AbstractPauliSum; thread=true)
    reflectionmerge(prop_cache::AbstractPauliPropagationCache; thread=true)

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
reflectionmerge(thing::Union{AbstractPauliSum,AbstractPauliPropagationCache}; thread::Bool=true) =
    symmetrymerge(_reflectionmapper(thing), thing; thread)

"""
    reflectionmerge!(psum::AbstractPauliSum; thread=true)
    reflectionmerge!(prop_cache::AbstractPauliPropagationCache; thread=true)

In-place version of [`reflectionmerge`](@ref) for a 1D chain.
"""
reflectionmerge!(psum; thread::Bool=true) = symmetrymerge!(_reflectionmapper(psum), psum; thread)

_reflectionmapper(psum) = _lowestpermutationmapper((_chainreflection(nqubits(psum)),))

"""
    reflectionmerge(psum::AbstractPauliSum, nx::Integer, ny::Integer; axes=(:x, :y), thread=true)
    reflectionmerge(prop_cache::AbstractPauliPropagationCache, nx::Integer, ny::Integer; axes=(:x, :y), thread=true)

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
function reflectionmerge(thing::Union{AbstractPauliSum,AbstractPauliPropagationCache}, nx::Integer, ny::Integer; axes=(:x, :y), thread::Bool=true)
    return symmetrymerge(_reflectionmapper(thing, nx, ny, axes), thing; thread)
end

"""
    reflectionmerge!(psum::AbstractPauliSum, nx::Integer, ny::Integer; axes=(:x, :y), thread=true)
    reflectionmerge!(prop_cache::AbstractPauliPropagationCache, nx::Integer, ny::Integer; axes=(:x, :y), thread=true)

In-place version of [`reflectionmerge`](@ref) for an `nx` x `ny` grid.
"""
function reflectionmerge!(psum, nx::Integer, ny::Integer; axes=(:x, :y), thread::Bool=true)
    return symmetrymerge!(_reflectionmapper(psum, nx, ny, axes), psum; thread)
end

function _reflectionmapper(psum, nx::Integer, ny::Integer, axes)
    _checkgridsize(psum, nx, ny)
    return _lowestpermutationmapper(_gridreflections(axes, nx, ny))
end


## Permutation symmetry

"""
    permutationmerge(psum::AbstractPauliSum; thread=true)
    permutationmerge(prop_cache::AbstractPauliPropagationCache; thread=true)

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
permutationmerge(thing::Union{AbstractPauliSum,AbstractPauliPropagationCache}; thread::Bool=true) =
    symmetrymerge(_permutationcanonicalform, thing; thread)

"""
    permutationmerge!(psum::AbstractPauliSum; thread=true)
    permutationmerge!(prop_cache::AbstractPauliPropagationCache; thread=true)

In-place version of [`permutationmerge`](@ref).
"""
permutationmerge!(psum; thread::Bool=true) = symmetrymerge!(_permutationcanonicalform, psum; thread)

"""
    permutationmerge(psum::AbstractPauliSum, blocks; thread=true)
    permutationmerge(prop_cache::AbstractPauliPropagationCache, blocks; thread=true)
    permutationmerge!(psum::AbstractPauliSum, blocks; thread=true)
    permutationmerge!(prop_cache::AbstractPauliPropagationCache, blocks; thread=true)

Merge Pauli strings related by permutations within each of the contiguous site blocks
`blocks = ((lo_1, hi_1), ..., (lo_k, hi_k))`, i.e. under `S_{B_1} x ... x S_{B_k}`.
The blocks must partition `1:nqubits(psum)` in order; empty blocks (`hi < lo`) are allowed.
Within each block the representative is the sorted string `X...X Y...Y Z...Z I...I`.
A single block `((1, nqubits),)` is the full permutation merge.

This is the merge that stays valid *inside* a block of commuting all-to-all gates applied
one by one, see [`residualpermutationblocks`](@ref): merging after every gate keeps the
intermediate Pauli sum polynomially small instead of letting it expand until the block ends.

# Example
```julia
psum = PauliSum(4)
add!(psum, [:X, :Z], [1, 3])
add!(psum, [:X, :Z], [2, 4])
permutationmerge(psum, ((1, 2), (3, 4)))   # swaps within {1,2} and within {3,4}
>>> PauliSum(nqubits: 4, 1 Pauli term: 
 2.0 * XIZI
)
```
"""
function permutationmerge(thing::Union{AbstractPauliSum,AbstractPauliPropagationCache}, blocks; thread::Bool=true)
    _checkblocks(nqubits(thing), blocks)
    return symmetrymerge(pstr -> _permutationcanonicalform(pstr, blocks), thing; thread)
end

function permutationmerge!(psum, blocks; thread::Bool=true)
    _checkblocks(nqubits(psum), blocks)
    return symmetrymerge!(pstr -> _permutationcanonicalform(pstr, blocks), psum; thread)
end
