### groups.jl
##
# Symmetry groups of the qubit sites as objects. A group knows how to merge (its canonical form,
# used by `symmetrymerge!`); its stabilizers inside a partially applied layer are computed in
# subgroups.jl. Site permutations and their action on Pauli strings are in `symmetry_utils.jl`:
# `perm[i]` is the image of site `i`, and the Pauli on site `i` moves to site `perm[i]`.
##
###


"""
    AbstractSiteSymmetry

A finite group of permutations of the qubit sites. A subtype provides `nqubits(G)` and
`elements(G)`, all its permutations with the identity first, and may override what is derived
from them: `generators` (all but the identity), `grouporder` (their number), `istrivial` (order
1), `canonicalform(G, TT)` (the smallest image of a Pauli string) and the stabilizer in
subgroups.jl. Two groups are equal when they have the same qubit count and the same elements,
whatever their types. Concrete groups: [`PermutationSymmetry`](@ref), [`TranslationSymmetry`](@ref),
[`ReflectionSymmetry`](@ref), [`SiteSymmetry`](@ref) and [`TrivialSymmetry`](@ref).
"""
abstract type AbstractSiteSymmetry end

# all site permutations of G, the identity first
function elements end
# whether G contains only the identity, so that merging under it does nothing
istrivial(G::AbstractSiteSymmetry) = grouporder(G) == 1
# permutations that certify invariance of a layer; all but the identity unless a group knows better
generators(G::AbstractSiteSymmetry) = elements(G)[2:end]
# the number of elements (a BigInt for PermutationSymmetry)
grouporder(G::AbstractSiteSymmetry) = length(elements(G))
# a function mapping a Pauli string of type TT to a fixed representative of its G-orbit
canonicalform(G::AbstractSiteSymmetry, ::Type) = _lowestimagemapper(elements(G)[2:end])

# groups of different types can be equal; the hash uses what every group can provide cheaply
Base.:(==)(G1::AbstractSiteSymmetry, G2::AbstractSiteSymmetry) =
    nqubits(G1) == nqubits(G2) && grouporder(G1) == grouporder(G2) && Set(elements(G1)) == Set(elements(G2))
Base.hash(G::AbstractSiteSymmetry, h::UInt) = hash((nqubits(G), grouporder(G)), h)

_checknqubits(G::AbstractSiteSymmetry, thing) =
    nqubits(G) == nqubits(thing) || throw(ArgumentError(
        "The symmetry acts on $(nqubits(G)) qubits but the Pauli sum has $(nqubits(thing))."))


## Merging under a group

# merge `thing` under G, without truncating
function _mergeunder!(G::AbstractSiteSymmetry, thing; thread::Bool)
    _checknqubits(G, thing)
    istrivial(G) && return thing
    return symmetrymerge!(canonicalform(G, paulitype(thing)), thing; thread)
end

"""
    symmetrymerge!(G::AbstractSiteSymmetry, psum; thread=true)
    symmetrymerge(G::AbstractSiteSymmetry, psum; thread=true)

Merge the Pauli strings of `psum` (or of a propagation cache) that lie in the same orbit of the
symmetry group `G`, in place or on a copy.
"""
symmetrymerge!(G::AbstractSiteSymmetry, psum::AbstractPauliSum; thread::Bool=true) = _mergeunder!(G, psum; thread)
# two methods rather than one with a `Union` argument, which would be ambiguous with the
# `symmetrymerge!(mapfunc, psum::AbstractPauliSum)` method in symmetries.jl
symmetrymerge!(G::AbstractSiteSymmetry, prop_cache::AbstractPauliPropagationCache; thread::Bool=true) = _mergeunder!(G, prop_cache; thread)
symmetrymerge(G::AbstractSiteSymmetry, thing::Union{AbstractPauliSum,AbstractPauliPropagationCache}; thread::Bool=true) =
    symmetrymerge!(G, deepcopy(thing); thread)


## The trivial group

"""
    TrivialSymmetry(nqubits)

The group containing only the identity, under which merging does nothing; the stabilizer of a
set of gates when no symmetry is left.
"""
struct TrivialSymmetry <: AbstractSiteSymmetry
    nqubits::Int
end

nqubits(G::TrivialSymmetry) = G.nqubits
elements(G::TrivialSymmetry) = [collect(1:G.nqubits)]
canonicalform(::TrivialSymmetry, ::Type) = identity
Base.show(io::IO, G::TrivialSymmetry) = print(io, "TrivialSymmetry($(G.nqubits))")


## The symmetric group and its Young subgroups

"""
    PermutationSymmetry(nqubits)
    PermutationSymmetry(nqubits, classes)

The group of all permutations of the `nqubits` sites (all-to-all connectivity), or, given
`classes` (disjoint site collections), the subgroup permuting the sites of each class among
themselves. The orbit of a Pauli string is fixed by how many `X`, `Y` and `Z` it has in each
class, so the canonical representative sorts each class as `X...X Y...Y Z...Z I...I` and merging
never enumerates the group.

# Example
```julia
psum = PauliSum(4)
add!(psum, [:Z, :X], [1, 4])
add!(psum, [:X, :Z], [2, 3])
symmetrymerge(PermutationSymmetry(4), psum)
>>> PauliSum(nqubits: 4, 1 Pauli term:
 2.0 * XZII
)
symmetrymerge(PermutationSymmetry(4, [[1, 2], [3, 4]]), psum)   # only swaps within {1,2} and {3,4}
>>> PauliSum(nqubits: 4, 2 Pauli terms:
 1.0 * XIZI
 1.0 * ZIXI
)
```
"""
struct PermutationSymmetry <: AbstractSiteSymmetry
    nqubits::Int
    classes::Vector{Vector{Int}}   # disjoint, each sorted and of length >= 2, ordered by first site

    function PermutationSymmetry(nqubits::Integer, classes)
        nqubits >= 1 || throw(ArgumentError("Need at least one qubit, got $(nqubits)."))
        cleaned = [sort!(collect(Int, class)) for class in classes]
        sites = reduce(vcat, cleaned; init=Int[])
        all(in(1:nqubits), sites) || throw(ArgumentError("The classes $(classes) are not within 1:$(nqubits)."))
        allunique(sites) || throw(ArgumentError("The classes $(classes) overlap."))
        filter!(class -> length(class) >= 2, cleaned)
        return new(Int(nqubits), sort!(cleaned; by=first))
    end
end

PermutationSymmetry(nqubits::Integer) = PermutationSymmetry(nqubits, (1:nqubits,))

nqubits(G::PermutationSymmetry) = G.nqubits
istrivial(G::PermutationSymmetry) = isempty(G.classes)
grouporder(G::PermutationSymmetry) = prod(factorial(big(length(class))) for class in G.classes; init=big(1))
elements(::PermutationSymmetry) = throw(ArgumentError(
    "The elements of a PermutationSymmetry are not listed; use `generators`."))
Base.:(==)(G1::PermutationSymmetry, G2::PermutationSymmetry) = G1.nqubits == G2.nqubits && G1.classes == G2.classes

# a listed group equals a PermutationSymmetry if it has the same order and all its elements
# permute the sites within the classes
Base.:(==)(G1::PermutationSymmetry, G2::AbstractSiteSymmetry) =
    nqubits(G1) == nqubits(G2) && grouporder(G1) == grouporder(G2) && all(perm -> _preservesclasses(G1, perm), elements(G2))
Base.:(==)(G1::AbstractSiteSymmetry, G2::PermutationSymmetry) = G2 == G1

function _preservesclasses(G::PermutationSymmetry, perm)
    class_of = zeros(Int, G.nqubits)
    for (k, class) in enumerate(G.classes), site in class
        class_of[site] = k
    end
    return all(class_of[site] == class_of[perm[site]] && (class_of[site] != 0 || perm[site] == site) for site in 1:G.nqubits)
end

_isfullgroup(G::PermutationSymmetry) = length(G.classes) == 1 && length(G.classes[1]) == G.nqubits
_iscontiguous(sites::Vector{Int}) = last(sites) - first(sites) + 1 == length(sites)

function Base.show(io::IO, G::PermutationSymmetry)
    _isfullgroup(G) && return print(io, "PermutationSymmetry($(G.nqubits))")
    ranges = [_iscontiguous(c) ? "$(first(c)):$(last(c))" : string(c) for c in G.classes]
    print(io, "PermutationSymmetry($(G.nqubits), [", join(ranges, ", "), "])")
end

# adjacent transpositions inside every class generate the Young subgroup
function generators(G::PermutationSymmetry)
    gens = Vector{Int}[]
    for class in G.classes, k in 1:length(class)-1
        perm = collect(1:G.nqubits)
        perm[class[k]], perm[class[k+1]] = class[k+1], class[k]
        push!(gens, perm)
    end
    return gens
end

# The representative sorts the Paulis inside each class as X...X Y...Y Z...Z I...I. Three
# implementations of the same map, fastest first: the whole string and contiguous classes as bit
# blocks (`_permutationcanonicalform` in symmetry_utils.jl), and a general per-site one.
function canonicalform(G::PermutationSymmetry, ::Type{TT}) where {TT<:PauliStringType}
    _isfullgroup(G) && return _permutationcanonicalform
    if all(_iscontiguous, G.classes)
        blocks = Tuple((first(class), last(class)) for class in G.classes)
        return pstr -> _permutationcanonicalform(pstr, blocks)
    end
    classes = G.classes
    return pstr -> _sortwithinclasses(pstr, classes)
end

function _sortwithinclasses(pstr::TT, classes) where {TT<:PauliStringType}
    for sites in classes
        num_x = num_y = num_z = 0
        for site in sites
            pauli = _getpaulibits(pstr, site)
            num_x += pauli == TT(1)
            num_y += pauli == TT(2)
            num_z += pauli == TT(3)
        end
        for (k, site) in enumerate(sites)
            pauli = k <= num_x ? TT(1) : k <= num_x + num_y ? TT(2) : k <= num_x + num_y + num_z ? TT(3) : TT(0)
            pstr = _setpaulibits(pstr, pauli, site)
        end
    end
    return pstr
end


## Groups given by their elements

"""
    SiteSymmetry(nqubits, generators; maxorder=100_000)

The group of site permutations generated by `generators`, each a vector `perm` with `perm[i]`
the image of site `i`. All elements are listed, so this is meant for small groups: point groups,
products of translations and reflections, the stabilizers inside a layer (subgroups.jl).

# Example
```julia
# the dihedral group of a ring of 6 sites: rotations and a reflection
G = SiteSymmetry(6, [[2, 3, 4, 5, 6, 1], [6, 5, 4, 3, 2, 1]])
```
"""
struct SiteSymmetry <: AbstractSiteSymmetry
    nqubits::Int
    elements::Vector{Vector{Int}}   # closed under composition, identity first

    function SiteSymmetry(nqubits::Integer, generators; maxorder::Integer=100_000)
        nqubits >= 1 || throw(ArgumentError("Need at least one qubit, got $(nqubits)."))
        return new(Int(nqubits), _closure(nqubits, generators; maxorder))
    end

    # for an element list that is known to be closed, with the identity first
    SiteSymmetry(nqubits::Integer, elements::Vector{Vector{Int}}, ::Val{:closed}) = new(Int(nqubits), elements)
end

nqubits(G::SiteSymmetry) = G.nqubits
elements(G::SiteSymmetry) = G.elements
Base.show(io::IO, G::SiteSymmetry) = print(io, "SiteSymmetry($(G.nqubits), order $(grouporder(G)))")

"""
    ReflectionSymmetry(nqubits)
    ReflectionSymmetry(nx, ny; axes=(:x, :y))

The reflection group of a chain (`site i <-> nqubits - i + 1`) or of an `nx` x `ny` grid with
mirrors along the given axes; the groups that [`reflectionmerge`](@ref) merges under.
"""
struct ReflectionSymmetry <: AbstractSiteSymmetry
    nx::Int
    ny::Int
    axes::Tuple{Vararg{Symbol}}
    elements::Vector{Vector{Int}}

    function ReflectionSymmetry(nx::Integer, ny::Integer; axes=(:x, :y))
        (nx >= 1 && ny >= 1) || throw(ArgumentError("Grid sizes must be positive, got nx=$(nx), ny=$(ny)."))
        axes = axes isa Symbol ? (axes,) : Tuple(axes)
        return new(Int(nx), Int(ny), axes, _closure(nx * ny, _gridreflections(axes, nx, ny); maxorder=8))
    end
end

ReflectionSymmetry(nqubits::Integer) = ReflectionSymmetry(nqubits, 1; axes=:x)

nqubits(G::ReflectionSymmetry) = G.nx * G.ny
elements(G::ReflectionSymmetry) = G.elements
Base.show(io::IO, G::ReflectionSymmetry) =
    G.ny == 1 ? print(io, "ReflectionSymmetry($(G.nx))") : print(io, "ReflectionSymmetry($(G.nx), $(G.ny), axes=$(G.axes))")


## Translations

"""
    TranslationSymmetry(nqubits)
    TranslationSymmetry(nx, ny)

The cyclic translations of a periodic chain, or of a periodic `nx` x `ny` grid with site `(x, y)`
numbered `(y - 1) * nx + x` as in `rectangletopology`. Merging uses the bit-shift canonical form
of [`translationmerge`](@ref); the stabilizers inside a layer are [`SiteSymmetry`](@ref) groups.
"""
struct TranslationSymmetry <: AbstractSiteSymmetry
    nx::Int
    ny::Int

    function TranslationSymmetry(nx::Integer, ny::Integer)
        (nx >= 1 && ny >= 1) || throw(ArgumentError("Grid sizes must be positive, got nx=$(nx), ny=$(ny)."))
        return new(Int(nx), Int(ny))
    end
end

TranslationSymmetry(nqubits::Integer) = TranslationSymmetry(nqubits, 1)

nqubits(G::TranslationSymmetry) = G.nx * G.ny
grouporder(G::TranslationSymmetry) = nqubits(G)
Base.show(io::IO, G::TranslationSymmetry) =
    G.ny == 1 ? print(io, "TranslationSymmetry($(G.nx))") : print(io, "TranslationSymmetry($(G.nx), $(G.ny))")

# the site permutation shifting the grid by (dx, dy)
function _gridshift(G::TranslationSymmetry, dx::Integer, dy::Integer)
    perm = Vector{Int}(undef, nqubits(G))
    for site in 1:nqubits(G)
        x, y = _indextocoord(site, G.nx)
        perm[site] = _coordtoindex(mod1(x + dx, G.nx), mod1(y + dy, G.ny), G.nx)
    end
    return perm
end

elements(G::TranslationSymmetry) = [_gridshift(G, dx, dy) for dy in 0:G.ny-1 for dx in 0:G.nx-1]

function generators(G::TranslationSymmetry)
    gens = Vector{Int}[]
    G.nx > 1 && push!(gens, _gridshift(G, 1, 0))
    G.ny > 1 && push!(gens, _gridshift(G, 0, 1))
    return gens
end

function canonicalform(G::TranslationSymmetry, ::Type{TT}) where {TT<:PauliStringType}
    nq = nqubits(G)
    G.ny == 1 && return pstr -> _translatetolowestinteger(pstr, nq)
    nx, ny = G.nx, G.ny
    main_mask, wrap_mask = _computeshiftleftmasks(TT, nx, ny)
    return pstr -> _translatetolowestinteger(pstr, nx, ny, main_mask, wrap_mask)
end
