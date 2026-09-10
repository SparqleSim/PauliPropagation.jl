###
##
# A Pauli sum split over one work zone per thread, with an owning zone per Pauli string.
# Everything below the type itself comes from the `MultiSumStorage` trait in `PropagationBase`.
##
###

"""
    MultiPauliSum(psum::AbstractPauliSum, n_zones=defaultnzones())
    MultiPauliSum(pstr::PauliString, n_zones=defaultnzones())
    MultiPauliSum(pstrs::Vector{PauliString}, n_zones=defaultnzones())
    MultiPauliSum(nq::Integer, n_zones=defaultnzones())
    MultiPauliSum(CoeffType, nq::Integer, n_zones=defaultnzones())

`MultiPauliSum` is a `struct` that represents a Pauli sum split over `n_zones` work zones, where every zone is a Pauli sum of the type that it was constructed from and is owned by a single thread.
`n_zones` defaults to `defaultnzones()`. Operations within a zone are single-threaded, so all parallelism comes from the zones.

`zoneof()` assigns every Pauli string to one zone, so that all copies of a Pauli string end up in the same zone and merging never has to look beyond a zone.
The Pauli strings that a gate creates for other zones are collected in an outbox and picked up by the owning zone in a second pass, rather than written into a zone that another thread owns.
Every zone is thus read and written by one thread only, and no operation on a zone needs to be thread-safe.

Splitting a `PauliSum` gives zones of `PauliSum`s and splitting a `VectorPauliSum` gives zones of `VectorPauliSum`s, and the type of the zones determines how they are propagated.

`n_zones` must be a power of two, which makes the zone assignment linear in the Pauli string and lets `PauliRotation` and the other gates that branch by a fixed bitmask take a faster path.
See `ZoneMap`.

Monte Carlo propagation (`mcpropagate()`, `mcsample()`, `resample()`) does not take a `MultiPauliSum`, and `rewindgradient()` gathers one into a `VectorPauliSum` before it runs.

# Examples
```julia
MultiPauliSum(4)                                # empty, on 4 qubits, over as many zones as threads
MultiPauliSum(PauliSum(PauliString(4, :X, 1)))  # split a Pauli sum
MultiPauliSum(VectorPauliSum(4), 8)             # empty, with VectorPauliSum zones, over 8 zones
```
"""
struct MultiPauliSum{TS<:AbstractPauliSum,ZM<:ZoneMap} <: AbstractPauliSum
    nqubits::Int
    zones::Vector{TS}
    zonemap::ZM
end

MultiPauliSum(psum::AbstractPauliSum, n_zones::Integer=defaultnzones()) =
    _fillzones!(_emptyzones(psum, nqubits(psum), n_zones), psum)

# splitting an already split sum re-zones it, so the zones are seeded from a zone and not from it
MultiPauliSum(msum::MultiPauliSum, n_zones::Integer=defaultnzones()) =
    _fillzones!(_emptyzones(first(zones(msum)), nqubits(msum), n_zones), msum)

MultiPauliSum(pstr::PauliString, n_zones::Integer=defaultnzones()) = MultiPauliSum(PauliSum(pstr), n_zones)
MultiPauliSum(pstrs::Union{AbstractArray,Tuple,Base.Generator}, n_zones::Integer=defaultnzones()) =
    MultiPauliSum(PauliSum(pstrs), n_zones)
MultiPauliSum(nq::Integer, n_zones::Integer=defaultnzones()) = MultiPauliSum(PauliSum(nq), n_zones)
MultiPauliSum(::Type{CT}, nq::Integer, n_zones::Integer=defaultnzones()) where {CT} = MultiPauliSum(PauliSum(CT, nq), n_zones)

# `seed` fixes the type the zones carry, and only the number of qubits is read off it
_emptyzones(seed::AbstractPauliSum, nq::Integer, n_zones::Integer) =
    MultiPauliSum(nq, [emptylike(seed) for _ in 1:n_zones], ZoneMap(paulitype(seed), n_zones))

function _fillzones!(msum::MultiPauliSum, psum)
    for (pstr, coeff) in psum
        PropagationBase._park!(msum, pstr, coeff)
    end
    return merge!(msum)
end

# the zones carry the trait, everything else is inherited from AbstractPauliSum
PropagationBase.storage(msum::MultiPauliSum) = msum.zones
nqubits(msum::MultiPauliSum) = msum.nqubits
paulitype(msum::MultiPauliSum) = termtype(msum)

PropagationBase.withzones(msum::MultiPauliSum, new_zones) =
    MultiPauliSum(msum.nqubits, new_zones, zonemap(msum))

convertcoefftype(::Type{CT}, msum::MultiPauliSum) where {CT} =
    withzones(msum, map(zone -> convertcoefftype(CT, zone), zones(msum)))

Base.conj!(msum::MultiPauliSum) = (foreach(conj!, zones(msum)); msum)

function Base.show(io::IO, msum::MultiPauliSum)
    println(io, "MultiPauliSum of $(nameof(eltype(zones(msum)))) with $(length(msum)) terms over $(nzones(msum)) zones:")
    println(io, "  zone sizes: ", zonesizes(msum))

    # the zones are printed in turn, so the Pauli strings do not come out in any particular order
    for (i, (pstr, coeff)) in enumerate(msum)
        if i > 20
            println(io, "  ...")
            break
        end
        println(io, "  ", coeff, " * ", inttostring(pstr, nqubits(msum)))
    end
end

"""
    PauliSum(msum::MultiPauliSum)
    VectorPauliSum(msum::MultiPauliSum)

Gather the zones of `msum` back into a single Pauli sum of the indicated type, leaving `msum` unchanged.
No merging across zones is needed, because every Pauli string is held by exactly one zone.
"""
PauliSum(::MultiPauliSum)

# every term sum type gathers the same way, so the method is written once for all of them
function (::Type{TS})(msum::MultiPauliSum) where {TS<:AbstractTermSum}
    psum = TS(coefftype(msum), nqubits(msum))
    for (pstr, coeff) in msum
        pushterm!(psum, pstr, coeff)
    end
    return psum
end

# `TS(prop_cache)` extracts the sum and converts it, so gathering has to be reachable through `convert`
Base.convert(::Type{PauliSum}, msum::MultiPauliSum) = PauliSum(msum)
Base.convert(::Type{VectorPauliSum}, msum::MultiPauliSum) = VectorPauliSum(msum)
