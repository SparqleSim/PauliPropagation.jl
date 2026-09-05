###
##
# A Pauli sum split over one work zone per thread, with an owning zone per Pauli string.
# Everything below the type itself comes from the `MultiSumStorage` trait in `PropagationBase`.
##
###

using .PropagationBase: _park!

"""
    MultiPauliSum(psum::AbstractPauliSum, n_zones=defaultnzones())
    MultiPauliSum(pstr::PauliString, n_zones=defaultnzones())
    MultiPauliSum(pstrs::Vector{PauliString}, n_zones=defaultnzones())
    MultiPauliSum(nq::Integer, n_zones=defaultnzones())
    MultiPauliSum(CoeffType, nq::Integer, n_zones=defaultnzones())

A Pauli sum split over `n_zones` work zones, each a Pauli sum of the type it was built from that a
single thread owns. `n_zones` defaults to the number of threads. Every zone runs single-threaded, so
parallelism comes from the zones alone.

`zoneof` assigns every Pauli string to one zone, so all copies of a Pauli string reach the same owner
and deduplication never has to look outside a zone. A gate makes Pauli strings that belong to other
zones. Rather than writing into a zone it does not own, a thread parks them in an outbox, and the
owner picks them up in a second pass. Every zone is then read and written by one thread only, and no
operation on a zone needs to be thread-safe.

Splitting a `PauliSum` gives zones of `PauliSum`s, splitting a `VectorPauliSum` gives zones of
`VectorPauliSum`s, and the zone type is what decides how a zone propagates.

`n_zones` must be a power of two, which makes the zone assignment linear in the Pauli string and lets
`PauliRotation` and the other gates that branch by a fixed mask take a faster path. See
[`ZoneMap`](@ref).

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
        _park!(msum, pstr, coeff)
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

Gather the zones of `msum` back into a single Pauli sum of the given type. Every zone holds Pauli
strings no other zone holds, so this needs no deduplication across zones, and it leaves `msum`
unchanged.
"""
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
