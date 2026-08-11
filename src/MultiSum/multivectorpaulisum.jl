### The data structure

"""
    MultiVectorPauliSum(psum, n_zones::Integer)

A Pauli sum split over `n_zones` work zones, each a `VectorPauliSum` that a single thread owns.
`zoneof` assigns every Pauli string to one zone, so all copies of a string reach the same owner and
deduplication never has to look outside a zone.

A gate makes products that belong to other zones. Rather than writing into a zone it does not own, a
thread parks them in an outbox, and the owner picks them up in a second pass. Every zone is then read
and written by one thread only, and no operation on a zone needs to be thread-safe.
"""
struct MultiVectorPauliSum{PC,VPS,TT}
    nqubits::Int
    zones::Vector{PC}
    outboxes::Vector{VPS}
    # counts[owner, source] terms of outbox `source` belong to zone `owner`, laid out in owner order
    counts::Matrix{Int}
    outbox_ordered::Vector{Bool}
    zonemasks::Vector{TT}
end

function MultiVectorPauliSum(thing, n_zones::Integer)
    ispow2(n_zones) || throw(ArgumentError("n_zones must be a power of two, got $n_zones."))

    vpsum = VectorPauliSum(thing)
    nq = nqubits(vpsum)
    CT = coefftype(vpsum)
    zonemasks = _zonemasks(paulitype(vpsum), trailing_zeros(n_zones))

    buckets = [(paulitype(vpsum)[], CT[]) for _ in 1:n_zones]
    for (pstr, coeff) in zip(paulis(vpsum), coefficients(vpsum))
        zone_terms, zone_coeffs = buckets[_zoneof(pstr, zonemasks)]
        push!(zone_terms, pstr)
        push!(zone_coeffs, coeff)
    end

    zones = map(buckets) do (zone_terms, zone_coeffs)
        zone = VectorPauliPropagationCache(VectorPauliSum(nq, zone_terms, zone_coeffs))
        merge!(zone; thread=false)
        zone
    end
    outboxes = [VectorPauliSum(CT, nq) for _ in 1:n_zones]

    return MultiVectorPauliSum(nq, zones, outboxes, zeros(Int, n_zones, n_zones), fill(false, n_zones), zonemasks)
end

nzones(mpsum::MultiVectorPauliSum) = length(mpsum.zones)
zonesizes(mpsum::MultiVectorPauliSum) = map(activesize, mpsum.zones)

PropagationBase.capacity(mpsum::MultiVectorPauliSum) = sum(capacity, mpsum.zones)

"""
    resize!(mpsum::MultiVectorPauliSum, n_new::Integer)

Give the zones room for `n_new` terms between them, and every outbox room for the products of its own
zone. The zone assignment is a hash, so the shares are equal.

A zone holds the products addressed to it next to the terms it already has, so its own share has to
cover that peak, not just the terms that survive the gate.
"""
function Base.resize!(mpsum::MultiVectorPauliSum, n_new::Integer)
    per_zone = cld(n_new, nzones(mpsum))
    for zone_id in 1:nzones(mpsum)
        resize!(mpsum.zones[zone_id], per_zone)
        resize!(mpsum.outboxes[zone_id], per_zone)
    end
    return mpsum
end

PauliPropagation.nqubits(mpsum::MultiVectorPauliSum) = mpsum.nqubits
Base.length(mpsum::MultiVectorPauliSum) = sum(activesize, mpsum.zones)
Base.isempty(mpsum::MultiVectorPauliSum) = length(mpsum) == 0

function Base.show(io::IO, mpsum::MultiVectorPauliSum)
    println(io, "MultiVectorPauliSum on $(mpsum.nqubits) qubits with $(length(mpsum)) terms over $(nzones(mpsum)) zones:")
    println(io, "  zone sizes: ", zonesizes(mpsum))
end


### Zone assignment

const _ZONEMASK_SEED = 0x9e3779b9

# any fixed set of masks works; random ones read every qubit, so no zone is favoured
function _zonemasks(::Type{TT}, n_bits::Int) where {TT}
    rng = MersenneTwister(_ZONEMASK_SEED)
    return TT[_randommask(TT, rng) for _ in 1:n_bits]
end

function _randommask(::Type{TT}, rng) where {TT}
    mask = zero(TT)
    for word in 0:(max(sizeof(TT), 8)÷8-1)
        mask |= (rand(rng, UInt64) % TT) << (64 * word)
    end
    return mask
end

"""
    zoneof(mpsum::MultiVectorPauliSum, pstr)

The zone that owns `pstr`. Each bit of the zone index is a parity of `pstr` under a fixed mask, which
makes the assignment linear: `zoneof(p ⊻ m) - 1 == (zoneof(p) - 1) ⊻ (zoneof(m) - 1)`. A Pauli
rotation moves every product by the same `⊻ m`, so each zone sends all of its products to exactly one
other zone, and each zone receives from exactly one.
"""
zoneof(mpsum::MultiVectorPauliSum, pstr) = _zoneof(pstr, mpsum.zonemasks)

@inline _zoneof(pstr, zonemasks) = _zonebits(pstr, zonemasks) + 1

@inline function _zonebits(pstr::TT, zonemasks::Vector{TT}) where {TT}
    bits = 0
    @inbounds for r in eachindex(zonemasks)
        bits |= Int(isodd(count_ones(pstr & zonemasks[r]))) << (r - 1)
    end
    return bits
end


### Back-conversions

function PauliPropagation.VectorPauliSum(mpsum::MultiVectorPauliSum)
    vpsum = VectorPauliSum(mpsum.nqubits, paulitype(mpsum)[], coefftype(mpsum)[])
    for zone in mpsum.zones
        append!(paulis(vpsum), activeterms(zone))
        append!(coefficients(vpsum), activecoeffs(zone))
    end
    return vpsum
end

function PauliPropagation.PauliSum(mpsum::MultiVectorPauliSum)
    psum = PauliSum(coefftype(mpsum), mpsum.nqubits)
    for zone in mpsum.zones
        for (pstr, coeff) in zip(activeterms(zone), activecoeffs(zone))
            add!(psum, pstr, coeff)
        end
    end
    return psum
end

PauliPropagation.paulitype(mpsum::MultiVectorPauliSum) = paulitype(mainsum(mpsum.zones[1]))
PropagationBase.coefftype(mpsum::MultiVectorPauliSum) = coefftype(mainsum(mpsum.zones[1]))

PauliPropagation.overlapwithzero(mpsum::MultiVectorPauliSum) = sum(z -> overlapwithzero(activesum(z)), mpsum.zones)
