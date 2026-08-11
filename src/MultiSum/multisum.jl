###
##
# A term sum split over one work zone per thread, with an owning zone per term.
# Every zone is a term sum of the carried type, and every operation on a zone is single-threaded.
##
###

using .PropagationBase: DictStorage, ArrayStorage

"""
    MultiSum(term_sum::AbstractTermSum, n_zones::Integer)

A term sum split over `n_zones` work zones, each a term sum of the type of `term_sum` that a single
thread owns. `n_zones` must be a power of two. Every zone runs single-threaded, so parallelism comes
from the zones alone.

`zoneof` assigns every term to one zone, so all copies of a term reach the same owner and
deduplication never has to look outside a zone. A gate makes terms that belong to other zones. Rather
than writing into a zone it does not own, a thread parks them in an outbox, and the owner picks them
up in a second pass. Every zone is then read and written by one thread only, and no operation on a
zone needs to be thread-safe.
"""
struct MultiSum{TS<:AbstractTermSum,TT} <: AbstractTermSum
    zones::Vector{TS}
    zonemasks::Vector{TT}
end

function MultiSum(term_sum::AbstractTermSum, n_zones::Integer)
    ispow2(n_zones) || throw(ArgumentError("n_zones must be a power of two, got $n_zones."))

    zonemasks = _zonemasks(termtype(term_sum), trailing_zeros(n_zones))
    msum = MultiSum([_emptylike(term_sum) for _ in 1:n_zones], zonemasks)

    for (term, coeff) in term_sum
        _park!(msum, term, coeff)
    end

    return merge!(msum)
end

MultiSum(pstr::PauliString, n_zones::Integer) = MultiSum(PauliSum(pstr), n_zones)

"""
    nzones(msum::MultiSum)

The number of work zones that `msum` is split over.
"""
nzones(msum::MultiSum) = length(msum.zones)

"""
    zonesizes(msum::MultiSum)

The number of terms in each zone of `msum`.
"""
zonesizes(msum::MultiSum) = map(length, msum.zones)

function Base.show(io::IO, msum::MultiSum)
    println(io, "MultiSum of $(nameof(eltype(msum.zones))) with $(length(msum)) terms over $(nzones(msum)) zones:")
    println(io, "  zone sizes: ", zonesizes(msum))
end


### Zone assignment

"""
    zoneof(msum::MultiSum, term)

The zone that owns `term`. Each bit of the zone index is a parity of `term` under a fixed mask, which
makes the assignment linear: `zoneof(t ⊻ m) - 1 == (zoneof(t) - 1) ⊻ (zoneof(m) - 1)`. A Pauli
rotation moves every term it branches by the same `⊻ m`, so each zone sends all of them to exactly one
other zone, and each zone receives from exactly one.
"""
zoneof(msum::MultiSum, term) = _zonebits(term, msum.zonemasks) + 1

@inline _zone(msum::MultiSum, term) = @inbounds msum.zones[zoneof(msum, term)]

@inline function _zonebits(term, zonemasks)
    bits = 0
    @inbounds for r in eachindex(zonemasks)
        bits |= Int(isodd(count_ones(term & zonemasks[r]))) << (r - 1)
    end
    return bits
end


### The term sum interface, routed through the owning zone

PropagationBase.storage(msum::MultiSum) = msum.zones
PropagationBase.StorageType(msum::MultiSum) = StorageType(first(msum.zones))

Base.length(msum::MultiSum) = sum(length, msum.zones)
Base.isempty(msum::MultiSum) = all(isempty, msum.zones)
PropagationBase.nsites(msum::MultiSum) = nsites(first(msum.zones))
PropagationBase.termtype(msum::MultiSum) = termtype(first(msum.zones))
PropagationBase.coefftype(msum::MultiSum) = coefftype(first(msum.zones))

PropagationBase.terms(msum::MultiSum) = Iterators.flatten(terms(zone) for zone in msum.zones)
PropagationBase.coefficients(msum::MultiSum) = Iterators.flatten(coefficients(zone) for zone in msum.zones)
@inline Base.iterate(msum::MultiSum, args...) = iterate(Iterators.flatten(msum.zones), args...)

PropagationBase.getcoeff(msum::MultiSum, term) = getcoeff(_zone(msum, term), term)
PropagationBase.getmergedcoeff(msum::MultiSum, term) = getmergedcoeff(_zone(msum, term), term)

PropagationBase.add!(msum::MultiSum, term, coeff) = (add!(_zone(msum, term), term, coeff); msum)
PropagationBase.set!(msum::MultiSum, term, coeff) = (set!(_zone(msum, term), term, coeff); msum)
Base.delete!(msum::MultiSum, term) = (delete!(_zone(msum, term), term); msum)

PropagationBase.mult!(msum::MultiSum, scalar::Number) = (foreach(zone -> mult!(zone, scalar), msum.zones); msum)
Base.empty!(msum::MultiSum) = (foreach(empty!, msum.zones); msum)
Base.merge!(msum::MultiSum) = (foreach(merge!, msum.zones); msum)
Base.similar(msum::MultiSum) = MultiSum(map(_emptylike, msum.zones), msum.zonemasks)

PropagationBase.maxabscoeff(msum::MultiSum) = maximum(maxabscoeff, msum.zones)

# a p-norm over the zones' p-norms is the p-norm over all coefficients
LinearAlgebra.norm(msum::MultiSum, L::Real=2) = norm([norm(zone, L) for zone in msum.zones], L)

function PropagationBase.truncate!(msum::MultiSum; kwargs...)
    foreach(zone -> truncate!(zone; kwargs...), msum.zones)
    return msum
end

function PropagationBase.truncate!(truncfunc::F, msum::MultiSum; kwargs...) where {F<:Function}
    foreach(zone -> truncate!(truncfunc, zone; kwargs...), msum.zones)
    return msum
end

function Base.copy!(dst_msum::MS, src_msum::MS) where {MS<:MultiSum}
    foreach(copy!, dst_msum.zones, src_msum.zones)
    return dst_msum
end


### Pauli conveniences, which a MultiSum does not inherit from AbstractPauliSum

nqubits(msum::MultiSum) = nqubits(first(msum.zones))
paulis(msum::MultiSum) = terms(msum)
paulitype(msum::MultiSum) = termtype(msum)
topaulistrings(msum::MultiSum) = [PauliString(nqubits(msum), term, coeff) for (term, coeff) in msum]

PropagationBase.add!(msum::MultiSum, pstr::PauliString) = add!(msum, pstr.term, convert(coefftype(msum), pstr.coeff))

function PropagationBase.add!(msum::MultiSum, paulis::Union{Symbol,Vector{Symbol}}, qinds, coeff=coefftype(msum)(1.0))
    _check_qind_range(nqubits(msum), qinds)
    return add!(msum, PauliString(nqubits(msum), paulis, qinds, coeff))
end

PropagationBase.getcoeff(msum::MultiSum, pstr::PauliString) = getcoeff(msum, pstr.term)
PropagationBase.getcoeff(msum::MultiSum, pauli::Symbol, qind::Integer) = getcoeff(msum, symboltoint(nqubits(msum), pauli, qind))
PropagationBase.getcoeff(msum::MultiSum, pstr::Vector{Symbol}) = getcoeff(msum, symboltoint(pstr))
PropagationBase.getcoeff(msum::MultiSum, pstr, qinds) = getcoeff(msum, symboltoint(nqubits(msum), pstr, qinds))


### Conversions

# every zone holds terms no other zone holds, so gathering them needs no deduplication
function (::Type{TS})(msum::MultiSum) where {TS<:AbstractTermSum}
    term_sum = TS(coefftype(msum), nsites(msum))
    for (term, coeff) in msum
        _park!(StorageType(term_sum), term_sum, term, coeff)
    end
    return term_sum
end


### Parking terms without deduplication

@inline _park!(msum::MultiSum, term, coeff) = _park!(StorageType(msum), _zone(msum, term), term, coeff)

@inline _park!(::DictStorage, term_sum, term, coeff) = add!(term_sum, term, coeff)

@inline function _park!(::ArrayStorage, term_sum, term, coeff)
    push!(terms(term_sum), term)
    push!(coefficients(term_sum), coeff)
    return term_sum
end


### Zone masks

const _ZONEMASK_SEED = 0x9e3779b97f4a7c15

# any fixed set of masks works; pseudo-random ones read every site, so no zone is favoured
function _zonemasks(::Type{TT}, n_bits::Int) where {TT}
    masks = TT[]
    state = _ZONEMASK_SEED

    for _ in 1:n_bits
        mask = zero(TT)
        for word in 0:max(sizeof(TT), 8)÷8-1
            state += _ZONEMASK_SEED
            mask |= (_mix64(state) % TT) << (64 * word)
        end
        push!(masks, mask)
    end

    return masks
end

@inline function _mix64(x::UInt64)
    z = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

_emptylike(term_sum::TS) where {TS<:AbstractTermSum} = Base.typename(TS).wrapper(coefftype(term_sum), nsites(term_sum))
