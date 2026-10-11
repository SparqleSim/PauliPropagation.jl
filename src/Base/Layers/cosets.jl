###
##
# Layers of commuting gates applied in one pass over the sum, coset by coset, through the gate path. A gate here leaves
# a term unchanged or mixes it with its partner, the term with the bits of the gate's mask flipped, by a rule of
# `xorbranch!`; a Pauli rotation is one. The gates of a layer commute: a term and the term with the mask of any gate of
# the layer flipped are acted on by the same gates. So the gates of the layer act on all terms of a coset of the span of
# their masks or on none of them, and mix only terms of one coset. The plan holds the reduced echelon basis of that
# span, by which a term is reduced to the representative of its coset. In one pass over the sum (`_applytolabels!`) the
# terms are sorted by the hash of their representative, and every coset is loaded into a small sum of its own, to which
# the gates acting on it are applied one after the other in the order of the layer as `propagate!` applies gates,
# through `xorbranchmergeandtruncate!` with the rule of each, with the truncations after each, as the gates one by one do.
#
# A basis provides, for a layer, a `LayerPlan` with the masks and rules of the gates and a lookup that finds the gates
# acting on a term (`_branchinggates`, a `PrecomputedLookup` built from the masks with which the basis decides whether a
# gate acts), and applies it with `_applypass!`.
##
###

# The bits of the basis vectors besides their pivot bits, added up for the pivot bits of a term in three parts: the
# shifts that the most basis vectors share, from their pivot bit to another of their bits, each with the pivot bits it
# moves; the vector that the most basis vectors share after those, added where an odd number of their pivot bits is set;
# and what is left of every basis vector, added bit by bit from `remainders` at the position of its pivot bit, plus one.
# On a lattice, the first two parts take all of it.
struct _Reduction{TT}
    shifts::NTuple{3,UInt}
    shifted_pivots::NTuple{3,TT}
    shared_vector::TT
    sharing_pivots::TT
    remaining_pivots::TT
    remainders::Vector{TT}

    function _Reduction{TT}(shifts, shifted_pivots, shared_vector, sharing_pivots, remaining_pivots, remainders) where {TT}
        # `_reduce` reads the remainders without bounds checks
        if length(remainders) != 8 * sizeof(TT)
            throw(ArgumentError("a reduction of terms of $(8 * sizeof(TT)) bits has $(length(remainders)) remainders"))
        end
        return new{TT}(shifts, shifted_pivots, shared_vector, sharing_pivots, remaining_pivots, remainders)
    end
end

"""
    LayerPlan(gate_masks, rules, lookup)

The gates of a layer, prepared for one pass over a sum: `gate_masks[i]` are the bits that the gate `i` flips,
`rules[i]` is its rule for `xorbranchmergeandtruncate!`, and `lookup` finds the gates that act on a term, as
`_branchinggates` describes.
"""
struct LayerPlan{TT,R,L}
    gate_masks::Vector{TT}
    rules::Vector{R}
    lookup::L

    # the reduced echelon basis of the gate masks: its pivot bits, and the other bits of the basis vectors of any pivot
    # bits
    pivots::TT
    reduction::_Reduction{TT}
end

function LayerPlan(gate_masks::Vector{TT}, rules::Vector{R}, lookup::L) where {TT,R,L}
    pivots, reductions = _echelonbasis(gate_masks)
    return LayerPlan{TT,R,L}(gate_masks, rules, lookup, pivots, _Reduction(reductions))
end

# The reduced echelon basis of the space that `gate_masks` span: its pivot bits, and at the position of every pivot bit,
# plus one, the other bits of the basis vector with that pivot. The pivot of a basis vector is its lowest bit, which no
# other basis vector has. A gate mask that adds nothing to the gate masks before it adds no basis vector.
function _echelonbasis(gate_masks::Vector{TT}) where {TT}
    basis = zeros(TT, 8 * sizeof(TT))
    pivots = zero(TT)
    for gate_mask in gate_masks
        # the basis vectors of the pivot bits of the gate mask clear those bits and set no other pivot bit
        vector = gate_mask ⊻ _xorofbits(basis, gate_mask & pivots)
        if iszero(vector)
            continue
        end
        pivot = trailing_zeros(vector)
        pivot_bit = one(TT) << pivot
        for position in _bitpositions(pivots)
            if !iszero(basis[position+1] & pivot_bit)
                basis[position+1] ⊻= vector
            end
        end
        basis[pivot+1] = vector
        pivots |= pivot_bit
    end
    reductions = zeros(TT, length(basis))
    for position in _bitpositions(pivots)
        reductions[position+1] = basis[position+1] ⊻ (one(TT) << position)
    end
    return pivots, reductions
end

# the XOR of the entries of `columns` at the positions of the set bits of `bits`, plus one
Base.@propagate_inbounds function _xorofbits(columns::Vector{TT}, bits::TT) where {TT}
    sum = zero(TT)
    limbs = _limbs(bits)
    for limb_index in eachindex(limbs)
        limb = limbs[limb_index]
        while limb != 0
            sum ⊻= columns[64*(limb_index-1)+trailing_zeros(limb)+1]
            limb &= limb - one(UInt64)
        end
    end
    return sum
end

# the positions of the set bits of `bits`, the lowest first, counted from 0
function _bitpositions(bits)
    positions = Int[]
    limbs = _limbs(bits)
    for limb_index in eachindex(limbs)
        limb = limbs[limb_index]
        while limb != 0
            push!(positions, 64 * (limb_index - 1) + trailing_zeros(limb))
            limb &= limb - one(UInt64)
        end
    end
    return positions
end

# The reduction of the basis vectors whose other bits are `reductions[position+1]` at their pivot bit `position`.
function _Reduction(reductions::Vector{TT}) where {TT}
    remainders = copy(reductions)
    positions = [position for position in eachindex(reductions) .- 1 if !iszero(reductions[position+1])]

    # the shifts that the most basis vectors share, each by at least two; the other bits of a basis vector lie above its
    # pivot bit, so every shift is at least one
    shift_counts = zeros(Int, length(reductions))
    for position in positions, bit in _bitpositions(reductions[position+1])
        shift_counts[bit-position] += 1
    end
    shared_shifts = [shift for shift in first(sortperm(shift_counts; rev=true), 3) if shift_counts[shift] >= 2]
    shifts = ntuple(index -> UInt(get(shared_shifts, index, 0)), Val(3))
    shifted_pivots = ntuple(index -> _movepivots!(remainders, positions, get(shared_shifts, index, 0)), Val(3))

    # the vector left that the most basis vectors share, by at least two
    sharing = Dict{TT,TT}()
    for position in positions
        if !iszero(remainders[position+1])
            sharing[remainders[position+1]] = get(sharing, remainders[position+1], zero(TT)) | (one(TT) << position)
        end
    end
    shared_vector, sharing_pivots = zero(TT), zero(TT)
    for (vector, pivots) in sharing
        if count_ones(pivots) >= max(2, count_ones(sharing_pivots) + 1)
            shared_vector, sharing_pivots = vector, pivots
        end
    end
    for position in _bitpositions(sharing_pivots)
        remainders[position+1] = zero(TT)
    end

    remaining_pivots = zero(TT)
    for position in positions
        if !iszero(remainders[position+1])
            remaining_pivots |= one(TT) << position
        end
    end
    return _Reduction{TT}(shifts, shifted_pivots, shared_vector, sharing_pivots, remaining_pivots, remainders)
end

# The pivot bits at `positions` whose remainder has the bit `shift` above them, which is taken out of the remainder. A
# shift of zero takes none.
function _movepivots!(remainders::Vector{TT}, positions, shift::Int) where {TT}
    moved = zero(TT)
    if shift == 0
        return moved
    end
    for position in positions
        bit = one(TT) << (position + shift)
        if !iszero(remainders[position+1] & bit)
            moved |= one(TT) << position
            remainders[position+1] ⊻= bit
        end
    end
    return moved
end

# the sum of the other bits of the basis vectors of the pivot bits `bits`
@inline function _reduce(reduction::_Reduction{TT}, bits::TT) where {TT}
    sum = zero(TT)
    for index in 1:3
        shifted = bits & reduction.shifted_pivots[index]
        if !iszero(reduction.shifted_pivots[index])
            sum ⊻= shifted << reduction.shifts[index]
        end
    end
    if !iszero(reduction.sharing_pivots)
        is_odd = isodd(count_ones(bits & reduction.sharing_pivots))
        sum ⊻= ifelse(is_odd, reduction.shared_vector, zero(TT))
    end
    remaining = bits & reduction.remaining_pivots
    if !iszero(remaining)
        sum ⊻= @inbounds _xorofbits(reduction.remainders, remaining)
    end
    return sum
end

"""
    _cosetrepresentative(plan, term)

The representative of the coset of `term` in the span of the gate masks of the plan: the term reduced by every basis
vector whose pivot bit it has.
A gate acting on a term flips a sum of basis vectors, so all terms of a coset have the same representative, and
terms with the same representative differ by a sum of gate masks, so the same gates act on them.
"""
@inline function _cosetrepresentative(plan::LayerPlan, term)
    set_pivots = term & plan.pivots
    return term ⊻ set_pivots ⊻ _reduce(plan.reduction, set_pivots)
end

# the label of a term, the hash of its coset representative, which picks its batch
@inline _cosetlabel(plan::LayerPlan, term) = _hashbits(_cosetrepresentative(plan, term)) % Int


### What a basis provides

"""
    _branchinggates(lookup, term)

The gates of the plan that act on `term`, as the bits of their positions in a tuple of words, the gate `i` of the plan
at the position `i - 1`, so that the bits are in the order in which the layer applies the gates.
Every lookup implements this. The terms of a coset give the same positions.
"""
function _branchinggates end


### Chunk tables

# A linear function of the bits of a term, the XOR of a column over its set bits, tabulated chunk by chunk: a table
# holds the function's value on every chunk of a term and each of its values, and the function of a term is the XOR of
# the entries of its chunks. The lookup of the gates acting on a term adds up the gates acting on its chunks this way.

# the bits of a term that a chunk table reads at once; the sums below hold for chunks of any size
const _CHUNK_BITS = 2

"""
    ChunkTable(columns, TT)

A function of the bits of a term of the type `TT` that is the XOR of `columns[bit+1]` over its set bits, as a table of
the function's value on every chunk of `_CHUNK_BITS` bits of a term and each of its values, so that the function of a
term is the XOR of the entries of its chunks.
"""
struct ChunkTable{TT,E}
    entries::Vector{E}

    function ChunkTable{TT,E}(entries::Vector{E}) where {TT,E}
        # `_xoroverchunks` reads the entry of every chunk value without bounds checks
        if length(entries) != (1 << _CHUNK_BITS) * div(8 * sizeof(TT), _CHUNK_BITS)
            throw(ArgumentError("a chunk table of terms of $(8 * sizeof(TT)) bits has $(length(entries)) entries"))
        end
        return new{TT,E}(entries)
    end
end

# the entry of every chunk value is that of the value without its lowest bit, plus the column of that bit
function ChunkTable(columns::Vector{E}, ::Type{TT}) where {E,TT}
    if length(columns) != 8 * sizeof(TT)
        throw(ArgumentError("a chunk table of terms of $(8 * sizeof(TT)) bits has $(length(columns)) columns"))
    end
    n_values = 1 << _CHUNK_BITS
    n_chunks = div(length(columns), _CHUNK_BITS)
    entries = Vector{E}(undef, n_values * n_chunks)
    for chunk in 0:n_chunks-1
        entries[n_values*chunk+1] = _zeroentry(E)
        for value in 1:n_values-1
            column = columns[_CHUNK_BITS*chunk+trailing_zeros(value)+1]
            entries[n_values*chunk+value+1] = _xorentries(entries[n_values*chunk+(value&(value-1))+1], column)
        end
    end
    return ChunkTable{TT,E}(entries)
end

# the entries of the chunks of `bits` that are not zero, added up
@inline function _xoroverchunks(table::ChunkTable{TT,E}, bits::TT) where {TT,E}
    sum = _zeroentry(E)
    chunk_mask = (one(UInt64) << _CHUNK_BITS) - one(UInt64)
    chunks_per_limb = div(64, _CHUNK_BITS)
    limbs = _limbs(bits)
    for limb_index in eachindex(limbs)
        limb = limbs[limb_index]
        occupied = _occupiedchunks(limb)
        while occupied != 0
            bit = trailing_zeros(occupied)
            occupied &= occupied - one(UInt64)
            value = (limb >> bit) & chunk_mask
            chunk = chunks_per_limb * (limb_index - 1) + div(bit, _CHUNK_BITS)
            sum = _xorentries(sum, @inbounds table.entries[(1<<_CHUNK_BITS)*chunk+(value%Int)+1])
        end
    end
    return sum
end

# the lowest bit of every chunk of `limb` in which a bit is set
@inline function _occupiedchunks(limb::UInt64)
    folded = limb
    for shift in 1:_CHUNK_BITS-1
        folded |= limb >> shift
    end
    return folded & (typemax(UInt64) ÷ ((one(UInt64) << _CHUNK_BITS) - one(UInt64)))
end

# the XOR of two entries and the entry that adds nothing, for integers and tuples of them; an entry of another type
# defines both
@inline _xorentries(a::T, b::T) where {T<:Integer} = a ⊻ b
@inline _xorentries(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T} = ntuple(index -> a[index] ⊻ b[index], Val(N))
_zeroentry(::Type{T}) where {T<:Integer} = zero(T)

_zeroentry(::Type{NTuple{N,T}}) where {N,T} = ntuple(_ -> zero(T), Val(N))


### Finding the gates acting on a term through a chunk table

"""
    PrecomputedLookup(acting_masks)

Finds the gates of a plan that act on a term, for a basis in which the gate `i` acts on a term if the bits that the
term and `acting_masks[i]` share are odd in number, so that it acts on a term if it acts on an odd number of its
chunks.
A chunk table holds, for every chunk of a term and each of its values, the positions of the gates acting on a term
with only those bits, and those of the chunks of a term add up with XOR.
"""
const PrecomputedLookup{TT,W} = ChunkTable{TT,NTuple{W,UInt64}}

function PrecomputedLookup(acting_masks::Vector{TT}) where {TT}
    return _precomputedlookup(acting_masks, Val(cld(length(acting_masks), 64)))
end

function _precomputedlookup(acting_masks::Vector{TT}, ::Val{W}) where {TT,W}
    # the gates acting on a term with only one bit: those whose acting mask has the bit
    columns = fill(_zeroentry(NTuple{W,UInt64}), 8 * sizeof(TT))
    for gate in eachindex(acting_masks)
        acting = _withposition(_zeroentry(NTuple{W,UInt64}), gate - 1)
        for bit in _bitpositions(acting_masks[gate])
            columns[bit+1] = _xorentries(columns[bit+1], acting)
        end
    end
    return ChunkTable(columns, TT)
end

# the gates acting on the chunks of the term added up
@inline _branchinggates(lookup::PrecomputedLookup{TT,W}, term::TT) where {TT,W} = _xoroverchunks(lookup, term)

# the positions with the bit of `position` set as well
@inline function _withposition(positions::NTuple{W,UInt64}, position::Int) where {W}
    word = (position >> 6) + 1
    return ntuple(w -> positions[w] | ifelse(w == word, one(UInt64) << (position & 63), zero(UInt64)), Val(W))
end


### A pass over the sum

# whether a layer is applied in one pass on the sum of the cache: its terms can be sorted by label in one pass, and its
# coefficients are of a type that the cosets are applied to with
_canapplylayer(prop_cache::AbstractPropagationCache) =
    _canapplytolabels(prop_cache) && _appliesincosets(coefftype(prop_cache))

# whether the cosets are applied to with coefficients of this type: numbers, and any other type that opts in
_appliesincosets(::Type{CT}) where {CT} = CT <: Number

"""
    _applypass!(prop_cache, plan, truncfunc; thread=true)

One pass of the gates of `plan` over the sum of the cache: the terms are sorted by the hash of their coset
representative, the terms of every label are split into their cosets, and every coset is loaded into the cache of the
`CosetScratch` of a task, where the gates acting on it are applied, truncating the terms for which `truncfunc` returns
`true` after every gate.
"""
function _applypass!(prop_cache::AbstractPropagationCache, plan::LayerPlan, truncfunc::F; thread::Bool=true) where {F}
    if length(prop_cache) == 0
        return prop_cache
    end
    TT = termtype(prop_cache)
    CT = coefftype(prop_cache)
    workspace = _takeworkspace(TT, CT, CosetScratch{TT,CT})
    try
        sources = _passsources(StorageType(prop_cache), prop_cache, workspace, thread)
        _applypass!(prop_cache, plan, truncfunc, workspace, sources, thread)
    finally
        _putbackworkspace!(workspace)
    end
    return prop_cache
end

# the pass with its workspace and the sources of its tasks given: the scratch of every task holds the cache that the
# cosets are loaded into, and the rules that the gates are applied with are those `_rulesof` gives
function _applypass!(prop_cache::AbstractPropagationCache, plan::LayerPlan, truncfunc::F, workspace, sources, thread::Bool) where {F}
    cosetlabel(term) = _cosetlabel(plan, term)
    applytolabel!(sink, scratch, label_terms, label_coeffs) =
        _applytocosets!(sink, scratch.cache, _rulesof(scratch, plan), plan, truncfunc, label_terms, label_coeffs)
    _applytolabels!(cosetlabel, applytolabel!, prop_cache, workspace, sources, thread)
    return prop_cache
end

# the rules that a task applies the gates with: those of the plan, unless its scratch holds rules of its own
_rulesof(scratch, plan::LayerPlan) = plan.rules

# The scratch of a task of a layer pass: a cache over a hashed sum, into which the task loads the cosets it applies the
# gates to, one after the other. The sum is on no number of sites in particular, since it holds the terms of any layer
# on its term type.
struct CosetScratch{TT,CT}
    cache::HashedPropagationCache{TT,CT}
end

CosetScratch{TT,CT}() where {TT,CT} = CosetScratch{TT,CT}(PropagationCache(HashedTermSum{TT,CT}(0)))


### The cosets of a label

# The terms `label_terms` share the hash of their coset representative: they are one coset, or, where two
# representatives share a hash, several. Every coset is moved to the front in turn and applied to: the first term and
# the terms with its representative.
function _applytocosets!(sink, cache, rules, plan, truncfunc::F, label_terms, label_coeffs) where {F}
    first = 1
    last = length(label_terms)
    while first <= last
        coset_end = first + 1
        if first < last
            representative = _cosetrepresentative(plan, label_terms[first])
            for index in first+1:last
                term = label_terms[index]
                if _cosetrepresentative(plan, term) == representative
                    if index != coset_end
                        coeff = label_coeffs[index]
                        label_terms[index] = label_terms[coset_end]
                        label_coeffs[index] = label_coeffs[coset_end]
                        label_terms[coset_end] = term
                        label_coeffs[coset_end] = coeff
                    end
                    coset_end += 1
                end
            end
        end
        coset = first:coset_end-1
        _applytocoset!(sink, cache, rules, plan, truncfunc, view(label_terms, coset), view(label_coeffs, coset))
        first = coset_end
    end
    return sink
end

"""
    _applytocoset!(sink, cache, rules, plan, truncfunc, coset_terms, coset_coeffs)

Applies the gates of the plan that act on the terms `coset_terms` with the coefficients `coset_coeffs`, one coset, as
`propagate!` applies gates to a sum: the terms are added to the emptied sum of `cache`, every acting gate `i` is
applied with `xorbranchmergeandtruncate!` and `rules[i]` in the order of the layer, and what is left in the sum is
written to `sink`. A coset that no gate acts on is merged and truncated by `truncfunc`.
"""
function _applytocoset!(sink, cache, rules, plan, truncfunc::F, coset_terms, coset_coeffs) where {F}
    positions = _branchinggates(plan.lookup, first(coset_terms))
    coset_sum = mainsum(cache)
    empty!(cache)
    # the terms of a coset differ in their pivot bits
    _distinguishby!(coset_sum, plan.pivots)
    # room for twice the terms, which loading the coset and branching it by its first gate need at most
    resize!(cache, 2 * length(coset_terms))
    for index in eachindex(coset_terms, coset_coeffs)
        add!(coset_sum, coset_terms[index], coset_coeffs[index])
    end

    if all(iszero, positions)
        mergeandtruncate!(truncfunc, cache; thread=false)
    end
    for word in eachindex(positions)
        bits = positions[word]
        while bits != 0
            gate = 64 * (word - 1) + trailing_zeros(bits) + 1
            bits &= bits - one(UInt64)
            xorbranchmergeandtruncate!(rules[gate], truncfunc, cache, plan.gate_masks[gate]; thread=false)
        end
    end

    for (term, coeff) in coset_sum
        _emit!(sink, term, coeff)
    end
    return sink
end
