###
##
# Layers of commuting gates applied in one pass over the sum, class by class, through the gate path. A gate here leaves
# a term unchanged or mixes it with its partner, the term with the bits of the gate's mask flipped, by a rule of
# `xorbranch!`; a Pauli rotation is one. The gates of a layer commute: a term and the term with the mask of any gate of
# the layer flipped are acted on by the same gates. The terms that differ from each other by a sum of gate masks form a
# class, so the gates of the layer act on all terms of a class or on none of them, and mix only terms of one class. The
# plan holds the reduced echelon basis of the span of the masks, by which a term is reduced to the representative of its
# class. In one pass over the sum (`_applytoclasses!`) the terms are grouped by their representative, and every class
# is loaded into a small sum of its own, to which the gates acting on it are applied one after the other in the order
# of the layer as `propagate!` applies gates, through `xorbranchmergeandtruncate!` with the rule of each, with the
# truncations after each, as the gates one by one do.
#
# A basis provides, for a layer, a `LayerPlan` with the masks and rules of the gates and a lookup that finds the gates
# acting on a term (`_branchinggates`, a `PrecomputedLookup` built from the masks with which the basis decides whether a
# gate acts), and applies it with `_applypass!`. For gates of other kinds, a basis that knows classes the gates map into
# themselves applies the gates to every class through the gate path with `_applygatesbyclass!`.
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
    _classrepresentative(plan, term)

The representative of the class of `term`: the term reduced by every basis vector of the plan whose pivot bit it has.
A gate acting on a term flips a sum of basis vectors, so all terms of a class have the same representative, and
terms with the same representative differ by a sum of gate masks, so the same gates act on them.
"""
@inline function _classrepresentative(plan::LayerPlan, term)
    set_pivots = term & plan.pivots
    return term ⊻ set_pivots ⊻ _reduce(plan.reduction, set_pivots)
end


### What a basis provides

"""
    _branchinggates(lookup, term)

The gates of the plan that act on `term`, as the bits of their positions in a tuple of words, the gate `i` of the plan
at the position `i - 1`, so that the bits are in the order in which the layer applies the gates.
Every lookup implements this. The terms of a class give the same positions.
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

# whether a layer is applied in one pass on the sum of the cache: its terms can be grouped by class in one pass, and its
# coefficients are of a type that the classes are applied to with
_canapplylayer(prop_cache::AbstractPropagationCache) =
    _canapplytoclasses(prop_cache) && _appliesinclasses(coefftype(prop_cache))

# whether the classes are applied to with coefficients of this type: numbers, and any other type that opts in
_appliesinclasses(::Type{CT}) where {CT} = CT <: Number

"""
    _applypass!(prop_cache, plan, truncfunc; thread=true)

One pass of the gates of `plan` over the sum of the cache: the terms are grouped by their class representative, and
every class is loaded into the cache of the `ClassScratch` of a task, where the gates acting on it are applied,
truncating the terms for which `truncfunc` returns `true` after every gate.
"""
function _applypass!(prop_cache::AbstractPropagationCache, plan::LayerPlan, truncfunc::F; thread::Bool=true) where {F}
    if length(prop_cache) == 0
        return prop_cache
    end
    TT = termtype(prop_cache)
    CT = coefftype(prop_cache)
    workspace = _takeworkspace(TT, CT, ClassScratch{TT,CT})
    try
        sources = _passsources(StorageType(prop_cache), prop_cache, workspace, thread)
        _applypass!(prop_cache, plan, truncfunc, workspace, sources, thread)
    finally
        _putbackworkspace!(workspace)
    end
    return prop_cache
end

# the pass with its workspace and the sources of its tasks given: the scratch of every task holds the cache that the
# classes are loaded into, and the rules that the gates are applied with are those `_rulesof` gives
function _applypass!(prop_cache::AbstractPropagationCache, plan::LayerPlan, truncfunc::F, workspace, sources, thread::Bool) where {F}
    representativeof(term) = _classrepresentative(plan, term)
    applytoclass!(sink, scratch, class_terms, class_coeffs) =
        _applytoclass!(sink, scratch.cache, _rulesof(scratch, plan), plan, truncfunc, class_terms, class_coeffs)
    _applytoclasses!(representativeof, applytoclass!, prop_cache, workspace, sources, thread)
    return prop_cache
end

# the rules that a task applies the gates with: those of the plan, unless its scratch holds rules of its own
_rulesof(scratch, plan::LayerPlan) = plan.rules

# The scratch of a task of a layer pass: a cache over a hashed sum, into which the task loads the classes it applies the
# gates to, one after the other. The sum is on no number of sites in particular, since it holds the terms of any layer
# on its term type.
struct ClassScratch{TT,CT}
    cache::HashedPropagationCache{TT,CT}
end

ClassScratch{TT,CT}() where {TT,CT} = ClassScratch{TT,CT}(PropagationCache(HashedTermSum{TT,CT}(0)))

"""
    _applytoclass!(sink, cache, rules, plan, truncfunc, class_terms, class_coeffs)

Applies the gates of the plan that act on the terms `class_terms` with the coefficients `class_coeffs`, one class, as
`propagate!` applies gates to a sum: the terms are added to the emptied sum of `cache`, every acting gate `i` is
applied with `xorbranchmergeandtruncate!` and `rules[i]` in the order of the layer, and what is left in the sum is
written to `sink`. A class that no gate acts on is merged and truncated by `truncfunc`.
"""
function _applytoclass!(sink, cache, rules, plan, truncfunc::F, class_terms, class_coeffs) where {F}
    positions = _branchinggates(plan.lookup, first(class_terms))
    class_sum = mainsum(cache)
    empty!(cache)
    # the terms of a class differ in their pivot bits
    _distinguishby!(class_sum, plan.pivots)
    # room for twice the terms, which loading the class and branching it by its first gate need at most
    resize!(cache, 2 * length(class_terms))
    for index in eachindex(class_terms, class_coeffs)
        add!(class_sum, class_terms[index], class_coeffs[index])
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

    for (term, coeff) in class_sum
        _emit!(sink, term, coeff)
    end
    return sink
end


### A pass over the sum through the gate path

"""
    _applygatesbyclass!(applygates!, representativeof, prop_cache, ::Type{S}; thread=true)

One pass over the sum in which the terms of every class, by their representative `representativeof(term)`, are loaded
into the cache of the scratch of a task, of type `S` with a field `cache`, `applygates!(cache, class_terms)` applies the
gates acting on them as `propagate!` applies gates to a sum, and the terms left in the cache are written back.
The gates must map every class into itself.
"""
function _applygatesbyclass!(applygates!::A, representativeof::R, prop_cache::AbstractPropagationCache, ::Type{S};
    thread::Bool=true) where {A,R,S}

    if length(prop_cache) == 0
        return prop_cache
    end
    function applytoclass!(sink, scratch, class_terms, class_coeffs)
        cache = scratch.cache
        empty!(cache)
        resize!(cache, 2 * length(class_terms))
        for index in eachindex(class_terms, class_coeffs)
            add!(mainsum(cache), class_terms[index], class_coeffs[index])
        end
        applygates!(cache, class_terms)
        # a gate may leave its terms in the auxiliary sum and swap the two, so the terms are those of the main sum now
        for (term, coeff) in mainsum(cache)
            _emit!(sink, term, coeff)
        end
        return sink
    end

    workspace = _takeworkspace(termtype(prop_cache), coefftype(prop_cache), S)
    try
        sources = _passsources(StorageType(prop_cache), prop_cache, workspace, thread)
        _applytoclasses!(representativeof, applytoclass!, prop_cache, workspace, sources, thread)
    finally
        _putbackworkspace!(workspace)
    end
    return prop_cache
end
