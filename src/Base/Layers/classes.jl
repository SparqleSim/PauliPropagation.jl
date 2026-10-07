###
##
# Layers of commuting rotations applied class by class. A rotation here is a gate that leaves a term unchanged or mixes
# it with its partner, the term with the bits of the rotation's gate mask flipped, by the cosine and sine of its angle
# and a sign for each of the two; a Pauli rotation is one. The rotations of a layer commute: a term and the term with
# the gate mask of any rotation of the layer flipped are acted on by the same rotations. The plan holds the reduced
# echelon basis of the space that the gate masks span. A term reduced by the basis vectors whose pivot bits the
# rotations acting on it flip is the representative of its class, which no rotation of the layer leaves, and those pivot
# bits, the distinguishing bits of the class, tell its terms apart: gathered into an integer, the distinguishing bits of
# a term are its coordinates in the class, the basis vectors that make the term from the representative. In one pass
# over the sum (`_applytolabels!`) the terms are sorted by the hash of their class representative, and every class is
# rotated one rotation after the other in the order of the layer, with the truncations applied after each, as the
# rotations one by one do: a class of few distinguishing bits densely, any other sparsely (classkernels.jl).
#
# A basis provides, for a layer, a `ClassPlan` with the gate masks and angles of the rotations, a lookup that finds the
# rotations acting on a term and the bits they flip (`_branchinggates`, a `PrecomputedLookup` built from the masks with
# which the basis decides whether a rotation acts), and the function that gives the signs of a pair.
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
    ClassPlan(gate_masks, angles, lookup, pairsigns; min_abs_coeff=0)

The rotations of a layer, prepared for rotating the classes of terms: `gate_masks[i]` are the bits that the rotation `i`
flips and `angles[i]` is its angle, `lookup` finds the rotations that act on a term, as `_branchinggates` describes, and
`pairsigns(gate_mask, lower_term)` gives the signs that the rotation with the gate mask `gate_mask` gives the two terms
of a pair it mixes, from `lower_term`, the term of the pair in which the lowest distinguishing bit that the rotation
flips is clear: the sign of what the lower term contributes to its partner, and the sign of what the partner contributes
to the lower term.
Coefficients below `min_abs_coeff` are truncated, so a rotation makes no partner whose coefficient cannot reach it.
"""
struct ClassPlan{TT,RT,L,F}
    gate_masks::Vector{TT}
    cosines::Vector{RT}
    sines::Vector{RT}
    min_coeffs_to_make::Vector{Float64}
    lookup::L
    pairsigns::F

    # the reduced echelon basis of the gate masks: its pivot bits, and the other bits of the basis vectors of any pivot
    # bits
    pivots::TT
    reduction::_Reduction{TT}
end

function ClassPlan(gate_masks::Vector{TT}, angles, lookup::L, pairsigns::F; min_abs_coeff::Real=0) where {TT,L,F}
    sines = sin.(angles)
    min_coeffs_to_make = [Float64(_mincoefftomake(min_abs_coeff, sin_val)) for sin_val in sines]
    pivots, reductions = _echelonbasis(gate_masks)
    return ClassPlan{TT,eltype(sines),L,F}(gate_masks, cos.(angles), sines, min_coeffs_to_make, lookup, pairsigns, pivots,
        _Reduction(reductions))
end

# The smallest coefficient that can make a partner the truncation keeps, of a rotation with this sine. Without a smallest
# kept coefficient, any can.
function _mincoefftomake(min_abs_coeff::Real, sin_val)
    if iszero(min_abs_coeff)
        return zero(float(min_abs_coeff))
    end
    return _MAKE_MARGIN * min_abs_coeff / abs(sin_val)
end

# A coefficient of at most this fraction below the smallest kept one is checked against the truncation when the partner
# it would make is decided, so that rounding cannot hide a partner that the truncation keeps.
const _MAKE_MARGIN = 1 - 1e-12

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

    # the shifts that the most basis vectors share, each by at least two
    shift_counts = Dict{Int,Int}()
    for position in positions, bit in _bitpositions(reductions[position+1])
        shift_counts[bit-position] = get(shift_counts, bit - position, 0) + 1
    end
    shared_shifts = [shift for (shift, n_sharing) in sort(collect(shift_counts); by=((shift, n_sharing),) -> (-n_sharing, shift))
                     if n_sharing >= 2]
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
    _classrepresentative(plan, term, flipped)

The representative of the class of `term`, from the bits `flipped` that the rotations acting on it flip together: the
term reduced by every basis vector of the plan whose pivot bit is among those bits.
A rotation acting on a term flips a sum of those basis vectors, so all terms of a class have the same representative.
Terms with the same representative differ by a sum of gate masks of the layer, so the same rotations act on them, and
they are one class.
"""
@inline function _classrepresentative(plan::ClassPlan, term, flipped)
    flipped_pivots = term & plan.pivots & flipped
    return term ⊻ flipped_pivots ⊻ _reduce(plan.reduction, flipped_pivots)
end

"""
    _distinguishingbits(plan, flipped)

The bits that tell the terms of a class apart, from the bits `flipped` that the rotations acting on them flip: the pivot
bits among those. No two terms of a class agree on all of them, and every rotation of the class flips a fixed set of them.
"""
@inline _distinguishingbits(plan::ClassPlan, flipped) = plan.pivots & flipped


### What a basis provides

"""
    _branchinggates(lookup, term)

The rotations of the plan that act on `term`, as the bits of their positions in a tuple of words, the rotation `i` of
the plan at the position `i - 1`, so that the bits are in the order in which the layer applies the rotations, and the
bits that those rotations flip together, the union of their gate masks.
Every lookup implements this. The terms of a class give the same positions and the same bits.
"""
function _branchinggates end

# the bits that the rotations acting on `term` flip together, which is all that the label of a term needs
@inline _flippedbits(lookup, term) = last(_branchinggates(lookup, term))

# What two terms that a rotation mixes contribute to the gradient of that rotation, which only the sink of a gradient
# pass adds up.
@inline _addgradient!(sink, rotation::Int, coeff, partner_coeff, sign_to_partner, sign_from_partner) = nothing


### Chunk tables

# A linear function of the bits of a term, the XOR of a column over its set bits, tabulated chunk by chunk: a table
# holds the function's value on every chunk of a term and each of its values, and the function of a term is the XOR of
# the entries of its chunks. The lookup of the rotations acting on a term adds up the rotations acting on its chunks
# this way.

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


### Finding the rotations acting on a term through a chunk table

struct _ActingRotations{W,D,TT}
    positions::NTuple{W,UInt64}
    mask_parts::NTuple{D,TT}
end

# the parts of a layer without rotations are an empty tuple, which does not name their type
@inline _xorentries(a::_ActingRotations{W,D,TT}, b::_ActingRotations{W,D,TT}) where {W,D,TT} =
    _ActingRotations{W,D,TT}(_xorentries(a.positions, b.positions), _xorentries(a.mask_parts, b.mask_parts))
_zeroentry(::Type{_ActingRotations{W,D,TT}}) where {W,D,TT} =
    _ActingRotations{W,D,TT}(ntuple(_ -> zero(UInt64), Val(W)), ntuple(_ -> zero(TT), Val(D)))

"""
    PrecomputedLookup(acting_masks, gate_masks)

Finds the rotations of a plan that act on a term, for a basis in which the rotation `i` acts on a term if the bits that
the term and `acting_masks[i]` share are odd in number, so that it acts on a term if it acts on an odd number of its
chunks.
A chunk table holds, for every chunk of a term and each of its values, the rotations acting on a term with only those
bits, and those of the chunks of a term add up with XOR: their positions, and the bits they flip, `gate_masks[i]` for
the rotation `i`, in as many parts as rotations flip one bit, so that the parts of two rotations never share a bit and
the bits that the acting rotations flip together are the union of the parts.
"""
const PrecomputedLookup{TT,W,D} = ChunkTable{TT,_ActingRotations{W,D,TT}}

function PrecomputedLookup(acting_masks::Vector{TT}, gate_masks::Vector{TT}) where {TT}
    n_parts = maximum(_flippingcounts(gate_masks); init=0)
    return _precomputedlookup(acting_masks, gate_masks, Val(cld(length(gate_masks), 64)), Val(n_parts))
end

function _precomputedlookup(acting_masks::Vector{TT}, gate_masks::Vector{TT}, ::Val{W}, ::Val{D}) where {TT,W,D}
    mask_parts = _maskparts(gate_masks, Val(D))

    # the rotations acting on a term with only one bit: those whose acting mask has the bit
    columns = fill(_zeroentry(_ActingRotations{W,D,TT}), 8 * sizeof(TT))
    for rotation in eachindex(acting_masks)
        acting = _ActingRotations{W,D,TT}(_withposition(ntuple(_ -> zero(UInt64), Val(W)), rotation - 1), mask_parts[rotation])
        for bit in _bitpositions(acting_masks[rotation])
            columns[bit+1] = _xorentries(columns[bit+1], acting)
        end
    end
    return ChunkTable(columns, TT)
end

# for every bit, the number of gate masks that have it
function _flippingcounts(gate_masks::Vector{TT}) where {TT}
    counts = zeros(Int, 8 * sizeof(TT))
    for gate_mask in gate_masks, bit in _bitpositions(gate_mask)
        counts[bit+1] += 1
    end
    return counts
end

# The gate masks in `D` parts, where no two masks have the same bit in one part: a bit of a mask goes into the part
# numbered by how many of the masks before it have the bit.
function _maskparts(gate_masks::Vector{TT}, ::Val{D}) where {TT,D}
    counts = zeros(Int, 8 * sizeof(TT))
    parts = Vector{NTuple{D,TT}}(undef, length(gate_masks))
    for rotation in eachindex(gate_masks)
        mask_parts = ntuple(_ -> zero(TT), Val(D))
        for bit in _bitpositions(gate_masks[rotation])
            part = counts[bit+1] + 1
            counts[bit+1] = part
            mask_parts = ntuple(index -> ifelse(index == part, mask_parts[index] | (one(TT) << bit), mask_parts[index]), Val(D))
        end
        parts[rotation] = mask_parts
    end
    return parts
end

# the rotations acting on the chunks of the term added up, with the flipped bits as the union of the parts
@inline function _branchinggates(lookup::PrecomputedLookup{TT,W,D}, term::TT) where {TT,W,D}
    acting = _xoroverchunks(lookup, term)
    flipped = zero(TT)
    for part in acting.mask_parts
        flipped |= part
    end
    return acting.positions, flipped
end

# the positions with the bit of `position` set as well
@inline function _withposition(positions::NTuple{W,UInt64}, position::Int) where {W}
    word = (position >> 6) + 1
    return ntuple(w -> positions[w] | ifelse(w == word, one(UInt64) << (position & 63), zero(UInt64)), Val(W))
end


### A pass over the sum

# whether the classes of a layer can be rotated in the sum of the cache: its terms can be sorted by label in one pass, and its
# coefficients are of a type that the classes are rotated with
_canrotateclasses(prop_cache::AbstractPropagationCache) =
    _canapplytolabels(prop_cache) && _rotatesinclasses(coefftype(prop_cache))

# whether the classes are rotated with coefficients of this type: numbers, and any other type that opts in
_rotatesinclasses(::Type{CT}) where {CT} = CT <: Number

# The label of a term, the hash of its class representative, which picks its batch. A term
# that no rotation acts on is its own representative.
@inline _classlabel(plan, term) = _hashbits(_classrepresentative(plan, term, _flippedbits(plan.lookup, term))) % Int

"""
    _applypass!(prop_cache, plan, truncfunc, workspace; thread=true)

One pass of the rotations of `plan` over the sum of the cache: the terms are sorted by the hash of their class
representative, the terms of every label are split into their classes, and every class is rotated, truncating the terms for which `truncfunc`
returns `true` after every rotation.
"""
function _applypass!(prop_cache::AbstractPropagationCache, plan::ClassPlan, truncfunc::F, workspace; thread::Bool=true) where {F}
    if length(prop_cache) == 0
        return prop_cache
    end
    sources = _passsources(StorageType(prop_cache), prop_cache, workspace, thread)
    _applypass!(prop_cache, plan, truncfunc, workspace, sources, thread)
    return prop_cache
end

function _applypass!(prop_cache::AbstractPropagationCache, plan::ClassPlan, truncfunc::F, workspace, sources, thread::Bool) where {F}
    classlabel(term) = _classlabel(plan, term)
    rotateclasses!(sink, scratch, label_terms, label_coeffs) = _rotateclasses!(sink, scratch, plan, truncfunc, label_terms, label_coeffs)
    _applytolabels!(classlabel, rotateclasses!, prop_cache, workspace, sources, thread)
    return prop_cache
end


### Rotating the classes of a label

# The terms `label_terms` share the hash of their class representative: they are one class, or, where two
# representatives share a hash, several. Every class is moved to the front in turn and rotated: the first term and the
# terms with its representative, read with the bits that the rotations acting on the first term flip.
function _rotateclasses!(sink, scratch, plan, truncfunc::F, label_terms, label_coeffs) where {F}
    first = 1
    last = length(label_terms)
    while first <= last
        first_term = label_terms[first]
        positions, flipped = _branchinggates(plan.lookup, first_term)
        # the last term of a label is a class of its own
        representative = if first < last
            _classrepresentative(plan, first_term, flipped)
        else
            first_term
        end
        class_end = first + 1
        for index in first+1:last
            term = label_terms[index]
            if _classrepresentative(plan, term, flipped) == representative
                if index != class_end
                    coeff = label_coeffs[index]
                    label_terms[index] = label_terms[class_end]
                    label_coeffs[index] = label_coeffs[class_end]
                    label_terms[class_end] = term
                    label_coeffs[class_end] = coeff
                end
                class_end += 1
            end
        end
        class = first:class_end-1
        _rotateclass!(sink, scratch, plan, truncfunc, view(label_terms, class), view(label_coeffs, class), positions, flipped)
        first = class_end
    end
    return sink
end

# the rotations of the class of a term, those that act on it, in the order of the layer, and their number
@inline function _rotationsinorder!(rotations::Vector{Int32}, plan, positions::NTuple{W,UInt64}) where {W}
    _ensurecapacity!(rotations, length(plan.gate_masks))
    n_found = 0
    for word in 1:W
        bits = positions[word]
        while bits != 0
            position = 64 * (word - 1) + trailing_zeros(bits)
            bits &= bits - one(UInt64)
            n_found += 1
            @inbounds rotations[n_found] = (position + 1) % Int32
        end
    end
    return n_found
end

# A class that no rotation acts on is one term, given once or more often, and is kept if the truncation keeps it.
function _emitunrotated!(sink, truncfunc::F, class_terms, class_coeffs) where {F}
    coeff = class_coeffs[1]
    for index in 2:length(class_coeffs)
        coeff = mergefunc(coeff, class_coeffs[index])
    end
    if !@inline(truncfunc(class_terms[1], coeff))
        _emit!(sink, class_terms[1], coeff)
    end
    return sink
end

# the most distinguishing bits of a class that is rotated densely, whose presence is one word
const _MAX_DENSE_DISTINGUISHING_BITS = 6

# whether a class of `n_distinguishing_bits` distinguishing bits is rotated densely, rather than sparsely
_rotatesdense(n_distinguishing_bits::Int) = n_distinguishing_bits <= _MAX_DENSE_DISTINGUISHING_BITS

# the most distinguishing bits whose coordinates fit an integer
const _MAX_COORDINATE_BITS = 64

"""
    _rotateclass!(sink, scratch, plan, truncfunc, class_terms, class_coeffs, positions, flipped)

Rotates the class of the terms `class_terms` with the coefficients `class_coeffs`, one rotation after the other, and
writes what the truncations keep to `sink`: densely if the class has few distinguishing bits, and sparsely otherwise.
`positions` and `flipped` are what `_branchinggates` reads from any term of the class.
"""
function _rotateclass!(sink, scratch, plan, truncfunc::F, class_terms, class_coeffs, positions, flipped) where {F}
    rotations = view(scratch.rotations, 1:_rotationsinorder!(scratch.rotations, plan, positions))
    if isempty(rotations)
        _emitunrotated!(sink, truncfunc, class_terms, class_coeffs)
        return sink
    end
    distinguishing_bits = _distinguishingbits(plan, flipped)
    n_distinguishing_bits = sum(count_ones, _limbs(distinguishing_bits))

    # the kernels are compiled for entries found by coordinates and by terms, behind this barrier
    if _rotatesdense(n_distinguishing_bits)
        _rotatedense!(sink, scratch.dense, plan, truncfunc, class_terms, class_coeffs, rotations, distinguishing_bits, n_distinguishing_bits)
    elseif n_distinguishing_bits <= _MAX_COORDINATE_BITS
        _rotatesparse!(sink, scratch.sparse, plan, truncfunc, class_terms, class_coeffs, rotations, scratch.sparse.entry_coordinates, distinguishing_bits, n_distinguishing_bits)
    else
        _rotatesparse!(sink, scratch.sparse, plan, truncfunc, class_terms, class_coeffs, rotations, scratch.sparse.entry_terms, nothing, n_distinguishing_bits)
    end
    return sink
end
