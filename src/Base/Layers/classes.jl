###
##
# Layers of commuting rotations applied class by class. A rotation here is a gate that leaves a term unchanged or mixes
# it with its partner, the term with the bits of the rotation's mask flipped, by the cosine and sine of its angle and a
# sign for each of the two; a Pauli rotation is one. The rotations of a layer commute: a term and the term with the mask
# of any rotation of the layer flipped are acted on by the same rotations. The plan holds the reduced echelon basis of the
# space that the masks span. A term reduced by the basis vectors whose pivot bits the rotations acting on it flip is the
# key of its class, which no rotation of the layer leaves, and those pivot bits tell the terms of a class apart. In one
# pass over the sum (`_applytogroups!`) the terms are grouped by the hash of the key of their class, and every class is
# rotated one rotation after the other in the order of the layer, with the truncations applied after each, as the
# rotations one by one do: a class of few key bits as a dense block (classblocks.jl), any other in a table of its own
# (classtables.jl).
#
# A basis provides, for a layer, a `ClassPlan` with the masks, angles and signs of the rotations and a reader that finds
# the rotations acting on a term and the bits they flip (`_branchinggates`, through the byte tables of a `TableReader`
# unless the basis reads faster), and the signs of a pair (`_pairsigns`).
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
    ClassPlan(masks, angles, signs, reader; min_abs_coeff=0)

The rotations of a layer, prepared for rotating the classes of terms: `masks[i]` are the bits that the rotation `i`
flips, `angles[i]` is its angle and `signs[i]` is what `_pairsigns` reads its signs from, and `reader` finds the
rotations that act on a term, as `_branchinggates` describes.
Coefficients below `min_abs_coeff` are truncated, so a rotation makes no partner whose coefficient cannot reach it.
"""
struct ClassPlan{TT,RT,ST,R}
    masks::Vector{TT}
    cosines::Vector{RT}
    sines::Vector{RT}
    min_coeffs_to_make::Vector{Float64}
    signs::Vector{ST}
    reader::R

    # the reduced echelon basis of the masks: its pivot bits, and the other bits of the basis vectors of any pivot bits
    pivots::TT
    reduction::_Reduction{TT}
end

function ClassPlan(masks::Vector{TT}, angles, signs::Vector{ST}, reader::R; min_abs_coeff::Real=0) where {TT,ST,R}
    sines = sin.(angles)
    min_coeffs_to_make = [Float64(_mincoefftomake(min_abs_coeff, sin_val)) for sin_val in sines]
    pivots, reductions = _echelonbasis(masks)
    return ClassPlan{TT,eltype(sines),ST,R}(masks, cos.(angles), sines, min_coeffs_to_make, signs, reader, pivots,
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

# The reduced echelon basis of the space that `masks` span: its pivot bits, and at the position of every pivot bit, plus
# one, the other bits of the basis vector with that pivot. The pivot of a basis vector is its lowest bit, which no other
# basis vector has. A mask that adds nothing to the masks before it adds no basis vector.
function _echelonbasis(masks::Vector{TT}) where {TT}
    basis = zeros(TT, 8 * sizeof(TT))
    pivots = zero(TT)
    for mask in masks
        # the basis vectors of the pivot bits of the mask clear those bits and set no other pivot bit
        vector = mask ⊻ _xorofbits(basis, mask & pivots)
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

# the XOR of the entries of `table` at the positions of the set bits of `bits`, plus one
Base.@propagate_inbounds function _xorofbits(table::Vector{TT}, bits::TT) where {TT}
    sum = zero(TT)
    limbs = _limbs(bits)
    for limb_index in eachindex(limbs)
        limb = limbs[limb_index]
        while limb != 0
            sum ⊻= table[64*(limb_index-1)+trailing_zeros(limb)+1]
            limb &= limb - one(UInt64)
        end
    end
    return sum
end

# the positions of the set bits of `bits`, the lowest first
_bitpositions(bits) = [position for position in 0:8*sizeof(bits)-1 if !iszero(bits & (one(bits) << position))]

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
    _classkey(plan, term, flipped)

The key of the class of `term`, from the bits `flipped` that the rotations acting on it flip together: the term reduced
by every basis vector of the plan whose pivot bit is among those bits.
A rotation acting on a term flips a sum of those basis vectors, so all terms of a class have the same key. Terms with the
same key differ by a sum of masks of the layer, so the same rotations act on them, and they are one class.
"""
@inline function _classkey(plan::ClassPlan, term, flipped)
    flipped_pivots = term & plan.pivots & flipped
    return term ⊻ flipped_pivots ⊻ _reduce(plan.reduction, flipped_pivots)
end

"""
    _keybits(plan, flipped)

The bits that tell the terms of a class apart, from the bits `flipped` that the rotations acting on them flip: the pivot
bits among those. No two terms of a class agree on all of them, and every rotation of the class flips a fixed set of them.
"""
@inline _keybits(plan::ClassPlan, flipped) = plan.pivots & flipped


### What a basis provides

"""
    _branchinggates(reader, term)

The rotations of the plan that act on `term`, as the bits of their positions in a tuple of words, the rotation `i` of
the plan at the position `i - 1`, so that the bits are in the order in which the layer applies the rotations, and the
bits that those rotations flip together, the union of their masks.
Every reader implements this. The terms of a class give the same positions and the same bits.
"""
function _branchinggates end

# The bits that the rotations acting on `term` flip together, which is all that the label of a term needs. A reader that
# finds them without the positions does so here.
@inline _flippedbits(reader, term) = last(_branchinggates(reader, term))

"""
    _pairsigns(signs, lower_term)

The signs that a rotation gives the two terms of a pair it mixes, read from the `signs` of the rotation in the plan for
`lower_term`, the term of the pair in which the lowest key bit of the rotation is clear: the sign of what the lower term
contributes to its partner, and the sign of what the partner contributes to the lower term.
Signs that are the same for every pair are a tuple of the two.
"""
@inline _pairsigns(signs::Tuple, lower_term) = signs

# What two terms that a rotation mixes contribute to the gradient of that rotation, which only the sink of a gradient
# pass adds up.
@inline _addgradient!(sink, rotation::Int, coeff, partner_coeff, sign_to_partner, sign_from_partner) = nothing


### Reading the rotations acting on a term through byte tables

"""
    TableReader(commutation_masks, masks)

A reader of the rotations of a plan through byte tables, for a basis in which a rotation acts on a term if the bits that
the term and the rotation's commutation mask `commutation_masks[i]` share are odd in number: the rotations acting on a
term are then the sum of those acting on each of its set bits.
For every byte of a term and each of its values, `tables` holds the positions of the rotations acting on a term with only
those bits. `masks[i]` are the bits that the rotation `i` flips.
"""
struct TableReader{TT,W}
    tables::Vector{NTuple{W,UInt64}}
    masks::Vector{TT}
end

function TableReader(commutation_masks::Vector{TT}, masks::Vector{TT}) where {TT}
    return _tablereader(commutation_masks, masks, Val(cld(length(commutation_masks), 64)))
end

function _tablereader(commutation_masks::Vector{TT}, masks::Vector{TT}, ::Val{W}) where {TT,W}
    no_positions = ntuple(_ -> zero(UInt64), Val(W))
    n_bits = 8 * sizeof(TT)
    columns = fill(no_positions, n_bits)

    for rotation in eachindex(commutation_masks)
        position_bits = _withposition(no_positions, rotation - 1)
        limbs = _limbs(commutation_masks[rotation])
        for limb_index in eachindex(limbs)
            limb = limbs[limb_index]
            while limb != 0
                bit = 64 * (limb_index - 1) + trailing_zeros(limb)
                limb &= limb - one(UInt64)
                columns[bit+1] = _xorwords(columns[bit+1], position_bits)
            end
        end
    end

    # the entry of every value of a byte is that of the value without its lowest bit, plus the column of that bit
    n_bytes = sizeof(TT)
    tables = Vector{NTuple{W,UInt64}}(undef, 256 * n_bytes)
    for byte in 0:n_bytes-1
        tables[256*byte+1] = no_positions
        for value in 1:255
            column = columns[8*byte+trailing_zeros(value)+1]
            tables[256*byte+value+1] = _xorwords(tables[256*byte+(value&(value-1))+1], column)
        end
    end
    return TableReader{TT,W}(tables, masks)
end

# The positions from the entries of the bytes of the term that are not zero, and the flipped bits from the positions.
# Kept out of line, so that the readers of a basis that reads faster stay small where they are inlined.
@noinline function _branchinggates(reader::TableReader{TT,W}, term::TT) where {TT,W}
    positions = ntuple(_ -> zero(UInt64), Val(W))
    limbs = _limbs(term)
    for limb_index in eachindex(limbs)
        limb = limbs[limb_index]
        while limb != 0
            byte_in_limb = trailing_zeros(limb) >> 3
            value = (limb >> (8 * byte_in_limb)) & 0xff
            limb &= ~(UInt64(0xff) << (8 * byte_in_limb))
            positions = _xorwords(positions, reader.tables[256*(8*(limb_index-1)+byte_in_limb)+(value%Int)+1])
        end
    end

    flipped = zero(TT)
    for word in 1:W
        bits = positions[word]
        while bits != 0
            position = 64 * (word - 1) + trailing_zeros(bits)
            bits &= bits - one(UInt64)
            flipped |= reader.masks[position+1]
        end
    end
    return positions, flipped
end

@inline _xorwords(a::NTuple{W,UInt64}, b::NTuple{W,UInt64}) where {W} = ntuple(word -> a[word] ⊻ b[word], Val(W))

# the positions with the bit of `position` set as well
@inline function _withposition(positions::NTuple{W,UInt64}, position::Int) where {W}
    word = (position >> 6) + 1
    return ntuple(w -> positions[w] | ifelse(w == word, one(UInt64) << (position & 63), zero(UInt64)), Val(W))
end


### A pass over the sum

# whether the classes of a layer can be rotated in the sum of the cache: its terms can be grouped in one pass, and its
# coefficients are of a type that the classes are rotated with
_canrotateclasses(prop_cache::AbstractPropagationCache) =
    _canapplytogroups(prop_cache) && _rotatesinclasses(coefftype(prop_cache))

# whether the classes are rotated with coefficients of this type: numbers, and any other type that opts in
_rotatesinclasses(::Type{CT}) where {CT} = CT <: Number

# The label of the record of a term, the hash of the key of its class, which picks its partition and its group. A term
# that no rotation acts on is its own key.
@inline _classlabel(plan, term) = _hashbits(_classkey(plan, term, _flippedbits(plan.reader, term))) % Int

"""
    _applypass!(prop_cache, plan, truncfunc, workspace; thread=true)

One pass of the rotations of `plan` over the sum of the cache: the terms are grouped by the hash of the key of their
class, every group is split into its classes, and every class is rotated, truncating the terms for which `truncfunc`
returns `true` after every rotation.
"""
function _applypass!(prop_cache::AbstractPropagationCache, plan::ClassPlan, truncfunc::F, workspace; thread::Bool=true) where {F}
    if length(prop_cache) == 0
        return prop_cache
    end
    sources = _recordsources(StorageType(prop_cache), prop_cache, workspace, thread)
    _applypass!(prop_cache, plan, truncfunc, workspace, sources, thread)
    return prop_cache
end

function _applypass!(prop_cache::AbstractPropagationCache, plan::ClassPlan, truncfunc::F, workspace, sources, thread::Bool) where {F}
    classlabel(term) = _classlabel(plan, term)
    rotategroup!(sink, scratch, group_terms, group_coeffs) = _rotategroup!(sink, scratch, plan, truncfunc, group_terms, group_coeffs)
    _applytogroups!(classlabel, rotategroup!, prop_cache, workspace, sources, thread)
    return prop_cache
end


### Rotating the classes of a group

# The terms `group_terms` share the hash of the key of their class: they are one class, or, where two keys share a hash,
# several. Every class is moved to the front in turn and rotated: the first term and the terms with its key, read with
# the bits that the rotations acting on the first term flip.
function _rotategroup!(sink, scratch, plan, truncfunc::F, group_terms, group_coeffs) where {F}
    first = 1
    last = length(group_terms)
    while first <= last
        first_term = group_terms[first]
        positions, flipped = _branchinggates(plan.reader, first_term)
        # the last term of a group is a class of its own
        key = if first < last
            _classkey(plan, first_term, flipped)
        else
            first_term
        end
        class_end = first + 1
        for index in first+1:last
            term = group_terms[index]
            if _classkey(plan, term, flipped) == key
                if index != class_end
                    coeff = group_coeffs[index]
                    group_terms[index] = group_terms[class_end]
                    group_coeffs[index] = group_coeffs[class_end]
                    group_terms[class_end] = term
                    group_coeffs[class_end] = coeff
                end
                class_end += 1
            end
        end
        class = first:class_end-1
        _rotateclass!(sink, scratch, plan, truncfunc, view(group_terms, class), view(group_coeffs, class), positions, flipped)
        first = class_end
    end
    return sink
end

# the rotations of the class of a term, those that act on it, in the order of the layer, and their number
@inline function _rotationsinorder!(rotations::Vector{Int32}, plan, positions::NTuple{W,UInt64}) where {W}
    _ensurecapacity!(rotations, length(plan.masks))
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

# the most key bits of a class that is rotated as a dense block
const _MAX_BLOCK_KEY_BITS = 8

# whether a class of `n_key_bits` key bits is rotated as a dense block, rather than in a table
_rotatesasblock(n_key_bits::Int) = n_key_bits <= _MAX_BLOCK_KEY_BITS

# the most bits that a key gathered into an integer can have
const _MAX_KEY_BITS = 64

"""
    _rotateclass!(sink, scratch, plan, truncfunc, class_terms, class_coeffs, positions, flipped)

Rotates the class of the terms `class_terms` with the coefficients `class_coeffs`, one rotation after the other, and
writes what the truncations keep to `sink`: as a dense block if the class has few key bits, and in a table otherwise.
`positions` and `flipped` are what `_branchinggates` reads from any term of the class.
"""
function _rotateclass!(sink, scratch, plan, truncfunc::F, class_terms, class_coeffs, positions, flipped) where {F}
    rotations = view(scratch.rotations, 1:_rotationsinorder!(scratch.rotations, plan, positions))
    if isempty(rotations)
        _emitunrotated!(sink, truncfunc, class_terms, class_coeffs)
        return sink
    end
    key_bits = _keybits(plan, flipped)
    n_key_bits = sum(count_ones, _limbs(key_bits))

    # the kernels are compiled for keys of each type, behind this barrier
    if _rotatesasblock(n_key_bits)
        _rotateblock!(sink, scratch.block, plan, truncfunc, class_terms, class_coeffs, rotations, key_bits, n_key_bits)
    elseif n_key_bits <= _MAX_KEY_BITS
        _rotatetable!(sink, scratch.table, plan, truncfunc, class_terms, class_coeffs, rotations, scratch.table.entry_keys, key_bits, n_key_bits)
    else
        _rotatetable!(sink, scratch.table, plan, truncfunc, class_terms, class_coeffs, rotations, scratch.table.entry_terms, nothing, n_key_bits)
    end
    return sink
end
