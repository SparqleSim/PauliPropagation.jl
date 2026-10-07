###
##
# Layers of commuting rotations applied class by class. A rotation here is a gate that leaves a term unchanged or mixes
# it with its partner, the term with the bits of the rotation's mask flipped, by the cosine and sine of its angle and a
# sign for each of the two; a Pauli rotation is one. The rotations of a layer commute, so a term and every term they make
# from it are acted on by the same rotations. The terms that the same rotations act on and that agree on what those
# rotations leave unchanged form a class, which no rotation of the layer leaves. In one pass over the sum
# (`_applytogroups!`) the terms are grouped by the hash of the key of their class, and every class is rotated one rotation
# after the other in the order of the layer, with the truncations applied after each, as the rotations one by one do: a
# class of few key bits as a dense block (classblocks.jl), any other in a table of its own (classtables.jl).
#
# A basis provides, for every pass of a layer, a `ClassPlan` with the masks, angles and signs of the rotations and a
# reader that finds the rotations acting on a term (`_branchinggates`, through the byte tables of a `TableReader` unless
# the basis reads faster), the key of a term and the key bits of its class (`_classkey`, `_keybits`), and the signs of a
# pair (`_pairsigns`).
##
###

"""
    ClassPlan(masks, angles, signs, reader; min_abs_coeff=0)

The rotations of one pass of a layer, prepared for rotating the classes of terms: `masks[i]` are the bits that the
rotation `i` flips, `angles[i]` is its angle and `signs[i]` is what `_pairsigns` reads its signs from, and `reader` finds
the rotations that act on a term, as `_branchinggates` describes.
Coefficients below `min_abs_coeff` are truncated, so a rotation makes no partner whose coefficient cannot reach it.
A plan may hold rotations that are not in its pass, which the reader never finds.
"""
struct ClassPlan{TT,RT,ST,R}
    masks::Vector{TT}
    cosines::Vector{RT}
    sines::Vector{RT}
    min_coeffs_to_make::Vector{Float64}
    signs::Vector{ST}
    reader::R
end

function ClassPlan(masks::Vector{TT}, angles, signs::Vector{ST}, reader::R; min_abs_coeff::Real=0) where {TT,ST,R}
    sines = sin.(angles)
    min_coeffs_to_make = [Float64(_mincoefftomake(min_abs_coeff, sin_val)) for sin_val in sines]
    return ClassPlan{TT,eltype(sines),ST,R}(masks, cos.(angles), sines, min_coeffs_to_make, signs, reader)
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


### What a basis provides

"""
    _branchinggates(reader, term)

The rotations of the pass that act on `term`, as the bits of their positions in a tuple of words, the rotation `i` of
the plan at the position `i - 1`, so that the bits are in the order in which the layer applies the rotations, and what
those rotations touch: whatever `_classkey` and `_keybits` read the key of the term's class and its key bits from.
Every reader implements this. The terms of a class give the same positions and the same touched bits.
"""
function _branchinggates end

# What the rotations acting on `term` touch, which is all the label of a term needs. A reader that finds it without the
# positions does so here.
@inline _touchedbits(reader, term) = last(_branchinggates(reader, term))

"""
    _classkey(term, touched)
    _keybits(touched)

The key of the class of `term`, from the term and what the rotations acting on it touch, and the bits that tell the
terms of a class apart.
All terms of a class have the same key, and two terms have the same key only if the same rotations act on them. A basis
whose rotations flip the bits of other rotations of the layer makes sure of the latter by the passes it splits the layer
into.
The terms of a class differ only in its key bits, of which every rotation of the class flips a fixed set.
"""
function _classkey end
function _keybits end

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
    TableReader(pass, commutation_masks, touched, untouched)

A reader of the rotations `pass` of a plan through byte tables, for a basis in which a rotation acts on a term if the
bits that the term and the rotation's commutation mask `commutation_masks[i]` share are odd in number: the rotations
acting on a term are then the sum of those acting on each of its set bits.
For every byte of a term and each of its values, `tables` holds the positions of the rotations acting on a term with only
those bits. `touched[i]` is what the rotation `i` touches, which `_jointouched` joins from `untouched` over the rotations
acting on a term.
"""
struct TableReader{TT,W,TO}
    tables::Vector{NTuple{W,UInt64}}
    touched::Vector{TO}
    untouched::TO
end

function TableReader(pass, commutation_masks::Vector{TT}, touched::Vector{TO}, untouched::TO) where {TT,TO}
    return _tablereader(pass, commutation_masks, touched, untouched, Val(cld(length(commutation_masks), 64)))
end

function _tablereader(pass, commutation_masks::Vector{TT}, touched::Vector{TO}, untouched::TO, ::Val{W}) where {TT,TO,W}
    no_positions = ntuple(_ -> zero(UInt64), Val(W))
    n_bits = 8 * sizeof(TT)
    columns = fill(no_positions, n_bits)
    touched_at = fill(untouched, length(commutation_masks))

    for rotation in pass
        position = rotation - 1
        position_bits = _withposition(no_positions, position)
        limbs = _limbs(commutation_masks[rotation])
        for limb_index in eachindex(limbs)
            limb = limbs[limb_index]
            while limb != 0
                bit = 64 * (limb_index - 1) + trailing_zeros(limb)
                limb &= limb - one(UInt64)
                columns[bit+1] = _xorwords(columns[bit+1], position_bits)
            end
        end
        touched_at[position+1] = touched[rotation]
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
    return TableReader{TT,W,TO}(tables, touched_at, untouched)
end

# The positions from the entries of the bytes of the term that are not zero, and the touched bits from the positions.
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

    touched = reader.untouched
    for word in 1:W
        bits = positions[word]
        while bits != 0
            position = 64 * (word - 1) + trailing_zeros(bits)
            bits &= bits - one(UInt64)
            touched = _jointouched(touched, reader.touched[position+1])
        end
    end
    return positions, touched
end

# what two rotations touch together: masks are joined bit by bit, tuples of masks mask by mask
@inline _jointouched(touched, touched_here) = touched | touched_here
@inline _jointouched(touched::Tuple, touched_here::Tuple) = map(|, touched, touched_here)

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

# The label of the record of a term: the hash of the key of its class, which picks its partition and its group. A term
# that no rotation acts on is its own key.
@inline _classlabel(plan, term) = _hashbits(_classkey(term, _touchedbits(plan.reader, term))) % Int

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
# several. Every class is moved to the front in turn and rotated. Two terms have the same key only if the same rotations
# act on them (see `_classkey`), so the key of the first term, with what its rotations touch, finds its whole class.
function _rotategroup!(sink, scratch, plan, truncfunc::F, group_terms, group_coeffs) where {F}
    first = 1
    last = length(group_terms)
    while first <= last
        first_term = group_terms[first]
        positions, touched = _branchinggates(plan.reader, first_term)
        key = _classkey(first_term, touched)
        class_end = first
        for index in first:last
            term = group_terms[index]
            if _classkey(term, touched) == key
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
        _rotateclass!(sink, scratch, plan, truncfunc, view(group_terms, class), view(group_coeffs, class), positions, touched)
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
    _rotateclass!(sink, scratch, plan, truncfunc, class_terms, class_coeffs, positions, touched)

Rotates the class of the terms `class_terms` with the coefficients `class_coeffs`, one rotation after the other, and
writes what the truncations keep to `sink`: as a dense block if the class has few key bits, and in a table otherwise.
`positions` and `touched` are what `_branchinggates` reads from any term of the class.
"""
function _rotateclass!(sink, scratch, plan, truncfunc::F, class_terms, class_coeffs, positions, touched) where {F}
    rotations = view(scratch.rotations, 1:_rotationsinorder!(scratch.rotations, plan, positions))
    if isempty(rotations)
        _emitunrotated!(sink, truncfunc, class_terms, class_coeffs)
        return sink
    end
    key_bits = _keybits(touched)
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
