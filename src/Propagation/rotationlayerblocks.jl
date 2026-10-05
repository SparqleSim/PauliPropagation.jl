###
##
# A class of few key bits rotated as a dense block. A class of n key bits has 2^n Pauli strings, and the block holds an
# entry for each, at the key bits of the string gathered into an integer. A rotation flips a fixed set of key bits, so it
# pairs every entry with the entry whose index differs by the rotation's key mask, and mixes the coefficients of every
# pair with at least one entry present. Which entries are present is kept as bits, 64 to a word, so that a rotation
# visits the words of the block and, in each, the pairs with an entry present.
##
###

# for the key bit b below 6, the positions in a word of the entries without bit b
const _LOWER_POSITIONS = (0x5555555555555555, 0x3333333333333333, 0x0f0f0f0f0f0f0f0f, 0x00ff00ff00ff00ff, 0x0000ffff0000ffff, 0x00000000ffffffff)

# the entry of `pstr` in the block of its class: its key bits gathered into an integer
@inline _blockentry(pstr, key_bits) = _keyof(pstr, key_bits) % Int

# Moves the bit at every position p of a word to p ⊻ low_mask, for low_mask below 64, one set bit of low_mask at a time:
# a rotation flips the key bits of the qubits it acts on, so a layer takes the same few steps for every word.
@inline function _xorpositions(word::UInt64, low_mask::Int)
    bits = low_mask & 63
    while bits != 0
        j = trailing_zeros(bits)
        bits &= bits - 1
        span = 1 << j
        lower = _LOWER_POSITIONS[j+1]
        word = ((word >> span) & lower) | ((word & lower) << span)
    end
    return word
end

# The block of a class: a coefficient and a Pauli string for every entry, and which entries are present when they take
# more than one word.
struct BlockScratch{TT,CT}
    coeffs::Vector{CT}
    terms::Vector{TT}
    words::Vector{UInt64}
end

BlockScratch{TT,CT}() where {TT,CT} = BlockScratch{TT,CT}(CT[], TT[], UInt64[])

"""
    _rotateblock!(sink, block, plan, truncfunc, class_terms, class_coeffs, rotations, key_bits, n_key_bits)

Rotates the class of the Pauli strings `class_terms` with the coefficients `class_coeffs` as a dense block of
`2^n_key_bits` entries, by the `rotations` one after the other, and writes what the truncations keep to `sink`.
"""
function _rotateblock!(sink, block, plan, truncfunc::F, class_terms, class_coeffs, rotations, key_bits::TT, n_key_bits::Int) where {F,TT}
    n_entries = 1 << n_key_bits
    coeffs = PropagationBase._ensurecapacity!(block.coeffs, n_entries)
    block_terms = PropagationBase._ensurecapacity!(block.terms, n_entries)

    # a block of at most 64 entries keeps which are present in one word, any other in an array of words
    if n_entries <= 64
        present = zero(UInt64)
        for index in eachindex(class_terms, class_coeffs)
            pstr = class_terms[index]
            entry = _blockentry(pstr, key_bits)
            bit = one(UInt64) << entry
            if present & bit != 0
                coeffs[entry+1] = mergefunc(coeffs[entry+1], class_coeffs[index])
            else
                coeffs[entry+1] = class_coeffs[index]
                block_terms[entry+1] = pstr
                present |= bit
            end
        end
        for step in eachindex(rotations)
            rotation = Int(rotations[step])
            entry_mask = _blockentry(plan.masks[rotation], key_bits)
            signs = _blocksigns(plan)
            present = _rotateblockword(present, coeffs, block_terms, n_entries, entry_mask, plan, rotation, signs, truncfunc)
        end
        _emitblockwords!(sink, (present,), 1, coeffs, block_terms)
    else
        n_words = n_entries >> 6
        present = PropagationBase._ensurecapacity!(block.words, n_words)
        for word in 1:n_words
            present[word] = zero(UInt64)
        end
        for index in eachindex(class_terms, class_coeffs)
            pstr = class_terms[index]
            entry = _blockentry(pstr, key_bits)
            word = (entry >> 6) + 1
            bit = one(UInt64) << (entry & 63)
            if present[word] & bit != 0
                coeffs[entry+1] = mergefunc(coeffs[entry+1], class_coeffs[index])
            else
                coeffs[entry+1] = class_coeffs[index]
                block_terms[entry+1] = pstr
                present[word] |= bit
            end
        end
        for step in eachindex(rotations)
            rotation = Int(rotations[step])
            entry_mask = _blockentry(plan.masks[rotation], key_bits)
            signs = _blocksigns(plan)
            _rotateblockwords!(present, n_words, coeffs, block_terms, n_entries, entry_mask, plan, rotation, signs, truncfunc)
        end
        _emitblockwords!(sink, present, n_words, coeffs, block_terms)
    end
    return sink
end

# The signs that a rotation gives every lower entry and its partner, if they are all the same, as for a rotation on one
# qubit, or `nothing`, and the signs are read for every pair.
@inline _blocksigns(plan) = plan.acts_on_one_qubit ? plan.lower_signs : nothing

# the signs that the rotation gives the partner of a lower entry and the lower entry, constant or read from its string
@inline _pairsigns(signs::Tuple, plan, pstr, rotation::Int) = signs

@inline function _pairsigns(::Nothing, plan, pstr, rotation::Int)
    paulis = _gatherpaulis(pstr, plan.shifts[rotation])
    return (plan.signs[paulis+1], plan.signs[(paulis⊻Int(plan.local_mask))+1])
end

# Rotates the pairs of one word: the entries at `lower_positions` of the word starting at `word_base` and their partners,
# whose presence `upper_present` holds at the positions of the lower entries. Pairs with both entries present mix both
# coefficients. An entry whose partner is absent keeps cos θ of its coefficient and makes the partner only if its
# coefficient can reach the truncation threshold. Returns which lower entries and which upper entries the truncations
# keep, both at the positions of the lower entries.
@inline function _rotatepairs!(coeffs::Vector{CT}, block_terms::Vector{TT}, word_base::Int, lower_present::UInt64, upper_present::UInt64,
    lower_positions::UInt64, entry_mask::Int, mask::TT, cos_val, sin_val, signs, min_coeff_to_make, plan, rotation::Int,
    truncfunc::F) where {F,CT,TT}

    both = lower_present & upper_present & lower_positions
    lower_alone = lower_present & ~upper_present & lower_positions
    upper_alone = upper_present & ~lower_present & lower_positions
    kept_lower = zero(UInt64)
    kept_upper = zero(UInt64)

    @inbounds while both != 0
        position = trailing_zeros(both)
        both &= both - one(UInt64)
        entry = word_base + position
        partner = entry ⊻ entry_mask
        pstr = block_terms[entry+1]
        partner_pstr = pstr ⊻ mask
        sign_to_partner, sign_from_partner = _pairsigns(signs, plan, pstr, rotation)
        coeff = coeffs[entry+1]
        partner_coeff = coeffs[partner+1]
        new_coeff = mergefunc(coeff * cos_val, partner_coeff * sin_val * sign_from_partner)
        new_partner_coeff = mergefunc(partner_coeff * cos_val, coeff * sin_val * sign_to_partner)
        coeffs[entry+1] = new_coeff
        coeffs[partner+1] = new_partner_coeff
        kept_lower |= UInt64(!@inline(truncfunc(pstr, new_coeff))) << position
        kept_upper |= UInt64(!@inline(truncfunc(partner_pstr, new_partner_coeff))) << position
    end

    @inbounds while lower_alone != 0
        position = trailing_zeros(lower_alone)
        lower_alone &= lower_alone - one(UInt64)
        entry = word_base + position
        coeff = coeffs[entry+1]
        pstr = block_terms[entry+1]
        new_coeff = coeff * cos_val
        coeffs[entry+1] = new_coeff
        kept_lower |= UInt64(!@inline(truncfunc(pstr, new_coeff))) << position
        if abs(coeff) >= min_coeff_to_make
            sign_to_partner, _ = _pairsigns(signs, plan, pstr, rotation)
            partner_pstr = pstr ⊻ mask
            new_partner_coeff = coeff * sin_val * sign_to_partner
            if !@inline(truncfunc(partner_pstr, new_partner_coeff))
                partner = entry ⊻ entry_mask
                coeffs[partner+1] = new_partner_coeff
                block_terms[partner+1] = partner_pstr
                kept_upper |= one(UInt64) << position
            end
        end
    end

    @inbounds while upper_alone != 0
        position = trailing_zeros(upper_alone)
        upper_alone &= upper_alone - one(UInt64)
        entry = word_base + position
        partner = entry ⊻ entry_mask
        partner_coeff = coeffs[partner+1]
        partner_pstr = block_terms[partner+1]
        new_partner_coeff = partner_coeff * cos_val
        coeffs[partner+1] = new_partner_coeff
        kept_upper |= UInt64(!@inline(truncfunc(partner_pstr, new_partner_coeff))) << position
        if abs(partner_coeff) >= min_coeff_to_make
            pstr = partner_pstr ⊻ mask
            _, sign_from_partner = _pairsigns(signs, plan, pstr, rotation)
            new_coeff = partner_coeff * sin_val * sign_from_partner
            if !@inline(truncfunc(pstr, new_coeff))
                coeffs[entry+1] = new_coeff
                block_terms[entry+1] = pstr
                kept_lower |= one(UInt64) << position
            end
        end
    end

    return kept_lower, kept_upper
end

# One rotation on a block of at most 64 entries, whose presence is one word. Returns which entries are present after it.
@inline function _rotateblockword(present::UInt64, coeffs::Vector{CT}, block_terms::Vector{TT}, n_entries::Int, entry_mask::Int, plan,
    rotation::Int, signs, truncfunc::F) where {F,CT,TT}

    if !(0 < entry_mask < n_entries <= 64) || n_entries > min(length(coeffs), length(block_terms))
        throw(ArgumentError("the block does not fit the workspace"))
    end
    sin_val = plan.sines[rotation]
    upper_present = _xorpositions(present, entry_mask)
    lower_positions = _LOWER_POSITIONS[trailing_zeros(entry_mask)+1]
    kept_lower, kept_upper = _rotatepairs!(coeffs, block_terms, 0, present, upper_present, lower_positions, entry_mask, plan.masks[rotation],
        plan.cosines[rotation], sin_val, signs, plan.min_coeffs_to_make[rotation], plan, rotation, truncfunc)
    return kept_lower | _xorpositions(kept_upper, entry_mask)
end

# One rotation on a block of several words, whose presence is updated in place. A pair within words is visited from the
# word of its lower entry, which writes the lower positions of that word and the upper positions of its partner word; a
# pair across words is visited from the lower word, which writes both words.
function _rotateblockwords!(present::Vector{UInt64}, n_words::Int, coeffs::Vector{CT}, block_terms::Vector{TT}, n_entries::Int,
    entry_mask::Int, plan, rotation::Int, signs, truncfunc::F) where {F,CT,TT}

    if !(0 < entry_mask < n_entries) || n_entries != 64 * n_words || n_words > length(present) ||
       n_entries > min(length(coeffs), length(block_terms))
        throw(ArgumentError("the block does not fit the workspace"))
    end
    mask = plan.masks[rotation]
    cos_val = plan.cosines[rotation]
    sin_val = plan.sines[rotation]
    min_coeff_to_make = plan.min_coeffs_to_make[rotation]
    lowest = trailing_zeros(entry_mask)
    low_mask = entry_mask & 63
    word_mask = entry_mask >> 6
    lower_positions = lowest < 6 ? _LOWER_POSITIONS[lowest+1] : typemax(UInt64)

    @inbounds for word in 0:n_words-1
        if lowest >= 6 && isodd(word >> (lowest - 6))
            continue
        end
        partner_word = word ⊻ word_mask
        kept_lower, kept_upper = _rotatepairs!(coeffs, block_terms, 64 * word, present[word+1], _xorpositions(present[partner_word+1], low_mask),
            lower_positions, entry_mask, mask, cos_val, sin_val, signs, min_coeff_to_make, plan, rotation, truncfunc)
        if lowest < 6
            present[word+1] = (present[word+1] & ~lower_positions) | kept_lower
            present[partner_word+1] = (present[partner_word+1] & lower_positions) | _xorpositions(kept_upper, low_mask)
        else
            present[word+1] = kept_lower
            present[partner_word+1] = _xorpositions(kept_upper, low_mask)
        end
    end
    return
end

# the entries of a block that are present, from the first `n_words` words of `present`
function _emitblockwords!(sink, present, n_words::Int, coeffs::Vector{CT}, block_terms::Vector{TT}) where {CT,TT}
    for word in 0:n_words-1
        bits = present[word+1]
        while bits != 0
            entry = 64 * word + trailing_zeros(bits)
            bits &= bits - one(UInt64)
            _emit!(sink, block_terms[entry+1], coeffs[entry+1])
        end
    end
    return
end
