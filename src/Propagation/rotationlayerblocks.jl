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

# moves the bit at every position p of a word to p ⊻ low_mask, for low_mask below 64
@inline function _xorpositions(word::UInt64, low_mask::Int)
    for j in 0:5
        if isodd(low_mask >> j)
            span = 1 << j
            lower = _LOWER_POSITIONS[j+1]
            word = ((word >> span) & lower) | ((word & lower) << span)
        end
    end
    return word
end

"""
    _rotateblock!(output, task, plan, truncation, record_terms, record_coeffs, lo, first, last, n_rotations, key_bits, n_key_bits)

Rotates the class of the records `class_records[first:last]`, counted from `lo`, as a dense block of `2^n_key_bits`
entries, one rotation after the other, and writes what the truncations keep to `output`.
"""
function _rotateblock!(output, task, plan, truncation, record_terms::Vector{TT}, record_coeffs::Vector{CT}, lo::Int, first::Int,
    last::Int, n_rotations::Int, key_bits::TT, n_key_bits::Int) where {TT,CT}

    n_entries = 1 << n_key_bits
    coeffs = _ensurelength!(task.block_coeffs, n_entries)
    block_terms = _ensurelength!(task.block_terms, n_entries)
    first_pstr = record_terms[lo-1+task.class_records[first]]
    first_entry = _blockentry(first_pstr, key_bits)

    # a block of at most 64 entries keeps which are present in one word, any other in an array of words
    if n_entries <= 64
        present = zero(UInt64)
        for index in first:last
            i = lo - 1 + task.class_records[index]
            pstr = record_terms[i]
            entry = _blockentry(pstr, key_bits)
            bit = one(UInt64) << entry
            if present & bit != 0
                coeffs[entry+1] = mergefunc(coeffs[entry+1], record_coeffs[i])
            else
                coeffs[entry+1] = record_coeffs[i]
                block_terms[entry+1] = pstr
                present |= bit
            end
        end
        for step in 1:n_rotations
            rotation = Int(task.rotations[step])
            entry_mask = _blockentry(plan.masks[rotation], key_bits)
            signs = _blocksigns(plan, rotation, entry_mask, key_bits, first_pstr, first_entry)
            present = _rotateblockword(present, coeffs, block_terms, n_entries, entry_mask, plan, rotation, signs, truncation)
        end
        _emitblockword!(output, present, coeffs, block_terms)
    else
        n_words = n_entries >> 6
        present = _ensurelength!(task.block_words, n_words)
        for word in 1:n_words
            present[word] = zero(UInt64)
        end
        for index in first:last
            i = lo - 1 + task.class_records[index]
            pstr = record_terms[i]
            entry = _blockentry(pstr, key_bits)
            word = (entry >> 6) + 1
            bit = one(UInt64) << (entry & 63)
            if present[word] & bit != 0
                coeffs[entry+1] = mergefunc(coeffs[entry+1], record_coeffs[i])
            else
                coeffs[entry+1] = record_coeffs[i]
                block_terms[entry+1] = pstr
                present[word] |= bit
            end
        end
        for step in 1:n_rotations
            rotation = Int(task.rotations[step])
            entry_mask = _blockentry(plan.masks[rotation], key_bits)
            signs = _blocksigns(plan, rotation, entry_mask, key_bits, first_pstr, first_entry)
            _rotateblockwords!(present, n_words, coeffs, block_terms, n_entries, entry_mask, plan, rotation, signs, truncation)
        end
        _emitblockwords!(output, present, n_words, coeffs, block_terms)
    end
    return output
end

# The signs of the rotation for every lower entry, if they are all the same: where the rotation's single key bit is the
# only key bit on its qubits, every lower entry has the Paulis of the lower entry of the first string's pair there.
# Otherwise `nothing`, and the signs are read for every pair.
@inline function _blocksigns(plan, rotation::Int, entry_mask::Int, key_bits::TT, first_pstr::TT, first_entry::Int) where {TT}
    if count_ones(entry_mask) != 1 || plan.qubit_masks[rotation] & key_bits != plan.masks[rotation] & key_bits
        return nothing
    end
    lower_pstr = ifelse(first_entry & entry_mask != 0, first_pstr ⊻ plan.masks[rotation], first_pstr)
    paulis = _localpaulis(plan, lower_pstr, rotation)
    return (plan.signs[paulis+1], plan.signs[(paulis⊻Int(plan.local_mask))+1])
end

# the signs that the rotation gives the partner of a lower entry and the lower entry, constant or read from its string
@inline _pairsigns(signs::Tuple, plan, pstr, rotation::Int) = signs

@inline function _pairsigns(::Nothing, plan, pstr, rotation::Int)
    paulis = _localpaulis(plan, pstr, rotation)
    return (plan.signs[paulis+1], plan.signs[(paulis⊻Int(plan.local_mask))+1])
end

# Rotates the pairs of one word: the entries at `lower_positions` of the word starting at `word_base` and their partners,
# whose presence `upper_present` holds at the positions of the lower entries. Pairs with both entries present mix both
# coefficients. An entry whose partner is absent keeps cos θ of its coefficient and makes the partner only if its
# coefficient can reach the truncation threshold. Returns which lower entries and which upper entries the truncations
# keep, both at the positions of the lower entries.
@inline function _rotatepairs!(coeffs::Vector{CT}, block_terms::Vector{TT}, word_base::Int, lower_present::UInt64, upper_present::UInt64,
    lower_positions::UInt64, entry_mask::Int, mask::TT, cos_val, sin_val, signs, min_coeff_to_make, plan, rotation::Int,
    truncation) where {CT,TT}

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
        kept_lower |= UInt64(!_istruncated(truncation, pstr, new_coeff)) << position
        kept_upper |= UInt64(!_istruncated(truncation, partner_pstr, new_partner_coeff)) << position
    end

    @inbounds while lower_alone != 0
        position = trailing_zeros(lower_alone)
        lower_alone &= lower_alone - one(UInt64)
        entry = word_base + position
        coeff = coeffs[entry+1]
        pstr = block_terms[entry+1]
        new_coeff = coeff * cos_val
        coeffs[entry+1] = new_coeff
        kept_lower |= UInt64(!_istruncated(truncation, pstr, new_coeff)) << position
        if abs(coeff) >= min_coeff_to_make
            sign_to_partner, _ = _pairsigns(signs, plan, pstr, rotation)
            partner_pstr = pstr ⊻ mask
            new_partner_coeff = coeff * sin_val * sign_to_partner
            if !_istruncated(truncation, partner_pstr, new_partner_coeff)
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
        kept_upper |= UInt64(!_istruncated(truncation, partner_pstr, new_partner_coeff)) << position
        if abs(partner_coeff) >= min_coeff_to_make
            pstr = partner_pstr ⊻ mask
            _, sign_from_partner = _pairsigns(signs, plan, pstr, rotation)
            new_coeff = partner_coeff * sin_val * sign_from_partner
            if !_istruncated(truncation, pstr, new_coeff)
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
    rotation::Int, signs, truncation) where {CT,TT}

    if !(0 < entry_mask < n_entries <= 64) || n_entries > min(length(coeffs), length(block_terms))
        throw(ArgumentError("the block does not fit the workspace"))
    end
    sin_val = plan.sines[rotation]
    upper_present = _xorpositions(present, entry_mask)
    lower_positions = _LOWER_POSITIONS[trailing_zeros(entry_mask)+1]
    kept_lower, kept_upper = _rotatepairs!(coeffs, block_terms, 0, present, upper_present, lower_positions, entry_mask, plan.masks[rotation],
        plan.cosines[rotation], sin_val, signs, _mincoefftomake(truncation, sin_val), plan, rotation, truncation)
    return kept_lower | _xorpositions(kept_upper, entry_mask)
end

# One rotation on a block of several words, whose presence is updated in place. A pair within words is visited from the
# word of its lower entry, which writes the lower positions of that word and the upper positions of its partner word; a
# pair across words is visited from the lower word, which writes both words.
function _rotateblockwords!(present::Vector{UInt64}, n_words::Int, coeffs::Vector{CT}, block_terms::Vector{TT}, n_entries::Int,
    entry_mask::Int, plan, rotation::Int, signs, truncation) where {CT,TT}

    if !(0 < entry_mask < n_entries) || n_entries != 64 * n_words || n_words > length(present) ||
       n_entries > min(length(coeffs), length(block_terms))
        throw(ArgumentError("the block does not fit the workspace"))
    end
    mask = plan.masks[rotation]
    cos_val = plan.cosines[rotation]
    sin_val = plan.sines[rotation]
    min_coeff_to_make = _mincoefftomake(truncation, sin_val)
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
            lower_positions, entry_mask, mask, cos_val, sin_val, signs, min_coeff_to_make, plan, rotation, truncation)
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

# the entries of a block of one word that are present
function _emitblockword!(output::ArrayOutputs, present::UInt64, coeffs::Vector{CT}, block_terms::Vector{TT}) where {CT,TT}
    output_terms, output_coeffs = _roomtoemit!(output, count_ones(present))
    n_written = output.n_written
    while present != 0
        entry = trailing_zeros(present)
        present &= present - one(UInt64)
        n_written += 1
        output_terms[n_written] = block_terms[entry+1]
        output_coeffs[n_written] = coeffs[entry+1]
    end
    output.n_written = n_written
    return
end

function _emitblockword!(output, present::UInt64, coeffs::Vector{CT}, block_terms::Vector{TT}) where {CT,TT}
    while present != 0
        entry = trailing_zeros(present)
        present &= present - one(UInt64)
        _emit!(output, block_terms[entry+1], coeffs[entry+1])
    end
    return
end

# the entries of a block of several words that are present
function _emitblockwords!(output::ArrayOutputs, present::Vector{UInt64}, n_words::Int, coeffs::Vector{CT}, block_terms::Vector{TT}) where {CT,TT}
    n_present = 0
    for word in 1:n_words
        n_present += count_ones(present[word])
    end
    output_terms, output_coeffs = _roomtoemit!(output, n_present)
    n_written = output.n_written
    for word in 0:n_words-1
        bits = present[word+1]
        while bits != 0
            entry = 64 * word + trailing_zeros(bits)
            bits &= bits - one(UInt64)
            n_written += 1
            output_terms[n_written] = block_terms[entry+1]
            output_coeffs[n_written] = coeffs[entry+1]
        end
    end
    output.n_written = n_written
    return
end

function _emitblockwords!(output, present::Vector{UInt64}, n_words::Int, coeffs::Vector{CT}, block_terms::Vector{TT}) where {CT,TT}
    for word in 0:n_words-1
        bits = present[word+1]
        while bits != 0
            entry = 64 * word + trailing_zeros(bits)
            bits &= bits - one(UInt64)
            _emit!(output, block_terms[entry+1], coeffs[entry+1])
        end
    end
    return
end
