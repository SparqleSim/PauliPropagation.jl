###
##
# Gate applications for `IndexedPauliPropagationCache`.
#
# A Pauli rotation splits the terms that anticommute with its generator and leaves the rest alone.
# The index names the first group, and the table places their products, so nothing else is read.
##
###

const _PERF = PauliPropagation.Performance

"""
    applymergetruncate!(gate::PauliRotation, cache::IndexedPauliPropagationCache, theta; kwargs...)

Apply one Pauli rotation to an indexed cache. The branching terms come from the transposed index
and their products are placed through the term table, so there is nothing left to merge afterwards.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.PauliRotation, cache::IndexedPauliPropagationCache, theta;
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, kwargs...)

    PauliPropagation._check_qind_range(nqubits(cache), gate.qinds)
    cache.active_size == 0 && return cache

    gate_mask = PauliPropagation.symboltoint(paulitype(cache), gate.symbols, gate.qinds)
    _rotate!(cache, gate_mask, cos(theta), sin(theta), min_abs_coeff, max_weight, Val(:PauliRotation))

    return cache
end

"""
    applymergetruncate!(gate::ImaginaryPauliRotation, cache::IndexedPauliPropagationCache, tau; normalize_coeffs=true, kwargs...)

Apply one imaginary Pauli rotation to an indexed cache. It splits the terms that *commute* with the
generator, which the index marks just as cheaply, and shares its kernel with the real rotation.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.ImaginaryPauliRotation, cache::IndexedPauliPropagationCache, tau;
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, normalize_coeffs::Bool=true, kwargs...)

    PauliPropagation._check_qind_range(nqubits(cache), gate.qinds)
    cache.active_size == 0 && return cache

    gate_mask = PauliPropagation.symboltoint(paulitype(cache), gate.symbols, gate.qinds)
    _rotate!(cache, gate_mask, cosh(tau), sinh(tau), min_abs_coeff, max_weight, Val(:ImaginaryPauliRotation))

    # this gate evolves states in the Schrödinger picture, and dividing by the coefficient of the
    # identity keeps that numerically stable
    if normalize_coeffs
        scale = 1 / getcoeff(cache, zero(paulitype(cache)))
        main_coeffs = coefficients(mainsum(cache))
        @inbounds for i in 1:cache.active_size
            main_coeffs[i] *= scale
        end
    end

    return cache
end

"""
    applymergetruncate!(gate::FrozenGate, cache::IndexedPauliPropagationCache; kwargs...)

Apply the wrapped gate with the parameter frozen into it. Only here to pick between the library's
`FrozenGate` method and this module's fallback, which are otherwise equally specific.
"""
PauliPropagation.applymergetruncate!(gate::PauliPropagation.FrozenGate, cache::IndexedPauliPropagationCache; kwargs...) =
    PauliPropagation.applymergetruncate!(gate.gate, cache, gate.parameter; kwargs...)

# A rotation pairs up the terms it acts on: if it takes `t` to `t ⊻ g`, it takes `t ⊻ g` back to
# `t`, and the two are split or left alone together. The pair is therefore rotated in one step,
# halving the table probes. Writing the real rotation of an anticommuting P as
# P -> cos θ P + sin θ (i G P), the same identity applied to i G P gives back -P, so the two
# products carry opposite signs; for the imaginary rotation, P -> cosh τ P - sinh τ P G on a
# commuting P, they carry the same sign. A term whose partner is not in the sum yet appends it.
function _rotate!(cache::IndexedPauliPropagationCache, gate_mask::TT, kept_val, new_val,
    min_abs_coeff::Real, max_weight::Real, gatetype::Val) where {TT}

    n_old = cache.active_size
    appendterms!(cache.index, terms(mainsum(cache)), n_old)

    n_branching = _markbranching!(cache.marks, cache.index, columnsof(gate_mask), n_old, gatetype)
    n_branching == 0 && return cache

    # every branching term may append a partner, and none of them may move the arrays afterwards
    _reserve!(cache, n_old + n_branching)
    _clearhandled!(cache, n_old)

    local_mask = _PERF._gatemask(gate_mask, terms(mainsum(cache)))
    _emitproducts!(cache, cache, n_old, local_mask, kept_val, new_val, min_abs_coeff, max_weight, gatetype)
    _maybecompact!(cache)

    return cache
end

# the sign the partner's contribution comes back with, relative to the one going out
@inline _pairsign(::Val{:PauliRotation}) = -1
@inline _pairsign(::Val{:ImaginaryPauliRotation}) = 1

# Walks the terms `src.marks` names, in ascending order, and rotates each against the partner that
# `dst`'s table finds for it, appending that partner to `dst` when it is not there yet. `src` and
# `dst` are the same store for a rotation inside one zone, and the two halves of a zone pair when a
# rotation moves terms between zones.
#
# A pair is handled once, at the first of the two walks that reaches it, and the other half is noted
# in `dst.handled` so that the walk that reaches it later skips it. Noting it in the marks
# themselves would be one bitmap less, but then the position of the next term would depend on the
# probe that just landed, which turns the whole loop into a pointer chase and leaves the memory
# system with a single outstanding miss.
function _emitproducts!(src::IndexedPauliPropagationCache, dst::IndexedPauliPropagationCache,
    n_old::Int, local_mask, kept_val, new_val, min_abs_coeff::Real, max_weight::Real, gatetype::Val)

    src_terms, src_coeffs = storage(mainsum(src))
    dst_terms, dst_coeffs = storage(mainsum(dst))
    marks, src_handled, dst_handled = src.marks, src.handled, dst.handled
    table = dst.table
    slots, slot_mask = table.slots, table.mask
    pair_sign = _pairsign(gatetype)

    position = dst.active_size
    src_dead = 0
    dst_dead = 0
    n_filled = table.nfilled

    GC.@preserve src_terms begin
        bytes = _PERF._bytesof(src_terms, local_mask)

        @inbounds for w in 1:(((n_old-1)>>6)+1)
            marked = marks[w]
            base = (w - 1) << 6

            while !iszero(marked)
                i = base + trailing_zeros(marked) + 1
                marked &= marked - one(UInt64)

                # the half of the pair that was reached first did the work for both
                iszero(src_handled[((i-1)>>6)+1] & (one(UInt64) << ((i - 1) & 63))) || continue

                # a truncated term contributes nothing; its partner, if it has one, revives it
                coeff = src_coeffs[i]
                iszero(coeff) && continue

                child, sign = _PERF._gateproduct(local_mask, src_terms[i], bytes, i)
                h = termhash(child)
                slot, partner = _findslot(slots, slot_mask, dst_terms, child, h)

                if partner != 0
                    partner_coeff = dst_coeffs[partner]
                    kept = _truncated(coeff * kept_val + partner_coeff * new_val * sign * pair_sign, min_abs_coeff)
                    moved = _truncated(partner_coeff * kept_val + coeff * new_val * sign, min_abs_coeff)

                    src_dead += iszero(kept)
                    dst_dead += iszero(moved) - iszero(partner_coeff)
                    src_coeffs[i] = kept
                    dst_coeffs[partner] = moved

                    dst_handled[((partner-1)>>6)+1] |= one(UInt64) << ((partner - 1) & 63)
                    continue
                end

                kept = _truncated(coeff * kept_val, min_abs_coeff)
                src_dead += iszero(kept)
                src_coeffs[i] = kept

                moved = _truncated(coeff * new_val * sign, min_abs_coeff)
                (iszero(moved) || _PERF._truncateweight(child, max_weight)) && continue

                position += 1
                dst_terms[position] = child
                dst_coeffs[position] = moved
                slots[slot] = slotentry(h, position)
                n_filled += 1
            end
        end
    end

    table.nfilled = n_filled
    dst.active_size = position
    src.n_dead += src_dead
    dst.n_dead += dst_dead

    return dst
end

@inline _truncated(coeff, min_abs_coeff::Real) = abs(coeff) < min_abs_coeff ? zero(coeff) : coeff

### Selecting the branching terms

# XORs the columns of the generator into `marks` and returns how many terms they mark. Unrolled for
# the column counts a one- or two-qubit rotation can produce.
function _markbranching!(marks::Vector{UInt64}, index::TransposedIndex, columns::Vector{Int}, upto::Int, gatetype::Val)
    n = length(columns)
    n == 1 && return _markcolumns!(marks, index, (columns[1],), upto, gatetype)
    n == 2 && return _markcolumns!(marks, index, (columns[1], columns[2]), upto, gatetype)
    n == 3 && return _markcolumns!(marks, index, (columns[1], columns[2], columns[3]), upto, gatetype)
    n == 4 && return _markcolumns!(marks, index, (columns[1], columns[2], columns[3], columns[4]), upto, gatetype)
    # a generator over more qubits than a two-qubit gate carries a Jordan-Wigner string, and its
    # columns are worth reading out of the vector rather than out of a tuple of unknown length
    return _markcolumns!(marks, index, columns, upto, gatetype)
end

function _markcolumns!(marks::Vector{UInt64}, index::TransposedIndex, columns, upto::Int, ::Val{GateType}) where {GateType}
    upto == 0 && return 0

    words, nwords = index.words, index.nwords
    last_word = ((upto - 1) >> 6) + 1
    # bits past `upto` in the last word stand for terms that are not in the sum
    last_mask = (upto & 63) == 0 ? ~zero(UInt64) : (one(UInt64) << (upto & 63)) - one(UInt64)

    count = 0
    @inbounds for w in 1:last_word
        marked = zero(UInt64)
        for c in columns
            marked ⊻= words[(c-1)*nwords+w]
        end
        # an imaginary rotation splits the commuting terms, which are the ones the columns miss
        GateType === :ImaginaryPauliRotation && (marked = ~marked)
        w == last_word && (marked &= last_mask)
        marks[w] = marked
        count += count_ones(marked)
    end

    return count
end

### Pauli noise

"""
    applymergetruncate!(gate::PauliNoise, cache::IndexedPauliPropagationCache, lambda; kwargs...)

Apply a Pauli noise channel to an indexed cache. Noise damps coefficients without touching Pauli
strings, so both indices survive it and only the damped terms have to be found.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.PauliNoise, cache::IndexedPauliPropagationCache, lambda;
    min_abs_coeff::Real=1e-10, kwargs...)

    PauliPropagation._check_qind_range(nqubits(cache), gate.qind)
    PauliPropagation._check_noise_strength(PauliPropagation.PauliNoise, lambda)
    cache.active_size == 0 && return cache

    main_terms, main_coeffs = storage(mainsum(cache))
    qind = gate.qind
    n_dead = cache.n_dead

    @inbounds for i in 1:cache.active_size
        coeff = main_coeffs[i]
        iszero(coeff) && continue
        PauliPropagation.isdamped(gate, getpauli(main_terms[i], qind)) || continue
        damped = _truncated(coeff * (1 - lambda), min_abs_coeff)
        n_dead += iszero(damped)
        main_coeffs[i] = damped
    end

    cache.n_dead = n_dead
    _maybecompact!(cache)

    return cache
end

### Everything else

function PauliPropagation.applymergetruncate!(gate, cache::IndexedPauliPropagationCache, args...; kwargs...)
    throw(ArgumentError("$(typeof(gate)) is not implemented for IndexedPauliPropagationCache."))
end
