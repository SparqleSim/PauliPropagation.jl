###
##
# Sorting the new terms a rotation gate appends, without comparing them.
#
# A rotation adds each new term as `pstr ⊻ gate_mask`, in the order of the terms they came from.
# Flipping a fixed set of bits keeps two terms in the same relative order unless the highest bit where
# they differ is one of the flipped ones. So one sweep per group of neighbouring flipped bits, highest
# group first, puts them in order. Within a sweep the terms arrive exactly reversed, so a sweep only
# copies blocks back to front.
#
# Only valid if every term the gate saw was already sorted; the caller checks.
##
###

# set to false to send the fused rotations back through the usual `merge!`, for timing comparisons
const USE_RADIX_TAILSORT = Ref(true)

# a gate spread over more groups than this is cheaper to sort the usual way
const _MAX_SWEEPS = 4

# below this many new terms, setting up the sweeps costs more than it saves
const _MIN_RADIX_TAIL = 64


### Planning the passes

"""
    _radixplan(gate_mask, bits)

Group the flipped bits, given ascending in `bits`, into runs of neighbours -- one sweep each, lowest
first. Each is kept as the group mask and a mask of everything above it. Returns `nothing` if there
are no bits, or too many groups.
"""
function _radixplan(gate_mask::TT, bits) where {TT}
    isempty(bits) && return nothing

    plan = Tuple{TT,TT}[]
    lo = 1
    while lo <= length(bits)
        hi = lo
        while hi < length(bits) && bits[hi+1] == bits[hi] + 1
            hi += 1
        end
        length(plan) == _MAX_SWEEPS && return nothing
        above = _bitsfrom(TT, bits[hi] + 1)
        push!(plan, (gate_mask & ~above & _bitsfrom(TT, bits[lo]), above))
        lo = hi + 1
    end

    return plan
end

# every bit from `lo` up; past the top of the string this is empty, which is what the top group wants
_bitsfrom(::Type{TT}, lo::Int) where {TT} = ~((one(TT) << lo) - one(TT))


### The passes

# blocks are often a single term, where the call overhead of copyto! outweighs the copy
@inline function _copyblock!(dst_terms, dst_coeffs, src_terms, src_coeffs, d0::Int, s0::Int, n::Int)
    if n < 32
        @inbounds for k in 0:n-1
            dst_terms[d0+k] = src_terms[s0+k]
            dst_coeffs[d0+k] = src_coeffs[s0+k]
        end
    else
        copyto!(dst_terms, d0, src_terms, s0, n)
        copyto!(dst_coeffs, d0, src_coeffs, s0, n)
    end
    return nothing
end

# One sweep. Within each run of terms agreeing above the group, write its blocks out back to front.
# Both the runs and the blocks are found by watching for a change, so no bit pattern is ever read out.
function _radixpass!(group, above, dst_terms, dst_coeffs, src_terms, src_coeffs, n::Int)
    i = 1
    @inbounds while i <= n
        j = i + 1
        while j <= n && iszero((src_terms[i] ⊻ src_terms[j]) & above)
            j += 1
        end

        write_pos = i
        block_hi = j
        while block_hi > i
            block_lo = block_hi - 1
            while block_lo > i && iszero((src_terms[block_lo-1] ⊻ src_terms[block_hi-1]) & group)
                block_lo -= 1
            end
            n_block = block_hi - block_lo
            _copyblock!(dst_terms, dst_coeffs, src_terms, src_coeffs, write_pos, block_lo, n_block)
            write_pos += n_block
            block_hi = block_lo
        end

        i = j
    end
    return nothing
end

"""
    _radixsorttail!(plan, main_terms, main_coeffs, n_old, n_tail, buf_terms, buf_coeffs)

Sorts the new terms at `main_terms[n_old+1:end]` with one sweep per group, using
`buf_terms`/`buf_coeffs` as scratch space. Returns `true` if they ended up in the scratch space.
"""
function _radixsorttail!(plan, main_terms, main_coeffs, n_old::Int, n_tail::Int, buf_terms, buf_coeffs)
    # both sides must be the same kind of view, or swapping them below makes every access slow
    src_terms = view(main_terms, n_old+1:n_old+n_tail)
    src_coeffs = view(main_coeffs, n_old+1:n_old+n_tail)
    dst_terms = view(buf_terms, 1:n_tail)
    dst_coeffs = view(buf_coeffs, 1:n_tail)

    # highest group first; `_radixplan` lists them lowest first
    for jj in length(plan):-1:1
        group, above = plan[jj]
        _radixpass!(group, above, dst_terms, dst_coeffs, src_terms, src_coeffs, n_tail)
        src_terms, dst_terms = dst_terms, src_terms
        src_coeffs, dst_coeffs = dst_coeffs, src_coeffs
    end

    return isodd(length(plan))
end


### Merge driver

"""
    _radixtailmerge!(prop_cache, plan; thread=true, truncfunc=nothing)

Like `sortedtailmerge!`, but sorts the new terms by sweeps instead of by comparison.
"""
function _radixtailmerge!(prop_cache, plan; thread::Bool=true, truncfunc=nothing)
    n_old = sortedprefix(mainsum(prop_cache))
    n_new = activesize(prop_cache)
    n_tail = n_new - n_old
    if n_tail == 0
        return prop_cache
    end

    main_terms, main_coeffs, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(prop_cache)

    buf_terms, buf_coeffs = PropagationBase._tailscratch(aux_terms, aux_coeffs, n_new, n_tail, main_terms, main_coeffs)

    in_buffer = _radixsorttail!(plan, main_terms, main_coeffs, n_old, n_tail, buf_terms, buf_coeffs)

    tail_terms = in_buffer ? view(buf_terms, 1:n_tail) : view(main_terms, n_old+1:n_new)
    tail_coeffs = in_buffer ? view(buf_coeffs, 1:n_tail) : view(main_coeffs, n_old+1:n_new)

    return PropagationBase._mergesortedhead!(prop_cache, aux_terms, aux_coeffs, main_terms, main_coeffs,
        n_old, tail_terms, tail_coeffs, n_tail, truncfunc, thread)
end

"""
    _mergeafterrotation!(prop_cache, gate_mask, mask_bits, was_sorted; kwargs...)

`merge!` after a rotation that branched: sweeps when the new terms are still in the order they were
added and the gate is confined enough, the usual `merge!` otherwise.
"""
function _mergeafterrotation!(prop_cache, gate_mask, mask_bits, was_sorted::Bool; thread::Bool=true, truncfunc=nothing, kwargs...)
    n_tail = activesize(prop_cache) - sortedprefix(mainsum(prop_cache))
    if USE_RADIX_TAILSORT[] && was_sorted && n_tail >= _MIN_RADIX_TAIL &&
       PropagationBase._iscpuarray(PauliPropagation.terms(mainsum(prop_cache)))
        plan = _radixplan(_plainmask(gate_mask), mask_bits)
        plan !== nothing && return _radixtailmerge!(prop_cache, plan; thread, truncfunc)
    end
    return PauliPropagation.merge!(prop_cache; thread, truncfunc, kwargs...)
end
