###
##
# Sorting the tail a gate appended as `term ⊻ mask`, without comparing terms.
#
# A gate that appends each new term as `term ⊻ mask`, in the order of the terms they came from,
# leaves the tail nearly sorted: flipping a fixed set of bits keeps two terms in the same relative
# order unless the highest bit where they differ is one of the flipped ones. So one pass per group of
# neighbouring flipped bits, highest group first, puts the tail in order. Within a pass the terms of
# a group arrive exactly reversed, so a pass only copies blocks back to front, and no bit pattern is
# ever read out.
#
# Only valid if every term the gate saw was already sorted; the caller states that.
##
###

# a mask spread over more groups than this is cheaper to sort the usual way
const _MAX_XOR_PASSES = 4

# below this many appended terms, setting up the passes costs more than it saves
const _MIN_XOR_TAIL = 64

# splitting a pass two ways is a wash: recovering the runs that straddle the one chunk boundary costs
# about what the second task saves. Three ways up it pays, and keeps paying.
const _MIN_XOR_PASS_TASKS = 3


### Merge driver

"""
    xorsortedtailmerge!(prop_cache::AbstractPropagationCache, xor_mask, sorted_before::Bool; thread=true, truncfunc=nothing, kwargs...)

`sortedtailmerge!` for a tail that a gate appended as `term ⊻ xor_mask`, in the order of the terms
they came from: the tail is sorted by one block-reversal pass per group of neighbouring set bits of
`xor_mask` (see the file header) instead of by a comparison sort, and the two sorted runs are then
combined by the same merge as `sortedtailmerge!`.

`sorted_before` states whether the whole active range was sorted and deduplicated before the gate was
applied, which is what makes the tail nearly sorted. Falls back to the generic `merge!` when it is
not, when the storage or term type does not support the passes, or when `xor_mask` is spread over too
many groups to pay off.
"""
function xorsortedtailmerge!(prop_cache::AbstractPropagationCache, xor_mask, sorted_before::Bool;
    thread::Bool=true, truncfunc=nothing, kwargs...)

    n_old = sortedprefix(mainsum(prop_cache))
    n_new = activesize(prop_cache)
    n_tail = n_new - n_old

    # nothing was appended; the active range is still sorted and deduplicated
    if n_tail == 0
        return prop_cache
    end

    main_terms, main_coeffs, aux_terms, aux_coeffs = _mainauxarrays(prop_cache)

    groups = (sorted_before && n_old > 0 && n_tail >= _MIN_XOR_TAIL) ? _xorplan(xor_mask, main_terms) : nothing
    if groups === nothing
        return merge!(prop_cache; thread, truncfunc, kwargs...)
    end

    # ping-pong pair A: the appended tail, in place at the end of the main arrays
    a_terms = view(main_terms, n_old+1:n_new)
    a_coeffs = view(main_coeffs, n_old+1:n_new)

    # ping-pong pair B: scratch, as views of the same kind so that swapping the two pairs stays
    # type stable
    buf_terms, buf_coeffs = _tailscratch(aux_terms, aux_coeffs, n_new, n_tail, main_terms, main_coeffs)
    b_terms = view(buf_terms, 1:n_tail)
    b_coeffs = view(buf_coeffs, 1:n_tail)

    tail_terms, tail_coeffs = _xorsorttail!(groups, a_terms, a_coeffs, b_terms, b_coeffs; thread)

    return _mergesortedhead!(prop_cache, aux_terms, aux_coeffs, main_terms, main_coeffs,
        n_old, tail_terms, tail_coeffs, n_tail, truncfunc, thread)
end


### Sorting the tail

"""
    _xorsorttail!(groups, a_terms, a_coeffs, b_terms, b_coeffs; thread=true)

Sort the tail held in pair A, given `a_terms[i] == sources[i] ⊻ mask` for a strictly ascending
sequence of sources and the `_maskgroups(mask)` of that mask, moving coefficients along with their
terms. Pair B is same-length scratch. Returns the pair holding the sorted tail.
"""
function _xorsorttail!(groups, a_terms, a_coeffs, b_terms, b_coeffs; thread::Bool=true)
    src_terms, src_coeffs = a_terms, a_coeffs
    dst_terms, dst_coeffs = b_terms, b_coeffs

    # highest group first; `_maskgroups` lists them lowest first
    for jj in length(groups):-1:1
        group, above = groups[jj]
        _xorpass!(dst_terms, dst_coeffs, src_terms, src_coeffs, group, above; thread)
        src_terms, dst_terms = dst_terms, src_terms
        src_coeffs, dst_coeffs = dst_coeffs, src_coeffs
    end

    return src_terms, src_coeffs
end

function _xorpass!(dst_terms, dst_coeffs, src_terms, src_coeffs, group, above; thread::Bool=true)
    task_partitioner, n_tasks = _preparetasks(length(src_terms), thread)

    if n_tasks < _MIN_XOR_PASS_TASKS
        _xorpassall!(dst_terms, dst_coeffs, src_terms, src_coeffs, group, above)
    else
        AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
            chunk = task_partitioner[task_id]
            _xorpasschunk!(dst_terms, dst_coeffs, src_terms, src_coeffs, group, above, chunk.start, chunk.stop)
        end
    end

    return nothing
end

# One pass over the whole tail: within each run of terms agreeing above the group, the blocks
# agreeing in the group go out back to front, taken from the top so the writes run forward. Runs and
# blocks are found by scanning for a change, so no bit pattern is ever read out.
function _xorpassall!(dst_terms, dst_coeffs, src_terms, src_coeffs, group::TT, above::TT) where {TT}
    n = length(src_terms)

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

# The same pass restricted to the source positions [c_lo, c_hi] of one task. Runs and blocks reaching
# over a chunk boundary are recovered in full on both sides, which costs the searches and clipping
# below -- hence the separate whole-tail version above. Block [bl, bh) of run [i, j) lands at
# i + (j - bh), which depends on the run alone, so tasks agree on where each element goes without
# communicating and no element is written twice.
function _xorpasschunk!(dst_terms, dst_coeffs, src_terms, src_coeffs, group::TT, above::TT, c_lo::Int, c_hi::Int) where {TT}
    n = length(src_terms)

    # the start of the run holding c_lo, even where that run reaches back into the previous chunk
    i = _agreefirst(src_terms, above, c_lo, 1)

    # `x` trails the lowest source position of the current run that this task owns
    x = c_lo
    @inbounds while x <= c_hi
        j = _stretchend(src_terms, above, x, c_hi, n)

        # likewise for a block reaching on into the next chunk
        y = min(j - 1, c_hi)
        bh = (y < j - 1 && iszero((src_terms[y+1] ⊻ src_terms[y]) & group)) ?
             _agreelast(src_terms, group, y, j - 1) + 1 : y + 1

        while y >= x
            bl = _stretchbegin(src_terms, group, y, x, i)
            lo = max(bl, x)
            _copyblock!(dst_terms, dst_coeffs, src_terms, src_coeffs, i + (j - bh) + (lo - bl), lo, y - lo + 1)
            y = bl - 1
            bh = bl
        end

        i = x = j
    end

    return nothing
end


### Planning the passes

# The passes are scalar CPU code that only ever XORs the mask into a term, so they need unsigned
# terms (bit order is value order), a matching mask, and a backing array that can be indexed cheaply.
function _xorplan(xor_mask, terms::AbstractArray{TT}) where {TT}
    (TT <: Unsigned && xor_mask isa TT && _iscpuarray(terms)) || return nothing
    return _maskgroups(xor_mask)
end

"""
    _maskgroups(mask)

Group the set bits of `mask` into runs of neighbours -- one pass each, lowest first. Each is kept as
the group mask and a mask of everything above it. Returns `nothing` if there are no bits, or too many
groups.
"""
function _maskgroups(mask::TT) where {TT}
    bits = _masksetbits(mask)
    isempty(bits) && return nothing

    groups = Tuple{TT,TT}[]
    lo = 1
    while lo <= length(bits)
        hi = lo
        while hi < length(bits) && bits[hi+1] == bits[hi] + 1
            hi += 1
        end
        length(groups) == _MAX_XOR_PASSES && return nothing
        above = _bitsfrom(TT, bits[hi] + 1)
        push!(groups, (mask & ~above & _bitsfrom(TT, bits[lo]), above))
        lo = hi + 1
    end

    return groups
end

# every bit from `lo` up; past the top of the term type this is empty, which is what the top group wants
_bitsfrom(::Type{TT}, lo::Int) where {TT} = ~((one(TT) << lo) - one(TT))


### Finding and moving runs and blocks

# Terms agreeing under `mask` are contiguous, because `terms .& mask` is monotone over the searched
# range, so both ends of such a stretch are a plain first-true binary search.

# one past the end of the stretch agreeing with terms[idx] under `mask`, scanned up to `scan_hi` and
# binary searched up to `search_hi` beyond that -- a stretch is usually short, but may span the array
@inline function _stretchend(terms, mask::TT, idx::Int, scan_hi::Int, search_hi::Int) where {TT}
    @inbounds begin
        key = terms[idx]
        k = idx + 1
        while k <= scan_hi && iszero((terms[k] ⊻ key) & mask)
            k += 1
        end
        if k > scan_hi && k <= search_hi && iszero((terms[k] ⊻ key) & mask)
            k = _agreelast(terms, mask, k, search_hi) + 1
        end
    end
    return k
end

# the same downwards: the first index of the stretch agreeing with terms[idx] under `mask`
@inline function _stretchbegin(terms, mask::TT, idx::Int, scan_lo::Int, search_lo::Int) where {TT}
    @inbounds begin
        key = terms[idx]
        k = idx
        while k > scan_lo && iszero((terms[k-1] ⊻ key) & mask)
            k -= 1
        end
        if k <= scan_lo && k > search_lo && iszero((terms[k-1] ⊻ key) & mask)
            k = _agreefirst(terms, mask, k, search_lo)
        end
    end
    return k
end

# first index in [lo, idx] agreeing with terms[idx] under `mask`
@inline function _agreefirst(terms, mask::TT, idx::Int, lo::Int) where {TT}
    @inbounds key = terms[idx]
    hi = idx
    @inbounds while lo < hi
        mid = (lo + hi) >>> 1
        if iszero((terms[mid] ⊻ key) & mask)
            hi = mid
        else
            lo = mid + 1
        end
    end
    return lo
end

# last index in [idx, hi] agreeing with terms[idx] under `mask`
@inline function _agreelast(terms, mask::TT, idx::Int, hi::Int) where {TT}
    @inbounds key = terms[idx]
    lo = idx
    @inbounds while lo < hi
        mid = (lo + hi + 1) >>> 1
        if iszero((terms[mid] ⊻ key) & mask)
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

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
