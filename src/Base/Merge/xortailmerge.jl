###
##
# Sorting the tail a gate appended as `term ⊻ mask`, without comparing terms.
#
# XORing a fixed mask into sorted terms keeps two terms in relative order unless their highest
# differing bit was flipped; one pass per group of neighbouring flipped bits, highest group first, orders it.
# Within a pass the affected blocks arrive exactly reversed, so a pass only copies blocks back to front.
# Only valid if the tail is `term ⊻ mask` of an ascending run, which one pass over the tail checks.
##
###

# a mask spread over more groups than this is cheaper to sort the usual way
const _MAX_XOR_PASSES = 4

# below this many appended terms, setting up the passes costs more than it saves
const _MIN_XOR_TAIL = 64

# need at least three passes to pay it up
const _MIN_XOR_PASS_TASKS = 3


### Merge driver

"""
    xormerge!(prop_cache::AbstractPropagationCache, mask; thread=true)

`merge!` for a cache that `xorbranch!` has branched by `mask`.
An array sorts the tail past its sorted prefix in by XOR passes instead of by comparison when the tail is `term ⊻ mask` of an ascending run of terms,
which one pass over the tail checks, and merges it generically otherwise.
A multi sum takes delivery of the outboxes `xorbranch!` filled, sorting each in from where it is.
"""
xormerge!(prop_cache::AbstractPropagationCache, mask; thread::Bool=true) =
    _xormerge!(StorageType(prop_cache), prop_cache, mask; thread)

"""
    xormergeandtruncate!(truncfunc, prop_cache::AbstractPropagationCache, mask; thread=true)

`mergeandtruncate!` for a cache that `xorbranch!` has branched by `mask`, see `xormerge!`.
Several tasks on an array call `truncfunc` once to count the terms and once to write them, so it must return the same for the same pair each time.
"""
xormergeandtruncate!(truncfunc::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true) where {F} =
    _xormergeandtruncate!(StorageType(prop_cache), truncfunc, prop_cache, mask; thread)

# only an array keeps the new terms apart in an order the mask can sort
_xormerge!(::StorageType, prop_cache::AbstractPropagationCache, mask; thread::Bool=true) =
    merge!(prop_cache; thread)

_xormergeandtruncate!(::StorageType, truncfunc::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true) where {F} =
    mergeandtruncate!(truncfunc, prop_cache; thread)

function _xormerge!(::ArrayStorage, prop_cache::AbstractPropagationCache, mask; thread::Bool=true)
    groups = _xortailplan(prop_cache, mask; thread)
    if groups === nothing
        return merge!(prop_cache; thread)
    end
    return _xorsortedtailmerge!(groups, prop_cache; thread)
end

function _xormergeandtruncate!(::ArrayStorage, truncfunc::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true) where {F}
    groups = _xortailplan(prop_cache, mask; thread)
    if groups === nothing
        return mergeandtruncate!(truncfunc, prop_cache; thread)
    end
    return _xorsortedtailmergeandtruncate!(groups, truncfunc, prop_cache; thread)
end

# The box a zone collected is its tail already, in the parent order of the zone that made it, so an
# array zone sorts it in from where it is instead of taking delivery first.
function _xormerge!(::MultiSumStorage, prop_cache::AbstractPropagationCache, mask; thread::Bool=true)
    zone_storage = zonestorage(prop_cache)
    merge_box!(owner) = _xormergebox!(zone_storage, zonecaches(prop_cache)[owner], _branchboxfor(prop_cache, owner, mask), mask)
    _eachzone(merge_box!, prop_cache, thread)
    return _syncsums!(prop_cache)
end

function _xormergeandtruncate!(::MultiSumStorage, truncfunc::F, prop_cache::AbstractPropagationCache, mask; thread::Bool=true) where {F}
    zone_storage = zonestorage(prop_cache)
    merge_box!(owner) = _xormergeandtruncatebox!(zone_storage, truncfunc, zonecaches(prop_cache)[owner], _branchboxfor(prop_cache, owner, mask), mask)
    _eachzone(merge_box!, prop_cache, thread)
    return _syncsums!(prop_cache)
end

_xormergebox!(::StorageType, zonecache, box, mask) = merge!(_deliver!(zonecache, box); thread=false)

_xormergeandtruncatebox!(::StorageType, truncfunc::F, zonecache, box, mask) where {F} =
    mergeandtruncate!(truncfunc, _deliver!(zonecache, box); thread=false)

function _xormergebox!(::ArrayStorage, zonecache, box, mask)
    groups = _xorboxplan(zonecache, box, mask; thread=false)
    if groups === nothing
        return merge!(_deliver!(zonecache, box); thread=false)
    end
    return _xorsortedboxmerge!(groups, zonecache, box; thread=false)
end

function _xormergeandtruncatebox!(::ArrayStorage, truncfunc::F, zonecache, box, mask) where {F}
    groups = _xorboxplan(zonecache, box, mask; thread=false)
    if groups === nothing
        return mergeandtruncate!(truncfunc, _deliver!(zonecache, box); thread=false)
    end
    return _xorsortedboxmergeandtruncate!(groups, truncfunc, zonecache, box; thread=false)
end


### Planning the merge

# The XOR passes that sort the tail past the sorted prefix, or `nothing` where the merge by
# comparison is the better or the only choice: a short tail, a mask that is not a term of a CPU
# array, a mask spread over too many groups, or a tail that is not `term ⊻ mask` of an ascending run.
function _xortailplan(prop_cache::AbstractPropagationCache, mask; thread::Bool=true)
    main_terms = terms(mainsum(prop_cache))
    return _xorpasses(main_terms, main_terms, sortedprefix(mainsum(prop_cache)) + 1, activesize(prop_cache), mask; thread)
end

# The same for a tail that is still in `box`, the outbox of an array zone, which the passes only
# sort in past a head that holds every active term.
function _xorboxplan(prop_cache::AbstractPropagationCache, box, mask; thread::Bool=true)
    if sortedprefix(mainsum(prop_cache)) != activesize(prop_cache)
        return nothing
    end
    return _xorpasses(terms(mainsum(prop_cache)), terms(box), 1, length(box), mask; thread)
end

function _xorpasses(main_terms, tail_terms, lo::Int, hi::Int, mask; thread::Bool=true)
    if hi - lo + 1 < _MIN_XOR_TAIL
        return nothing
    end

    groups = _xorplan(mask, main_terms)
    if groups === nothing || !_isxortail(tail_terms, lo, hi, mask; thread)
        return nothing
    end
    return groups
end


### Sorting the tail in

# `sortedtailmerge!` for a tail appended as `term ⊻ mask` in parent order, sorted by the passes of
# `groups`
_xorsortedtailmerge!(groups, prop_cache::AbstractPropagationCache; thread::Bool=true) =
    _xorsortedtailmergeandtruncate!(groups, nothing, prop_cache; thread)

# the same, dropping the pairs `truncfunc` rejects as they are written when there is one
function _xorsortedtailmergeandtruncate!(groups, truncfunc::F, prop_cache::AbstractPropagationCache; thread::Bool=true) where {F}
    n_old = sortedprefix(mainsum(prop_cache))
    n_new = activesize(prop_cache)
    n_tail = n_new - n_old
    main_terms, main_coeffs, aux_terms, aux_coeffs = _mainauxarrays(prop_cache)

    # ping-pong pair A: the appended tail, in place at the end of the main arrays
    a_terms = view(main_terms, n_old+1:n_new)
    a_coeffs = view(main_coeffs, n_old+1:n_new)

    # ping-pong pair B: scratch
    buf_terms, buf_coeffs = _tailscratch(aux_terms, aux_coeffs, n_new, n_tail, main_terms, main_coeffs)
    b_terms = view(buf_terms, 1:n_tail)
    b_coeffs = view(buf_coeffs, 1:n_tail)

    tail_terms, tail_coeffs = _xorsorttail!(groups, a_terms, a_coeffs, b_terms, b_coeffs; thread)

    # an ascending run has no duplicates, so neither does the tail
    return _mergesortedhead!(prop_cache, aux_terms, aux_coeffs, main_terms, main_coeffs,
        n_old, tail_terms, tail_coeffs, n_tail, truncfunc, thread, Val(true))
end

# The same for a tail that is still in `box`, a term sum of the same array type. The XOR passes
# read the box where it is and ping-pong between it and the room past the active terms, so the
# tail is moved by the sort alone instead of being copied in first. The box is empty afterwards.
_xorsortedboxmerge!(groups, prop_cache::AbstractPropagationCache, box; thread::Bool=true) =
    _xorsortedboxmergeandtruncate!(groups, nothing, prop_cache, box; thread)

function _xorsortedboxmergeandtruncate!(groups, truncfunc::F, prop_cache::AbstractPropagationCache, box; thread::Bool=true) where {F}
    n_tail = length(box)
    n_old = activesize(prop_cache)

    # the head keeps its place, the sorted tail lands past it or stays in the box, and the merge
    # writes both into aux: room for all of them in main and aux alike, and half as much again
    n_new = n_old + n_tail
    if capacity(prop_cache) < n_new
        n_room = n_new + n_tail
        resize!(prop_cache, n_room + n_room >> 1)
    end
    main_terms, main_coeffs, aux_terms, aux_coeffs = _mainauxarrays(prop_cache)

    # ping-pong pair A: the box, in place
    a_terms = view(terms(box), 1:n_tail)
    a_coeffs = view(coefficients(box), 1:n_tail)

    # ping-pong pair B: past the active terms of the main arrays
    b_terms = view(main_terms, n_old+1:n_new)
    b_coeffs = view(main_coeffs, n_old+1:n_new)

    tail_terms, tail_coeffs = _xorsorttail!(groups, a_terms, a_coeffs, b_terms, b_coeffs; thread)

    _mergesortedhead!(prop_cache, aux_terms, aux_coeffs, main_terms, main_coeffs,
        n_old, tail_terms, tail_coeffs, n_tail, truncfunc, thread, Val(true))

    empty!(box)

    return prop_cache
end


### Checking the tail

# whether terms[lo:hi] is `term ⊻ mask` of a strictly ascending run, which is what the XOR passes sort
function _isxortail(terms, lo::Int, hi::Int, mask; thread::Bool=true)
    checkbounds(terms, lo:hi)
    n = hi - lo + 1
    task_partitioner, n_tasks = _preparetasks(n, thread)

    if n_tasks == 1
        return _isxortailchunk(terms, lo, hi, mask)
    end

    # a chunk also checks the step from the last term of the chunk before it
    ascending = Vector{Bool}(undef, n_tasks)
    function check_chunk!(task_id)
        chunk = task_partitioner[task_id]
        ascending[task_id] = _isxortailchunk(terms, lo + chunk.start - 1 - (task_id > 1), lo + chunk.stop - 1, mask)
    end
    _eachtask(check_chunk!, n_tasks)

    return all(ascending)
end

function _isxortailchunk(terms, lo::Int, hi::Int, mask)
    @inbounds for ii in lo:hi-1
        if !((terms[ii] ⊻ mask) < (terms[ii+1] ⊻ mask))
            return false
        end
    end
    return true
end


### Sorting the tail

# sort the tail in pair A (a_terms[i] == sources[i] ⊻ mask, sources strictly ascending),
# in a ping-pong fashion into pair B; returns the pair holding the result
function _xorsorttail!(groups, a_terms, a_coeffs, b_terms, b_coeffs; thread::Bool=true)
    # a pass indexes all four arrays up to the length of the pair it reads
    if !(length(a_terms) == length(a_coeffs) == length(b_terms) == length(b_coeffs))
        throw(ArgumentError("the ping-pong pairs must be the same length"))
    end

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
        _eachtask(n_tasks) do task_id
            chunk = task_partitioner[task_id]
            _xorpasschunk!(dst_terms, dst_coeffs, src_terms, src_coeffs, group, above, chunk.start, chunk.stop)
        end
    end

    return nothing
end

# One pass over the whole tail: within each run of terms agreeing above the group, the blocks
# agreeing in the group go out back to front, taken from the top so the writes run forward.
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

# The same pass restricted to one task's source positions [c_lo, c_hi]; runs and blocks reaching
# over a chunk boundary are recovered in full on both sides. Block [bl, bh) of run [i, j) lands at
# i + (j - bh), which depends on the run alone, so tasks agree on destinations without communicating.
function _xorpasschunk!(dst_terms, dst_coeffs, src_terms, src_coeffs, group::TT, above::TT, c_lo::Int, c_hi::Int) where {TT}
    n = length(src_terms)

    # start of the run holding c_lo, possibly back in the previous chunk
    i = _agreefirst(src_terms, above, c_lo, 1)

    # `x` trails the lowest position of the current run that this task owns
    x = c_lo
    @inbounds while x <= c_hi
        j = _stretchend(src_terms, above, x, c_hi, n)

        # likewise for a block reaching into the next chunk
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

function _xorplan(xor_mask, terms::AbstractArray{TT}) where {TT}
    if TT <: Unsigned && xor_mask isa TT && _iscpuarray(terms)
        return _maskgroups(xor_mask)
    else
        return nothing
    end
end

# group the set bits of `mask` into runs of neighbours, lowest first, each as (group mask, mask of
# everything above it); `nothing` for no bits or more than _MAX_XOR_PASSES groups
function _maskgroups(mask::TT) where {TT}
    bits = _masksetbits(mask)
    if isempty(bits)
        return nothing
    end

    groups = Tuple{TT,TT}[]
    lo = 1
    while lo <= length(bits)
        hi = lo
        while hi < length(bits) && bits[hi+1] == bits[hi] + 1
            hi += 1
        end
        if length(groups) == _MAX_XOR_PASSES
            return nothing
        end
        above = _bitsfrom(TT, bits[hi] + 1)
        push!(groups, (mask & ~above & _bitsfrom(TT, bits[lo]), above))
        lo = hi + 1
    end

    return groups
end

# every bit from `lo` up (empty past the top of the type, as the top group needs)
_bitsfrom(::Type{TT}, lo::Int) where {TT} = ~((one(TT) << lo) - one(TT))


### Finding and moving runs and blocks

# `terms .& mask` is monotone over the searched range, so terms agreeing under `mask` are contiguous
# and both ends of a stretch are binary-searchable.

# one past the end of the stretch agreeing with terms[idx] under `mask`: scan to `scan_hi`, then
# binary search to `search_hi` (a stretch is usually short, but may span the array)
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
