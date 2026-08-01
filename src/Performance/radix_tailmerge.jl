###
##
# Sorting the new terms a rotation gate appends, without comparing them.
#
# A rotation adds each new term as `pstr ⊻ gate_mask`, in the order of the (already sorted) terms they
# came from. Flipping a fixed set of bits keeps two terms in the same relative order unless the highest
# bit where they differ is one of the flipped ones. So the new terms can be put in order by one sweep
# per group of neighbouring flipped bits, highest group first.
#
# Within each such sweep the terms arrive in exactly the reverse of the order they want, so a sweep is
# just a matter of copying blocks back to front. Nothing is compared and no bit pattern is ever read
# out, which is why the cost does not depend on how many bits a group covers.
#
# An ordinary sort cannot replace this: a sweep is only correct inside a group of terms that already
# agree on every bit above it, and bits the gate does not touch sit above and between those groups.
#
# Only valid if every term the gate saw was already sorted. The caller checks, and otherwise falls back
# to the usual merge.
##
###

# set to false to send the fused rotations back through the usual `merge!`, for timing comparisons
const USE_RADIX_TAILSORT = Ref(true)

# a gate spread over more groups than this is cheaper to sort the usual way
const _MAX_DIGITS = 4

# below this many new terms, setting up the sweeps costs more than it saves
const _MIN_RADIX_TAIL = 64

# each sweep copies every term once, so extra sweeps only pay for short terms or a small batch
const _MAX_RADIX_TERMBYTES = 32
const _MAX_RADIX_PASS_BYTES = 2 << 20

_radixpays(plan, n_tail::Int, termbytes::Int) =
    length(plan) == 1 || termbytes <= _MAX_RADIX_TERMBYTES || n_tail * termbytes <= _MAX_RADIX_PASS_BYTES


### Planning the passes

"""
    _radixplan(gate_mask, bits)

Collect the bits the gate flips, given ascending in `bits`, into groups of neighbouring bits -- one
sweep each, lowest first. Each group is kept as two masks, the group itself and everything above it,
so a sweep needs nothing but `⊻`, `&` and `iszero`. Returns `nothing` if there are none, or too many.
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
        length(plan) == _MAX_DIGITS && return nothing
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
function _radixpass!(digit, above, dst_terms, dst_coeffs, src_terms, src_coeffs, n::Int)
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
            while block_lo > i && iszero((src_terms[block_lo-1] ⊻ src_terms[block_hi-1]) & digit)
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
        digit, above = plan[jj]
        _radixpass!(digit, above, dst_terms, dst_coeffs, src_terms, src_coeffs, n_tail)
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

    # the merge writes at most n_new entries into aux, so anything past that is free scratch space
    if length(aux_terms) - n_new >= n_tail
        buf_terms = view(aux_terms, n_new+1:n_new+n_tail)
        buf_coeffs = view(aux_coeffs, n_new+1:n_new+n_tail)
    else
        buf_terms = similar(main_terms, n_tail)
        buf_coeffs = similar(main_coeffs, n_tail)
    end

    in_buffer = _radixsorttail!(plan, main_terms, main_coeffs, n_old, n_tail, buf_terms, buf_coeffs)

    tail_terms = in_buffer ? view(buf_terms, 1:n_tail) : view(main_terms, n_old+1:n_new)
    tail_coeffs = in_buffer ? view(buf_coeffs, 1:n_tail) : view(main_coeffs, n_old+1:n_new)

    merged_count = _headtailmerge!(aux_terms, aux_coeffs, main_terms, main_coeffs, n_old,
        tail_terms, tail_coeffs, n_tail, truncfunc, thread)

    PropagationBase._commitwrite!(prop_cache, merged_count, merged_count)

    return prop_cache
end

# Merge the old terms against the sorted new ones, into aux. Split the same way as
# `sortedtailmerge!`: divide the old terms, cut the new ones at the same places, count, then write.
function _headtailmerge!(aux_terms, aux_coeffs, main_terms, main_coeffs, n_old::Int,
    tail_terms, tail_coeffs, n_tail::Int, truncfunc, thread::Bool)

    task_partitioner, n_tasks = PropagationBase._preparetasks(n_old, thread)

    if n_tasks == 1
        return PropagationBase._tailmerge_write!(aux_terms, aux_coeffs, 1,
            main_terms, main_coeffs, 1, n_old, tail_terms, tail_coeffs, 1, n_tail, truncfunc, Val(true))
    end

    tail_bounds = Vector{Int}(undef, n_tasks + 1)
    tail_bounds[1] = 1
    tail_bounds[n_tasks+1] = n_tail + 1
    @inbounds for task_id in 1:(n_tasks-1)
        boundary_term = main_terms[task_partitioner[task_id].stop]
        tail_bounds[task_id+1] = searchsortedlast(tail_terms, boundary_term) + 1
    end

    counts = Vector{Int}(undef, n_tasks)
    AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
        head_range = task_partitioner[task_id]
        counts[task_id] = PropagationBase._tailmerge_write!(aux_terms, aux_coeffs, 1,
            main_terms, main_coeffs, head_range.start, head_range.stop,
            tail_terms, tail_coeffs, tail_bounds[task_id], tail_bounds[task_id+1] - 1, truncfunc, Val(false))
    end

    offsets = PropagationBase._offsetsfromcounts(counts)

    AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
        head_range = task_partitioner[task_id]
        PropagationBase._tailmerge_write!(aux_terms, aux_coeffs, offsets[task_id],
            main_terms, main_coeffs, head_range.start, head_range.stop,
            tail_terms, tail_coeffs, tail_bounds[task_id], tail_bounds[task_id+1] - 1, truncfunc, Val(true))
    end

    return offsets[end] - 1
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
        if plan !== nothing && _radixpays(plan, n_tail, sizeof(eltype(PauliPropagation.terms(mainsum(prop_cache)))))
            return _radixtailmerge!(prop_cache, plan; thread, truncfunc)
        end
    end
    return PauliPropagation.merge!(prop_cache; thread, truncfunc, kwargs...)
end
