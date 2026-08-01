###
##
# Radix tail sort for the two branching rotation gates.
#
# A rotation appends its new terms as `pstr ⊻ gate_mask`, written in the order of the (sorted) terms
# they branched off. XOR by a fixed mask preserves the relative order of two terms unless the most
# significant bit in which they differ is a set bit of the mask, so the tail can be sorted by one
# radix pass per run of adjacent mask bits ("digit"), most significant first, at O(n_tail) per pass
# and with no comparisons at all: before a digit's pass the tail is ascending in
# `value ⊻ (mask bits at or below the digit)`, so within every group of terms agreeing on the bits
# above the digit, each digit value occupies one contiguous block.
#
# A digit is a run of *set* mask bits, so a term's digit is the bitwise complement of the digit its
# parent carried, and the order those blocks arrive in is exactly the reverse of the one they want:
# a pass is a block reversal within each group. That needs neither the value of a digit nor any
# bookkeeping per value, only equality of digits, so nothing about a pass depends on how wide its
# digit is -- a two-bit Y digit costs what a one-bit X digit costs.
#
# (MajoranaPropagation.jl sorts its tails the same way, one bit at a time.)
#
# A library sort cannot stand in for the passes. A global stable sort keyed on the flipped bits is
# wrong, LSD or MSD -- with mask `01`, parents `00`,`11` branch to `01`,`10`, which keyed on bit 0
# sorts to `10`,`01` -- because a pass is only valid inside a group already agreeing on every bit
# above its digit, and non-mask bits interleave above and between the digits. Keying on
# `v >> lowest_flipped_bit` is correct but is a full comparison sort with a shift per comparison,
# several times slower than a plain `sort!`.
#
# The premise holds only when every term the gate saw was sorted to begin with; the caller checks
# that before the gate is applied and falls back to the stock merge otherwise.
##
###

# Set this to false to route the fused rotations back through the stock `merge!` (for A/B timing).
const USE_RADIX_TAILSORT = Ref(true)

# A gate spanning more digits than this gets the stock path -- one pass each is no longer a win.
const _MAX_DIGITS = 4

# Tails shorter than this keep the stock sort, whose constant factor is lower than a pass setup.
const _MIN_RADIX_TAIL = 64

# Each pass moves every term once, whereas the stock path moves it once in total (its comparisons
# stop at the first differing word and stay cheap however wide the string is). So a second pass only
# pays for narrow terms, or for a tail small enough that it stays in cache: measured on 100k-term
# tails, two digits win 1.5x at 16 bytes per term and lose (0.6-0.75x) from 56 bytes up, while a
# single digit wins at every width.
const _MAX_RADIX_TERMBYTES = 32
const _MAX_RADIX_PASS_BYTES = 2 << 20

_radixpays(plan, n_tail::Int, termbytes::Int) =
    length(plan) == 1 || termbytes <= _MAX_RADIX_TERMBYTES || n_tail * termbytes <= _MAX_RADIX_PASS_BYTES


### Planning the passes

"""
    _radixplan(gate_mask, bits)

Group the set bits of `gate_mask`, listed ascending in `bits`, into runs of adjacent bits -- one
radix pass each, listed least significant first. Each run is kept as the two whole-string masks its
pass tests terms against: the run itself, and everything above it. Holding them that way keeps the
passes free of any per-Pauli-type knowledge; they need only `⊻`, `&` and `iszero`.

Returns `nothing` if there are no set bits, or more runs than the tail sort can pay for.
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

# The Pauli string with every bit from `lo` up set. A shift past the top of the string gives zero and
# so the empty mask, which is what a digit at the very top of the string wants.
_bitsfrom(::Type{TT}, lo::Int) where {TT} = ~((one(TT) << lo) - one(TT))


### The passes

# Copies src[s0 .+ (0:n-1)] onto dst[d0 .+ (0:n-1)]. Blocks are frequently singletons, where
# copyto!'s per-call overhead dominates the move itself.
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

# One pass: within each group of terms agreeing above the digit, re-emit the digit's blocks back to
# front (see file header). Both the group and its blocks are found by scanning for a change, so no
# digit is ever decoded and the number of values it can take never enters.
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

Sorts the parent-ordered tail at `main_terms[n_old+1:end]` by one radix pass per digit, using
`buf_terms`/`buf_coeffs` as scratch. Returns `true` if the sorted tail ended up in the scratch.
"""
function _radixsorttail!(plan, main_terms, main_coeffs, n_old::Int, n_tail::Int, buf_terms, buf_coeffs)
    # both slots are normalized to the same view type so that the ping-pong below stays type stable;
    # without that, every element access in a pass goes through a dynamic dispatch
    src_terms = view(main_terms, n_old+1:n_old+n_tail)
    src_coeffs = view(main_coeffs, n_old+1:n_old+n_tail)
    dst_terms = view(buf_terms, 1:n_tail)
    dst_coeffs = view(buf_coeffs, 1:n_tail)

    # most significant digit first; `_radixplan` lists them ascending
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

Counterpart of `sortedtailmerge!` that sorts the tail by `_radixsorttail!` instead of a comparison
sort. Only valid when the tail is in parent order (see file header).
"""
function _radixtailmerge!(prop_cache, plan; thread::Bool=true, truncfunc=nothing)
    n_old = sortedprefix(mainsum(prop_cache))
    n_new = activesize(prop_cache)
    n_tail = n_new - n_old
    if n_tail == 0
        return prop_cache
    end

    main_terms, main_coeffs, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(prop_cache)

    # the head/tail merge writes into aux[1:merged] <= n_new, so any capacity past n_new is free
    # scratch for the tail -- else allocate
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

# Two-pointer merge of main[1:n_old] against the sorted tail, into aux. Same task partitioning as
# `sortedtailmerge!`: split the head into task ranges, cut the tail at the same terms, size each
# task's output with a dry run, then write.
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

`merge!` for a sum a rotation gate just branched: takes the radix tail sort when the gate's new terms
are in parent order and the mask is local enough, and the stock `merge!` otherwise.
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
