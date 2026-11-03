"""
    maxabscoeff(term_sum::AbstractTermSum; thread=true)
    maxabscoeff(prop_cache::AbstractPropagationCache; thread=true)

Return the largest absolute coefficient in `term_sum`, or in the active view of `prop_cache`.
"""
maxabscoeff(thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    mapreducecoeffs(coefficient -> abs(tonumber(coefficient)), max, thing;
        neutral=zero(real(numcoefftype(thing))), thread)


"""
    topkabscoeff(term_sum::AbstractTermSum, k; rtol=0.05, thread=true)
    topkabscoeff(prop_cache::AbstractPropagationCache, k; rtol=0.05, thread=true)

A threshold `t` such that about `k` active coefficients satisfy `abs(coeff) >= t`, within `rtol * k` of `k`.
It is found by radix select on the bits of the magnitudes, never by sorting: a pass counts them into 65536
buckets, and the bucket holding the `k`-th largest is split by further passes on the next 16 bits until keeping
or dropping it lands within `rtol * k` of `k`, at most four passes in all. `rtol=0` is exact up to ties:
coefficients of equal magnitude are kept or dropped together, whichever count is closer to `k`.
Returns zero when `k` is at least the number of terms.
"""
function topkabscoeff(thing::Union{AbstractTermSum,AbstractPropagationCache}, k::Integer; rtol::Real=0.05, thread::Bool=true)
    k >= 1 || throw(ArgumentError("k must be at least 1, got $k"))
    rtol >= 0 || throw(ArgumentError("rtol must be non-negative, got $rtol"))
    k >= length(thing) && return 0.0

    prefix = UInt64(0)   # the bits above `shift + 16` of the bucket being split
    above = 0            # terms in buckets above the one being split
    for shift in (48, 32, 16, 0)
        counts = _magnitudehistogram(StorageType(thing), thing, prefix, shift, thread)
        for key in (length(counts)-1):-1:0
            with = above + counts[key+1]
            if with < k
                above = with
                continue
            end
            # the whole cutoff bucket is kept or dropped, whichever lands closer to k (ties keep it), once
            # that lands within the tolerance; after the last pass the bucket is a genuine tie
            keep_error = with - k
            drop_error = k - above
            if min(keep_error, drop_error) <= rtol * k || shift == 0
                edge = keep_error <= drop_error ? key : key + 1
                return reinterpret(Float64, ((prefix << 16) + UInt64(edge)) << shift)
            end
            prefix = (prefix << 16) | UInt64(key)
            break
        end
    end
    return 0.0   # unreachable: the bucket being split always holds at least k - above terms
end

# the bits of a non-negative float are monotone in its value
_magnitudebits(coefficient) = reinterpret(UInt64, Float64(abs(tonumber(coefficient))))

# every storage but a multi sum counts its own coefficients
_magnitudehistogram(::StorageType, thing, prefix::UInt64, shift::Int, thread::Bool) =
    _countmagnitudes(coefficients(thing), prefix, shift, thread)

# a multi sum counts each zone on the zone's own thread, with no tasks inside a zone, and adds the histograms
function _magnitudehistogram(::MultiSumStorage, thing, prefix::UInt64, shift::Int, thread::Bool)
    count_zone(zone) = _magnitudehistogram(StorageType(zone), zone, prefix, shift, false)
    return sum(_zonevalues(count_zone, Vector{Int}, thing, thread))
end

# Counts the coefficients whose magnitude bits above `shift + 16` equal `prefix` into 2^16 buckets by the
# 16 bits at `shift`. On the first pass `prefix` is 0 and matches everything, since `x >> 64 == 0`.
function _countmagnitudes(coeffs, prefix::UInt64, shift::Int, thread::Bool)
    function count_into!(counts, coefficients)
        @inbounds for coefficient in coefficients
            bits = _magnitudebits(coefficient)
            if bits >> (shift + 16) == prefix
                counts[Int((bits>>shift)&0xFFFF)+1] += 1
            end
        end
        return counts
    end

    # dictionary values are iterated by one task
    if !(coeffs isa AbstractArray)
        return count_into!(zeros(Int, 2^16), coeffs)
    end

    # ponytail: CPU only; a GPU array needs an AK/atomic histogram or a host copy
    _iscpuarray(coeffs) || throw(ArgumentError("topkabscoeff is only supported for CPU arrays, got $(typeof(coeffs))"))
    task_partitioner, n_tasks = _preparetasks(length(coeffs), thread)
    counts = zeros(Int, 2^16, n_tasks)
    count_chunk!(task_id) = count_into!(view(counts, :, task_id), view(coeffs, task_partitioner[task_id]))
    _eachtask(count_chunk!, n_tasks)
    return vec(sum(counts; dims=2))
end
