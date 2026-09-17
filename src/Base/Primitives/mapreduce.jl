"""
    mapreduce(mapper, reducer, term_sum::AbstractTermSum; init, neutral, thread=true)
    mapreduce(mapper, reducer, prop_cache::AbstractPropagationCache; init, neutral, thread=true)

Reduce `mapper(term, coefficient)` with `reducer` over the `(term, coefficient)` pairs of `term_sum`,
or over the active pairs of `prop_cache`. `init` defaults to the zero of the real coefficient type.
`neutral` is the identity used to reduce independent chunks. It is inferred for the standard
reductions supported by `AcceleratedKernels`; pass it explicitly for a custom reducer. `init` is
applied exactly once, after those chunks have been combined.
"""
function Base.mapreduce(mapper, reducer, thing::Union{AbstractTermSum,AbstractPropagationCache};
    init=zero(real(numcoefftype(thing))), neutral=_DEFAULT_REDUCTION_NEUTRAL, thread::Bool=true)

    mappedtype = Base.promote_op(mapper, termtype(thing), coefftype(thing))
    neutral = _reductionneutral(neutral, reducer, mappedtype)
    return _mapreduce(StorageType(thing), mapper, reducer, thing; init, neutral, thread)
end

function _mapreduce(::DictStorage, mapper::F, reducer::O, thing; init, neutral, thread::Bool) where {F,O}
    accumulated = init
    for (term, coefficient) in thing
        accumulated = reducer(accumulated, @inline mapper(term, coefficient))
    end
    return accumulated
end

# The pairs are reduced by index, so the backend of the arrays has to be named.
function _mapreduce(::ArrayStorage, mapper::F, reducer::O, thing; init, neutral, thread::Bool) where {F,O}
    active_terms = terms(thing)
    active_coefficients = coefficients(thing)

    if _iscpuarray(active_terms)
        return _mapreducecpu(mapper, reducer, active_terms, active_coefficients; init, neutral, thread)
    end

    map_index(index) = @inbounds @inline mapper(active_terms[index], active_coefficients[index])
    return AK.mapreduce(map_index, reducer, eachindex(active_coefficients), AK.get_backend(active_coefficients);
        init, neutral, max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
end

# Every task reduces its part from the reducer's identity on the workers of the propagation in progress.
function _mapreducecpu(mapper::F, reducer::O, terms, coefficients; init, neutral, thread::Bool) where {F,O}
    task_partitioner, n_tasks = _preparetasks(length(terms), thread)
    mappedtype = Base.promote_op(mapper, eltype(terms), eltype(coefficients))
    partialtype = Base.promote_op(reducer, typeof(neutral), mappedtype)
    partials = Vector{partialtype}(undef, n_tasks)

    function reduce_chunk!(task_id)
        chunk = task_partitioner[task_id]
        partials[task_id] = _mapreducerange(mapper, reducer, terms, coefficients, chunk.start, chunk.stop, neutral)
    end
    _eachtask(reduce_chunk!, n_tasks)

    return reduce(reducer, partials; init)
end

@inline function _mapreducerange(mapper::F, reducer::O, terms, coefficients, lo::Int, hi::Int, accumulated) where {F,O}
    @inbounds for ii in lo:hi
        accumulated = reducer(accumulated, @inline mapper(terms[ii], coefficients[ii]))
    end
    return accumulated
end


"""
    mapreducecoeffs(mapper, reducer, term_sum::AbstractTermSum; init, neutral, thread=true)
    mapreducecoeffs(mapper, reducer, prop_cache::AbstractPropagationCache; init, neutral, thread=true)

Reduce `mapper(coefficient)` with `reducer` over the coefficients of `term_sum`, or over the active
coefficients of `prop_cache`. `init` defaults to the zero of the real coefficient type.
`neutral` follows the same rules as [`mapreduce`](@ref): it is the identity used for independent
chunks, while `init` is applied once to the completed reduction.
"""
function mapreducecoeffs(mapper, reducer, thing::Union{AbstractTermSum,AbstractPropagationCache};
    init=zero(real(numcoefftype(thing))), neutral=_DEFAULT_REDUCTION_NEUTRAL, thread::Bool=true)

    mappedtype = Base.promote_op(mapper, coefftype(thing))
    neutral = _reductionneutral(neutral, reducer, mappedtype)
    return _mapreducecoeffs(StorageType(thing), mapper, reducer, thing; init, neutral, thread)
end

_mapreducecoeffs(::DictStorage, mapper, reducer, thing; init, neutral, thread::Bool) =
    mapreduce(mapper, reducer, coefficients(thing); init)

# `AcceleratedKernels` handles CPU and accelerator arrays here; both need the same chunk identity.
_mapreducecoeffs(::ArrayStorage, mapper, reducer, thing; init, neutral, thread::Bool) =
    AK.mapreduce(mapper, reducer, coefficients(thing); init, neutral,
        max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)


function _mapreduce(::MultiSumStorage, f::F, op::O, msum::AbstractTermSum; init, neutral, thread::Bool) where {F,O}
    mappedtype = Base.promote_op(f, termtype(msum), coefftype(msum))
    zonevaluetype = Base.promote_op(op, typeof(neutral), mappedtype)
    reduce_zone(zone) = mapreduce(f, op, zone; init=neutral, neutral, thread=false)
    return reduce(op, _zonevalues(reduce_zone, zonevaluetype, msum, thread); init)
end

function _mapreduce(::MultiSumStorage, f::F, op::O, prop_cache::AbstractPropagationCache; init, neutral, thread::Bool) where {F,O}
    mappedtype = Base.promote_op(f, termtype(prop_cache), coefftype(prop_cache))
    zonevaluetype = Base.promote_op(op, typeof(neutral), mappedtype)
    reduce_zone(zonecache) = mapreduce(f, op, zonecache; init=neutral, neutral, thread=false)
    return reduce(op, _zonevalues(reduce_zone, zonevaluetype, prop_cache, thread); init)
end

function _mapreducecoeffs(::MultiSumStorage, f::F, op::O, msum::AbstractTermSum; init, neutral, thread::Bool) where {F,O}
    mappedtype = Base.promote_op(f, coefftype(msum))
    zonevaluetype = Base.promote_op(op, typeof(neutral), mappedtype)
    reduce_zone(zone) = mapreducecoeffs(f, op, zone; init=neutral, neutral, thread=false)
    return reduce(op, _zonevalues(reduce_zone, zonevaluetype, msum, thread); init)
end

function _mapreducecoeffs(::MultiSumStorage, f::F, op::O, prop_cache::AbstractPropagationCache; init, neutral, thread::Bool) where {F,O}
    mappedtype = Base.promote_op(f, coefftype(prop_cache))
    zonevaluetype = Base.promote_op(op, typeof(neutral), mappedtype)
    reduce_zone(zonecache) = mapreducecoeffs(f, op, zonecache; init=neutral, neutral, thread=false)
    return reduce(op, _zonevalues(reduce_zone, zonevaluetype, prop_cache, thread); init)
end

# Each zone is reduced on its own thread, with no tasks started inside a zone.
function _zonevalues(zonefunc::F, ::Type{T}, msum::AbstractTermSum, thread::Bool) where {F,T}
    values = Vector{T}(undef, nzones(msum))
    store_zone_value!(zone_id) = (values[zone_id] = zonefunc(zones(msum)[zone_id]))
    _eachzone(store_zone_value!, msum, thread)
    return values
end

# One value of type `T` per zone, each computed on the zone's own thread.
function _zonevalues(zonefunc::F, ::Type{T}, prop_cache::AbstractPropagationCache, thread::Bool) where {F,T}
    values = Vector{T}(undef, nzones(prop_cache))
    store_zone_value!(zone_id) = (values[zone_id] = zonefunc(zonecaches(prop_cache)[zone_id]))
    _eachzone(store_zone_value!, prop_cache, thread)
    return values
end


struct _DefaultReductionNeutral end
const _DEFAULT_REDUCTION_NEUTRAL = _DefaultReductionNeutral()

# AcceleratedKernels supplies the identities required to reduce chunks, including typemax(T) for
# min and typemin(T) for max. Its fallback asks callers to provide a neutral for a custom reducer.
_reductionneutral(::_DefaultReductionNeutral, reducer, mappedtype) =
    AK.neutral_element(reducer, mappedtype)

_reductionneutral(neutral, reducer, mappedtype) = neutral
