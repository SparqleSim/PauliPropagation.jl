"""
    mapreduce(mapper, reducer, term_sum::AbstractTermSum; init, thread=true)
    mapreduce(mapper, reducer, prop_cache::AbstractPropagationCache; init, thread=true)

Reduce `mapper(term, coefficient)` with `reducer` over the `(term, coefficient)` pairs of `term_sum`,
or over the active pairs of `prop_cache`. `init` defaults to the zero of the real coefficient type.
"""
function Base.mapreduce(mapper, reducer, thing::Union{AbstractTermSum,AbstractPropagationCache};
    init=zero(real(numcoefftype(thing))), thread::Bool=true)

    return _mapreduce(StorageType(thing), mapper, reducer, thing; init, thread)
end

function _mapreduce(::DictStorage, mapper::F, reducer::O, thing; init, thread::Bool) where {F,O}
    accumulated = init
    for (term, coefficient) in thing
        accumulated = reducer(accumulated, @inline mapper(term, coefficient))
    end
    return accumulated
end

# The pairs are reduced by index, so the backend of the arrays has to be named.
function _mapreduce(::ArrayStorage, mapper::F, reducer::O, thing; init, thread::Bool) where {F,O}
    active_terms = terms(thing)
    active_coefficients = coefficients(thing)

    if _iscpuarray(active_terms)
        return _mapreducecpu(mapper, reducer, active_terms, active_coefficients; init, thread)
    end

    map_index(index) = @inbounds @inline mapper(active_terms[index], active_coefficients[index])
    return AK.mapreduce(map_index, reducer, eachindex(active_coefficients), AK.get_backend(active_coefficients);
        init, neutral=zero(init), max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
end

# Every task reduces its part from `zero(init)` on the workers of the propagation in progress.
function _mapreducecpu(mapper::F, reducer::O, terms, coefficients; init, thread::Bool) where {F,O}
    task_partitioner, n_tasks = _preparetasks(length(terms), thread)
    partials = Vector{typeof(init)}(undef, n_tasks)

    function reduce_chunk!(task_id)
        chunk = task_partitioner[task_id]
        partials[task_id] = _mapreducerange(mapper, reducer, terms, coefficients, chunk.start, chunk.stop, zero(init))
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
    mapreducecoeffs(mapper, reducer, term_sum::AbstractTermSum; init, thread=true)
    mapreducecoeffs(mapper, reducer, prop_cache::AbstractPropagationCache; init, thread=true)

Reduce `mapper(coefficient)` with `reducer` over the coefficients of `term_sum`, or over the active
coefficients of `prop_cache`. `init` defaults to the zero of the real coefficient type.
"""
function mapreducecoeffs(mapper, reducer, thing::Union{AbstractTermSum,AbstractPropagationCache};
    init=zero(real(numcoefftype(thing))), thread::Bool=true)

    return _mapreducecoeffs(StorageType(thing), mapper, reducer, thing; init, thread)
end

_mapreducecoeffs(::DictStorage, mapper, reducer, thing; init, thread::Bool) =
    mapreduce(mapper, reducer, coefficients(thing); init)

# Every task starts from `zero(init)`, which serves addition and maximum reductions over
# non-negative mapped coefficients.
_mapreducecoeffs(::ArrayStorage, mapper, reducer, thing; init, thread::Bool) =
    AK.mapreduce(mapper, reducer, coefficients(thing); init, neutral=zero(init),
        max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
