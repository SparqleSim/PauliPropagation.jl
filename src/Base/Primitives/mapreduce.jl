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
