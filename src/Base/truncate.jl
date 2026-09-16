function Base.truncate(term_sum::AbstractTermSum; min_abs_coeff::Real=eps(), customtruncfunc::F=_alwaysfalse, kwargs...) where F<:Function
    return truncate!(deepcopy(term_sum); min_abs_coeff, customtruncfunc, kwargs...)
end

function truncate!(term_sum::AbstractTermSum; min_abs_coeff::Real=eps(), customtruncfunc::F=_alwaysfalse, kwargs...) where F<:Function
    # bundle truncation functions 
    truncfunc = (pstr, coeff) -> truncatemincoeff(coeff, min_abs_coeff) || customtruncfunc(pstr, coeff)

    truncate!(truncfunc, term_sum; min_abs_coeff, customtruncfunc, kwargs...)

    return term_sum
end

function truncate!(prop_cache::AbstractPropagationCache; min_abs_coeff::Real=eps(), customtruncfunc::F=_alwaysfalse, kwargs...) where F<:Function
    # bundle truncation functions 
    truncfunc = (pstr, coeff) -> truncatemincoeff(coeff, min_abs_coeff) || customtruncfunc(pstr, coeff)

    prop_cache = truncate!(truncfunc, prop_cache; kwargs...)
    return prop_cache
end

# this can can be a term sum or a propagation cache
function truncate!(truncfunc::F, term_sum::Union{AbstractTermSum,AbstractPropagationCache}; kwargs...) where F<:Function
    return _truncate!(StorageType(term_sum), truncfunc, term_sum; kwargs...)
end


function _truncate!(::DictStorage, truncfunc::F, prop_cache::AbstractPropagationCache; kwargs...) where F<:Function
    term_sum = mainsum(prop_cache)
    term_sum = _truncate!(StorageType(term_sum), truncfunc, term_sum; kwargs...)
    setmainsum!(prop_cache, term_sum)
    return prop_cache
end

function _truncate!(::DictStorage, truncfunc::F, term_sum::AbstractTermSum; kwargs...) where F<:Function
    filter!(_invertfunc(truncfunc), storage(term_sum))
    return term_sum
end

function _truncate!(::ArrayStorage, truncfunc::F, prop_cache::AbstractPropagationCache; thread::Bool=true, kwargs...) where F<:Function

    if isempty(prop_cache)
        return prop_cache
    end

    # flag the indices that we keep
    keepfunc(pstr, coeff) = !truncfunc(pstr, coeff)
    flag!(keepfunc, prop_cache; thread)

    filterviaflags!(prop_cache; thread)

    return prop_cache
end

function _truncate!(::ArrayStorage, truncfunc::F, term_sum::AbstractTermSum; kwargs...) where F<:Function
    # convert to propagation cache for easier handling
    prop_cache = PropagationCache(term_sum)

    prop_cache = _truncate!(ArrayStorage(), truncfunc, prop_cache; kwargs...)

    # extracts the original input term sum
    return extractsum!(prop_cache, term_sum)
end


"""
    mapreducecoeffs(f, op, term_sum::AbstractTermSum; init, thread=true)
    mapreducecoeffs(f, op, prop_cache::AbstractPropagationCache; init, thread=true)

Reduce `f(coeff)` with `op` over the coefficients of `term_sum`, or of the active view of `prop_cache`, like `mapreduce(f, op, coefficients(thing); init)`.
`init` defaults to the zero of the real coefficient type. `thread=false` reduces on the calling thread alone.
"""
function mapreducecoeffs(f::F, op::O, thing::Union{AbstractTermSum,AbstractPropagationCache}; init=zero(real(numcoefftype(thing))), thread::Bool=true) where {F,O}
    return _mapreducecoeffs(StorageType(thing), f, op, thing; init, thread)
end

_mapreducecoeffs(::DictStorage, f::F, op::O, thing; init, thread::Bool) where {F,O} =
    mapreduce(f, op, coefficients(thing); init)

# every task starts from the zero of `init`, which serves `+` and, over non-negative values, `max`
_mapreducecoeffs(::ArrayStorage, f::F, op::O, thing; init, thread::Bool) where {F,O} =
    AK.mapreduce(f, op, coefficients(thing); init, neutral=zero(init), max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)

"""
    maxabscoeff(term_sum::AbstractTermSum; thread=true)
    maxabscoeff(prop_cache::AbstractPropagationCache; thread=true)

Returns the maximum absolute coefficient currently present in `term_sum`, or in the active
view of `prop_cache`. `thread=false` reduces on the calling thread alone.
"""
maxabscoeff(thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    mapreducecoeffs(coeff -> abs(tonumber(coeff)), max, thing; thread)


# Truncations on unsuitable coefficient types defaults to false.
function truncatemincoeff(coeff, min_abs_coeff)
    return false
end


# This should work for any complex and real coefficient
function truncatemincoeff(coeff::Number, min_abs_coeff::Real)
    return abs(coeff) < min_abs_coeff
end


_alwaysfalse(::Any...) = false
