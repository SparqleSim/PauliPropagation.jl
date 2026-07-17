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

    # capture before filterviaflags!() swaps which sum is "main"
    old_sorted = sortedprefix(mainsum(prop_cache))

    # flag the indices that we keep
    keepfunc(pstr, coeff) = !truncfunc(pstr, coeff)
    flag!(keepfunc, prop_cache; thread)

    filterviaflags!(prop_cache; thread)

    # filtering keeps relative order, so however many of the old sorted terms survived is exactly
    # how many new leading terms are still sorted -- that count is already sitting in indices(...)
    new_sorted = old_sorted == 0 ? 0 : indices(prop_cache)[old_sorted]
    setsortedprefix!(mainsum(prop_cache), new_sorted)

    return prop_cache
end

function _truncate!(::ArrayStorage, truncfunc::F, term_sum::AbstractTermSum; kwargs...) where F<:Function
    # convert to propagation cache for easier handling
    prop_cache = PropagationCache(term_sum)

    prop_cache = _truncate!(ArrayStorage(), truncfunc, prop_cache; kwargs...)

    # extracts the original input term sum
    return extractsum!(prop_cache, term_sum)
end


# Truncations on unsuitable coefficient types defaults to false.
function truncatemincoeff(coeff, min_abs_coeff)
    return false
end


# This should work for any complex and real coefficient
function truncatemincoeff(coeff::Number, min_abs_coeff::Real)
    return abs(coeff) < min_abs_coeff
end


_alwaysfalse(::Any...) = false
