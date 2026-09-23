"""
    truncate(term_sum::AbstractTermSum; min_abs_coeff=eps(), customtruncfunc=nothing, thread=true)
    truncate(prop_cache::AbstractPropagationCache; min_abs_coeff=eps(), customtruncfunc=nothing, thread=true)

Drop the active `(term, coefficient)` pairs whose coefficient is smaller than `min_abs_coeff` in
absolute value, or for which `customtruncfunc(term, coefficient)` returns `true`, returning a copy.
"""
Base.truncate(thing::Union{AbstractTermSum,AbstractPropagationCache}; kwargs...) =
    truncate!(deepcopy(thing); kwargs...)

"""
    truncate!(term_sum::AbstractTermSum; min_abs_coeff=eps(), customtruncfunc=nothing, thread=true)
    truncate!(prop_cache::AbstractPropagationCache; min_abs_coeff=eps(), customtruncfunc=nothing, thread=true)
    truncate!(truncfunc, term_sum::AbstractTermSum; thread=true)
    truncate!(truncfunc, prop_cache::AbstractPropagationCache; thread=true)

Drop the active `(term, coefficient)` pairs whose coefficient is smaller than `min_abs_coeff` in
absolute value, or for which `customtruncfunc(term, coefficient)` returns `true`.
Given `truncfunc` directly, drop the pairs for which `truncfunc(term, coefficient)` returns `true`,
which is `filter!` with the test inverted.
"""
function truncate!(thing::Union{AbstractTermSum,AbstractPropagationCache};
    min_abs_coeff::Real=eps(), customtruncfunc::C=nothing, kwargs...) where {C}

    truncfunc(term, coefficient) = truncatemincoeff(coefficient, min_abs_coeff) ||
        (customtruncfunc !== nothing && customtruncfunc(term, coefficient))

    return truncate!(truncfunc, thing; kwargs...)
end

function truncate!(truncfunc::F, thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true, kwargs...) where {F}
    keep(term, coefficient) = !truncfunc(term, coefficient)
    return filter!(keep, thing; thread)
end

# `truncate!` by a truncation function that may be `nothing`, which the merges take for no truncation
_truncate!(truncfunc::F, thing; thread::Bool=true) where {F} = truncate!(truncfunc, thing; thread)
_truncate!(::Nothing, thing; thread::Bool=true) = thing

# Truncations on unsuitable coefficient types default to false.
truncatemincoeff(coeff, min_abs_coeff::Real) = false
truncatemincoeff(coeff::Number, min_abs_coeff::Real) = abs(coeff) < min_abs_coeff
