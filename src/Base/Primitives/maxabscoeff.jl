"""
    maxabscoeff(term_sum::AbstractTermSum; thread=true)
    maxabscoeff(prop_cache::AbstractPropagationCache; thread=true)

Return the largest absolute coefficient in `term_sum`, or in the active view of `prop_cache`.
"""
maxabscoeff(thing::Union{AbstractTermSum,AbstractPropagationCache}; thread::Bool=true) =
    mapreducecoeffs(coefficient -> abs(tonumber(coefficient)), max, thing; thread)
