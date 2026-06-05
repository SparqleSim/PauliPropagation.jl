
function sortbyterm!(prop_cache::AbstractPropagationCache; lt=isless, by=identity, rev=false, order=Base.Forward)

    # if terms are are not native data types, sorting kwargs need to be provided
    AK.sortperm!(activeindices(prop_cache), activeterms(prop_cache); lt, by, rev, order)

    permuteviaindices!(prop_cache)

    return prop_cache
end


# map function f will receive (term, coeff) and should return a bool
function flag!(f::F, prop_cache::AbstractPropagationCache) where F<:Function
    flag!(f, activeflags(prop_cache), activeterms(prop_cache), activecoeffs(prop_cache))
    return prop_cache
end

function flag!(f::F, dst_flags, terms, coeffs) where F<:Function
    @assert length(dst_flags) <= length(terms)
    @assert length(dst_flags) <= length(coeffs)

    AK.foreachindex(dst_flags) do ii
        dst_flags[ii] = f(terms[ii], coeffs[ii])
    end
    return dst_flags
end

# map function f will receive term only and should return a bool
function flagterms!(f::F, prop_cache::AbstractPropagationCache) where F<:Function
    flagterms!(f, activeflags(prop_cache), activeterms(prop_cache))
    return prop_cache
end

function flagterms!(f::F, dst_flags, terms) where F<:Function
    @assert length(dst_flags) <= length(terms)

    AK.foreachindex(dst_flags) do ii
        dst_flags[ii] = f(terms[ii])
    end
    return dst_flags
end

# map function f will receive coeff only and should return a bool
function flagcoeffs!(f::F, prop_cache::AbstractPropagationCache) where F<:Function
    flagcoeffs!(f, activeflags(prop_cache), activecoeffs(prop_cache))
    return prop_cache
end

function flagcoeffs!(f::F, dst_flags, coeffs) where F<:Function
    @assert length(dst_flags) <= length(coeffs)

    AK.foreachindex(dst_flags) do ii
        dst_flags[ii] = f(coeffs[ii])
    end
    return dst_flags
end

flagstoindices!(prop_cache::AbstractPropagationCache) = flagstoindices!(activeindices(prop_cache), activeflags(prop_cache))
flagstoindices!(dst_indices, flags) = AK.accumulate!(+, dst_indices, flags; init=zero(eltype(dst_indices)))

function permuteviaindices!(prop_cache::AbstractPropagationCache)
    indices_view = activeindices(prop_cache)
    term_view = activeterms(prop_cache)
    coeffs_view = activecoeffs(prop_cache)
    aux_terms_view = activeauxterms(prop_cache)
    aux_coeffs_view = activeauxcoeffs(prop_cache)

    permuteviaindices!(aux_terms_view, aux_coeffs_view, term_view, coeffs_view, indices_view)

    # the destination arrays should be the main ones 
    swapsums!(prop_cache)

    return prop_cache
end

function permuteviaindices!(dst_terms, dst_coeffs, src_terms, src_coeffs, indices)
    @assert length(indices) <= length(src_terms) && length(indices) <= length(src_coeffs)
    @assert length(indices) <= length(dst_terms) && length(indices) <= length(dst_coeffs)

    AK.foreachindex(indices) do ii
        sorted_idx = indices[ii]
        dst_terms[ii] = src_terms[sorted_idx]
        dst_coeffs[ii] = src_coeffs[sorted_idx]
    end
    return dst_terms, dst_coeffs
end


# what is flagged will be kept
# should we name this something with copy?
function filterviaflags!(prop_cache::AbstractPropagationCache)
    terms_view = activeterms(prop_cache)
    coeffs = activecoeffs(prop_cache)
    aux_terms = activeauxterms(prop_cache)
    aux_coeffs = activeauxcoeffs(prop_cache)
    flags = activeflags(prop_cache)
    indices = activeindices(prop_cache)

    filterviaflags!(flags, indices, aux_terms, aux_coeffs, terms_view, coeffs)

    swapsums!(prop_cache)

    n_new = lastactiveindex(prop_cache)
    setactivesize!(prop_cache, n_new)

    return prop_cache
end

# TODO: turn this into copyflagged!()
function filterviaflags!(flags, dst_indices, dst_terms, dst_coeffs, src_terms, src_coeffs)
    @assert length(flags) <= length(src_terms) && length(flags) <= length(src_coeffs)
    @assert length(flags) <= length(dst_terms) && length(flags) <= length(dst_coeffs)

    flagstoindices!(dst_indices, flags)

    AK.foreachindex(flags) do ii
        if flags[ii]
            dst_terms[dst_indices[ii]] = src_terms[ii]
            dst_coeffs[dst_indices[ii]] = src_coeffs[ii]
        end
    end
    return dst_terms, dst_coeffs
end


function _copy!(dst_terms, dst_coeffs, src_terms, src_coeffs)
    @assert length(dst_terms) >= length(src_terms)
    @assert length(dst_coeffs) >= length(src_coeffs)
    AK.foreachindex(src_terms) do ii
        dst_terms[ii] = src_terms[ii]
        dst_coeffs[ii] = src_coeffs[ii]
    end
    return dst_terms, dst_coeffs
end