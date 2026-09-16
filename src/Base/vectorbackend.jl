###
##
# Array-backed propagation-cache implementation helpers.
##
###


# Writes `(term, coefficient)` at `position` when `DoWrite`, and advances `position` in both the
# dry run and the writing pass.
@inline function _writeandadvance!(output_terms, output_coefficients, position, term, coefficient, ::Val{DoWrite}) where DoWrite
    if DoWrite
        @inbounds output_terms[position] = term
        @inbounds output_coefficients[position] = coefficient
    end
    return position + 1
end


# Flagging and prefix scans support branching gates and array-backed filtering.
function flag!(predicate, prop_cache::AbstractPropagationCache; thread::Bool=true)
    flag!(predicate, activeflags(prop_cache), activeterms(prop_cache), activecoeffs(prop_cache); thread)
    return prop_cache
end

function flag!(predicate, destination_flags, source_terms, source_coefficients; thread::Bool=true)
    @assert length(destination_flags) <= length(source_terms)
    @assert length(destination_flags) <= length(source_coefficients)

    AK.foreachindex(destination_flags; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do index
        destination_flags[index] = predicate(source_terms[index], source_coefficients[index])
    end
    return destination_flags
end

function flagterms!(predicate, prop_cache::AbstractPropagationCache; thread::Bool=true)
    flagterms!(predicate, activeflags(prop_cache), activeterms(prop_cache); thread)
    return prop_cache
end

function flagterms!(predicate, destination_flags, source_terms; thread::Bool=true)
    @assert length(destination_flags) <= length(source_terms)

    AK.foreachindex(destination_flags; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do index
        destination_flags[index] = predicate(source_terms[index])
    end
    return destination_flags
end

function flagcoeffs!(predicate, prop_cache::AbstractPropagationCache; thread::Bool=true)
    flagcoeffs!(predicate, activeflags(prop_cache), activecoeffs(prop_cache); thread)
    return prop_cache
end

function flagcoeffs!(predicate, destination_flags, source_coefficients; thread::Bool=true)
    @assert length(destination_flags) <= length(source_coefficients)

    AK.foreachindex(destination_flags; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do index
        destination_flags[index] = predicate(source_coefficients[index])
    end
    return destination_flags
end

flagstoindices!(prop_cache::AbstractPropagationCache; thread::Bool=true) =
    flagstoindices!(activeindices(prop_cache), activeflags(prop_cache); thread)

flagstoindices!(destination_indices, source_flags; thread::Bool=true) =
    AK.accumulate!(+, destination_indices, source_flags; init=zero(eltype(destination_indices)),
        max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)


# Permutation and compaction use the auxiliary term sum as their destination buffer.
function permuteviaindices!(prop_cache::AbstractPropagationCache; thread::Bool=true)
    input_terms = activeterms(prop_cache)
    input_coefficients = activecoeffs(prop_cache)
    output_terms = activeauxterms(prop_cache)
    output_coefficients = activeauxcoeffs(prop_cache)
    active_permutation = activeindices(prop_cache)

    permuteviaindices!(output_terms, output_coefficients, input_terms, input_coefficients, active_permutation; thread)

    swapsums!(prop_cache)
    setsortedprefix!(mainsum(prop_cache), 0)
    return prop_cache
end

function permuteviaindices!(output_terms, output_coefficients, input_terms, input_coefficients, permutation; thread::Bool=true)
    @assert length(permutation) <= length(input_terms) && length(permutation) <= length(input_coefficients)
    @assert length(permutation) <= length(output_terms) && length(permutation) <= length(output_coefficients)

    AK.foreachindex(permutation; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do index
        input_index = permutation[index]
        output_terms[index] = input_terms[input_index]
        output_coefficients[index] = input_coefficients[input_index]
    end
    return output_terms, output_coefficients
end

function filterviaflags!(prop_cache::AbstractPropagationCache; thread::Bool=true)
    input_terms = activeterms(prop_cache)
    input_coefficients = activecoeffs(prop_cache)
    output_terms = activeauxterms(prop_cache)
    output_coefficients = activeauxcoeffs(prop_cache)
    active_flags = activeflags(prop_cache)
    active_indices = activeindices(prop_cache)

    old_sorted_prefix = sortedprefix(mainsum(prop_cache))
    old_sorted_prefix > length(active_indices) && (old_sorted_prefix = 0)

    filterviaflags!(active_flags, active_indices, output_terms, output_coefficients, input_terms, input_coefficients; thread)

    swapsums!(prop_cache)
    setactivesize!(prop_cache, lastactiveindex(prop_cache))

    new_sorted_prefix = old_sorted_prefix == 0 ? 0 : active_indices[old_sorted_prefix]
    setsortedprefix!(mainsum(prop_cache), new_sorted_prefix)
    return prop_cache
end

function filterviaflags!(source_flags, destination_indices, output_terms, output_coefficients,
    input_terms, input_coefficients; thread::Bool=true)

    @assert length(source_flags) <= length(input_terms) && length(source_flags) <= length(input_coefficients)
    @assert length(source_flags) <= length(output_terms) && length(source_flags) <= length(output_coefficients)

    flagstoindices!(destination_indices, source_flags; thread)

    AK.foreachindex(source_flags; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do index
        if source_flags[index]
            output_terms[destination_indices[index]] = input_terms[index]
            output_coefficients[destination_indices[index]] = input_coefficients[index]
        end
    end
    return output_terms, output_coefficients
end

function _copy!(output_terms, output_coefficients, input_terms, input_coefficients; thread::Bool=true)
    @assert length(output_terms) >= length(input_terms)
    @assert length(output_coefficients) >= length(input_coefficients)

    AK.foreachindex(input_terms; max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK) do index
        output_terms[index] = input_terms[index]
        output_coefficients[index] = input_coefficients[index]
    end
    return output_terms, output_coefficients
end


# Coefficient cumulative sums serve the resampling implementation.
coeffcumsum!(coefficients; thread::Bool=true) =
    AK.accumulate!((left, right) -> left + abs(right), coefficients; init=zero(eltype(coefficients)),
        neutral=zero(eltype(coefficients)), max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)

coeffcumsum(coefficients; thread::Bool=true) = coeffcumsum(coefficients, 1; thread)

function coeffcumsum(coefficients, power::Real; thread::Bool=true)
    # `abs(coefficient)^power` is real, while `AK.map!` retains the destination element type.
    cumulative_coefficients = similar(coefficients, real(eltype(coefficients)))
    AK.map!(coefficient -> abs(coefficient)^power, cumulative_coefficients, coefficients;
        max_tasks=maxtasks(thread), min_elems=_MIN_ELEMS_PER_TASK)
    coeffcumsum!(cumulative_coefficients; thread)
    return cumulative_coefficients
end
