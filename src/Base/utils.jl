tonumber(x::Number) = x

# the positions of the set bits of `mask`, ascending
function _masksetbits(mask::TT) where {TT}
    bits = Int[]
    while !iszero(mask)
        push!(bits, trailing_zeros(mask))
        mask &= mask - one(TT)
    end
    return bits
end

# for CPU-only code using Threads.@spawn
# GPU extensions override this for their array for fallback functionality
# TODO: Use AK method to check whether an array is on GPU.
_iscpuarray(::AbstractArray) = true

# one element of an array, for host code that needs a value the device wrote. An
# array elsewhere than the CPU is copied from instead of indexed, so the read needs no scalar-indexing grant
# from any backend; a CPU array is indexed as before. A view is where its parent is.
function _readelement(array::AbstractArray, index::Int)
    _iscpuarray(parent(array)) && return array[index]
    element = Vector{eltype(array)}(undef, 1)
    copyto!(element, 1, array, index, 1)
    return element[1]
end

function _thrownotimplemented(::Type{T}, func_name::Symbol) where T
    error("Function '", func_name, "' not implemented for type '", T, "'.")
end

function _thrownotimplemented(::T, func_name::Symbol) where T
    _thrownotimplemented(T, func_name)
end