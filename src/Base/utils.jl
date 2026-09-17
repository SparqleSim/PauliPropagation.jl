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
_iscpuarray(::AbstractArray) = true

function _thrownotimplemented(::Type{T}, func_name::Symbol) where T
    error("Function '", func_name, "' not implemented for type '", T, "'.")
end

function _thrownotimplemented(::T, func_name::Symbol) where T
    _thrownotimplemented(T, func_name)
end