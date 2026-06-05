tonumber(x::Number) = x

function _thrownotimplemented(::Type{T}, func_name::Symbol) where T
    error("Function '", func_name, "' not implemented for type '", T, "'.")
end

function _thrownotimplemented(::T, func_name::Symbol) where T
    _thrownotimplemented(T, func_name)
end