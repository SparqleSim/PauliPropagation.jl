###
##
# Truncation for the fused gate applications.
##
###

@inline function _fusedtruncfunc(pstr, coeff; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc)
    PauliPropagation.truncatemincoeff(coeff, min_abs_coeff) && return true
    _truncateweight(pstr, max_weight) && return true
    PauliPropagation.truncatefrequency(coeff, max_freq) && return true
    PauliPropagation.truncatesins(coeff, max_sins) && return true
    !isnothing(customtruncfunc) && customtruncfunc(pstr, coeff) && return true
    return false
end

# below this width the library's whole-string count is faster; measured, the two draw level here
const _MIN_WORD_BYTES = 256

"""
    _truncateweight(pstr, max_weight)

`PauliPropagation.truncateweight`, counting a very wide Pauli string one 64-bit word at a time. A
qubit's two bits never straddle a word boundary, so the words' weights just add, and nothing operates
on the whole string, which is what gets slow at these widths.

Narrower strings, and any width that is not a whole number of words, use the library version. The
test is on a type, so it settles at compile time.
"""
@inline _truncateweight(pstr, max_weight::Real) = PauliPropagation.truncateweight(pstr, max_weight)

@inline function _truncateweight(pstr::TT, max_weight::Real) where {TT<:Unsigned}
    if sizeof(TT) < _MIN_WORD_BYTES || !iszero(sizeof(TT) % 8)
        return PauliPropagation.truncateweight(pstr, max_weight)
    end
    isinf(max_weight) && return false
    return _wordweight(pstr) > max_weight
end

# kept out of line: every term then only pays the check for whether there is a weight limit at all
function _wordweight(pstr::TT) where {TT<:Unsigned}
    # ...0101: the low bit of every Pauli pair
    altword = 0x5555555555555555

    weight = 0
    for word in reinterpret(NTuple{sizeof(TT) ÷ 8,UInt64}, pstr)
        weight += count_ones((word | (word >> 1)) & altword)
    end
    return weight
end
