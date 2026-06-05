
# TermSum is an abstract type for container types carrying terms and coefficients
# we expect that it 
abstract type AbstractTermSum end

### Interface functions to be defined for normal use of TermSum types 


abstract type StorageType end
struct DictStorage <: StorageType end
struct ArrayStorage <: StorageType end


StorageType(term_sum::AbstractTermSum) = _storagetype(storage(term_sum))
# storage types for common data structures
_storagetype(::Dict) = PropagationBase.DictStorage()
_storagetype(::Tuple{AbstractArray,AbstractArray}) = PropagationBase.ArrayStorage()
_storagetype(x) = _thrownotimplemented(typeof(x), :StorageType)

# storage() is expected to return the internal storage representation of the TermSum
# often this is a Dict{TermType,CoeffType} but it can be anything
storage(term_sum::TS) where TS<:AbstractTermSum = _thrownotimplemented(TS, :storage)

Base.length(term_sum::AbstractTermSum) = length(terms(term_sum))

nsites(term_sum::TS) where TS<:AbstractTermSum = _thrownotimplemented(TS, :nsites)

terms(::StorageType, term_sum::TS) where TS<:AbstractTermSum = _thrownotimplemented(TS, :terms)
terms(term_sum::AbstractTermSum) = _terms(StorageType(term_sum), term_sum)
_terms(::DictStorage, term_sum::AbstractTermSum) = keys(storage(term_sum))
_terms(::ArrayStorage, term_sum::AbstractTermSum) = storage(term_sum)[1]


coefficients(::StorageType, term_sum::TS) where TS<:AbstractTermSum = _thrownotimplemented(TS, :coefficients)
coefficients(term_sum::AbstractTermSum) = _coefficients(StorageType(term_sum), term_sum)
_coefficients(::DictStorage, term_sum::AbstractTermSum) = values(storage(term_sum))
_coefficients(::ArrayStorage, term_sum::AbstractTermSum) = storage(term_sum)[2]
coeffs(term_sum::AbstractTermSum) = coefficients(term_sum)

# receives the object
termtype(term_sum::TS) where TS<:AbstractTermSum = eltype(terms(term_sum))
coefftype(term_sum::TS) where TS<:AbstractTermSum = eltype(coefficients(term_sum))

# this is used to determine type-stable return values for numerical operations
numcoefftype(term_sum::TS) where TS<:AbstractTermSum = numcoefftype(coefftype(term_sum))
numcoefftype(::Type{CT}) where CT<:Number = CT
numcoefftype(::Type{T}) where T = _thrownotimplemented(T, :numcoefftype)


function getcoeff(term_sum::AbstractTermSum, trm)
    return _getcoeff(StorageType(term_sum), term_sum, trm)
end

function _getcoeff(::DictStorage, term_sum::AbstractTermSum, trm)
    term_dict = storage(term_sum)
    return get(term_dict, trm, zero(coefftype(term_sum)))
end

# default implementation
function _getcoeff(::ST, term_sum::AbstractTermSum, trm) where {ST<:StorageType}
    # TODO: GPU kernel for this
    val = zero(coefftype(term_sum))
    for (term, coeff) in term_sum
        if term == trm
            val += coeff
        end
    end
    return val
end

# this assumes everything is merged and de-duplicated
# may result in wrong results if not
function getmergedcoeff(term_sum::AbstractTermSum, trm)
    return _getmergedcoeff(StorageType(term_sum), term_sum, trm)
end

# for the DictStorage, this is the same as getcoeff
function _getmergedcoeff(::DictStorage, term_sum::AbstractTermSum, trm)
    return _getcoeff(DictStorage(), term_sum, trm)
end

# everywhere else, we do a linear search
function _getmergedcoeff(::ArrayStorage, term_sum::AbstractTermSum, trm)
    terms_vec, coeffs_vec = storage(term_sum)
    i = findfirst(t -> t == trm, terms_vec)
    if isnothing(i)
        return zero(coefftype(term_sum))
    else
        return coeffs_vec[i]
    end
end


### Default functions defined for all TermSum types
@inline function Base.iterate(term_sum::AbstractTermSum, args...)
    return _iterate(StorageType(term_sum), term_sum, args...)
end

@inline function _iterate(::DictStorage, term_sum::AbstractTermSum, args...)
    dict_storage = storage(term_sum)
    return iterate(dict_storage, args...)
end


@inline function _iterate(::StorageType, term_sum::AbstractTermSum)
    # 1. Create the iterator we are delegating to
    iter = zip(terms(term_sum), coefficients(term_sum))

    # 2. Start its iteration
    next = iterate(iter)

    # 3. Return the first item and a new state tuple: (iterator, iterator_state)
    #    We use a ternary operator for compactness.
    return next === nothing ? nothing : (next[1], (iter, next[2]))
end

@inline function _iterate(::StorageType, term_sum::AbstractTermSum, state)
    # 1. Unpack the state tuple
    (iter, inner_state) = state

    # 2. Continue the delegated iteration
    next = iterate(iter, inner_state)

    # 3. Return the next item and the updated state tuple
    return next === nothing ? nothing : (next[1], (iter, next[2]))
end


"""
    norm(psum::AbstractTermSum, L=2)

Calculate the norm of a Pauli sum `psum` with respect to the `L`-norm. 
Calls `LinearAlgebra.norm(coefficients(psum))`.
If `psum` contains duplicate terms, the coefficients are NOT merged before hand
and the norm value will be affected.
"""
function LinearAlgebra.norm(psum::AbstractTermSum, L::Real=2)
    if length(psum) == 0
        return zero(numcoefftype(psum))
    end
    return LinearAlgebra.norm((tonumber(coeff) for coeff in coefficients(psum)), L)
end

# Copy from one AbstractTermSum to another
# They must be of the same typing
function Base.copy!(dst_term_sum::TS, src_term_sum::TS) where {TS<:AbstractTermSum}
    _copy!(StorageType(dst_term_sum), dst_term_sum, src_term_sum)
    return dst_term_sum
end

function _copy!(::StorageType, dst_term_sum::AbstractTermSum, src_term_sum::AbstractTermSum)
    copy!(storage(dst_term_sum), storage(src_term_sum))
    return dst_term_sum
end

function _copy!(::ArrayStorage, dst_term_sum::AbstractTermSum, src_term_sum::AbstractTermSum)
    dst_terms = terms(dst_term_sum)
    dst_coeffs = coefficients(dst_term_sum)
    src_terms = terms(src_term_sum)
    src_coeffs = coefficients(src_term_sum)

    # in vectorbackend.jl
    _copy!(dst_terms, dst_coeffs, src_terms, src_coeffs)

    resize!(dst_term_sum, length(src_term_sum))
    resize!(dst_terms, length(src_term_sum))

    return dst_term_sum
end

### Adding, setting, and deleting
"""
    add!(term_sum::AbstractTermSum, term, coeff)

Add `coeff` to the coefficient of `term` in `term_sum`.
Calls `_add!(StorageType(term_sum), term_sum, term, coeff)` internally.
For custom behavior, overload `storage()` and/or `_add!` for the specific TermSum type.
"""
@inline function add!(term_sum::AbstractTermSum, term, coeff)
    _add!(StorageType(term_sum), term_sum, term, coeff)
    return term_sum
end


@inline function _add!(::DictStorage, term_sum::AbstractTermSum, term, coeff)
    dict_storage = storage(term_sum)
    if haskey(dict_storage, term)
        dict_storage[term] += coeff
    else
        dict_storage[term] = coeff
    end
    return term_sum
end

@inline function _add!(::ArrayStorage, term_sum::AbstractTermSum, term, coeff)
    terms_vec, coeffs_vec = storage(term_sum)
    ind = findfirst(t -> t == term, terms_vec)
    if !isnothing(ind)
        coeffs_vec[ind] += coeff
    else
        push!(terms_vec, term)
        push!(coeffs_vec, coeff)
    end
    return term_sum
end

@inline function _add!(::StorageType, term_sum::AbstractTermSum, term, coeff)
    thrownotimplemented(typeof(term_sum), :_add!)
end


function add!(term_sum1::AbstractTermSum, term_sum2::AbstractTermSum)
    for (term, coeff) in term_sum2
        add!(term_sum1, term, coeff)
    end
    return term_sum1
end

function subtract!(term_sum1::AbstractTermSum, term_sum2::AbstractTermSum)
    for (term, coeff) in term_sum2
        add!(term_sum1, term, -coeff)
    end
    return term_sum1
end


"""
    set!(term_sum::AbstractTermSum, term, coeff)
    
Set the coefficient of `term` in `term_sum` to `coeff`.
Calls `_set!(StorageType(term_sum), term_sum, term, coeff)` internally.
For custom behavior, overload `storage()` and/or `_set!` for the specific TermSum type.
"""
@inline function set!(term_sum::AbstractTermSum, term, coeff)
    _set!(StorageType(term_sum), term_sum, term, coeff)
    return term_sum
end


@inline function _set!(::DictStorage, term_sum::AbstractTermSum, term, coeff)
    dict_storage = storage(term_sum)
    dict_storage[term] = coeff
    return term_sum
end

@inline function _set!(::ArrayStorage, term_sum::AbstractTermSum, term, coeff)
    terms_vec, coeffs_vec = storage(term_sum)
    ind = findfirst(t -> t == term, terms_vec)
    if !isnothing(ind)
        coeffs_vec[ind] = coeff
    else
        push!(terms_vec, term)
        push!(coeffs_vec, coeff)
    end
    return term_sum
end

@inline function _set!(ST::StorageType, term_sum::AbstractTermSum, term, coeff)
    old_coeff = getcoeff(term_sum, term)
    if old_coeff == zero(coefftype(term_sum))
        _add!(ST, term_sum, term, coeff)
    else
        delta = coeff - old_coeff
        _add!(ST, term_sum, term, delta)
    end
    return term_sum
end

"""
    mult!(term_sum::AbstractTermSum, scalar::Number)

Multiply all coefficients in `term_sum` by `scalar`.
Calls `mult!(StorageType(term_sum), term_sum, scalar)` internally.
For custom behavior, overload `storage()` and/or `mult!` for the specific TermSum type.
"""
function mult!(term_sum::AbstractTermSum, scalar::Number)
    return _mult!(StorageType(term_sum), term_sum, scalar)
end


function _mult!(::DictStorage, term_sum::AbstractTermSum, scalar::Number)
    dict_storage = storage(term_sum)
    for (term, coeff) in dict_storage
        dict_storage[term] = coeff * scalar
    end
    return term_sum
end

function _mult!(::ArrayStorage, term_sum::AbstractTermSum, scalar::Number)
    terms_vec, coeffs_vec = storage(term_sum)
    coeffs_vec .*= scalar
    return term_sum
end

# super slow default
function _mult!(::StorageType, term_sum::AbstractTermSum, scalar::Number)
    for (term, coeff) in zip(terms(term_sum), coefficients(term_sum))
        set!(term_sum, term, coeff * scalar)
    end
    return term_sum
end

function Base.delete!(term_sum::AbstractTermSum, term)
    _delete!(StorageType(term_sum), term_sum, term)
    return term_sum
end

function _delete!(::DictStorage, term_sum::AbstractTermSum, term)
    dict_storage = storage(term_sum)
    delete!(dict_storage, term)
    return term_sum
end

function Base_delete!(::ArrayStorage, term_sum::AbstractTermSum, term)
    terms_vec, coeffs_vec = storage(term_sum)
    ind = findfirst(t -> t == term, terms_vec)
    if !isnothing(ind)
        deleteat!(terms_vec, ind)
        deleteat!(coeffs_vec, ind)
    end
    return term_sum
end

function _delete!(::StorageType, term_sum::AbstractTermSum, term)
    # by default we set the coefficient to zero
    set!(term_sum, term, zero(coefftype(term_sum)))
    return term_sum
end

function Base.empty!(term_sum::AbstractTermSum)
    return _empty!(StorageType(term_sum), term_sum)
end

function _empty!(::DictStorage, term_sum::AbstractTermSum)
    dict_storage = storage(term_sum)
    empty!(dict_storage)
    return term_sum
end

function _empty!(::ArrayStorage, term_sum::AbstractTermSum)
    terms_vec, coeffs_vec = storage(term_sum)
    empty!(terms_vec)
    empty!(coeffs_vec)
    return term_sum
end

function _empty!(::StorageType, term_sum::AbstractTermSum)
    for term in terms(term_sum)
        _delete!(StorageType(term_sum), term_sum, term)
    end
    return term_sum
end


function Base.similar(term_sum::AbstractTermSum)
    similar_term_sum = deepcopy(term_sum)
    empty!(similar_term_sum)
    return similar_term_sum
end


### Short out-of-place algebra

function Base.:+(term_sum1::AbstractTermSum, term_sum2::AbstractTermSum)
    result = deepcopy(term_sum1)
    add!(result, term_sum2)
    return result
end


function Base.:-(term_sum1::AbstractTermSum, term_sum2::AbstractTermSum)
    result = deepcopy(term_sum1)
    subtract!(result, term_sum2)
    return result
end

function Base.:*(term_sum::AbstractTermSum, scalar)
    result = deepcopy(term_sum)
    mult!(result, scalar)
    return result
end

Base.:*(c::Number, term_sum::AbstractTermSum) = term_sum * c

Base.:/(term_sum::AbstractTermSum, scalar) = term_sum * (one(scalar) / scalar)


# check for equality by equaility on all fields

function Base.:(==)(term_sum1::AbstractTermSum, term_sum2::AbstractTermSum)
    # need to merge to efficiently check for equality
    return _comparison(==, merge(term_sum1), merge(term_sum2))
end

function Base.:(≈)(term_sum1::AbstractTermSum, term_sum2::AbstractTermSum; approx_kwargs...)
    return _comparison(≈, merge(term_sum1), merge(term_sum2))
end


function _comparison(compfunc::f, term_sum1::AbstractTermSum, term_sum2::AbstractTermSum) where {f<:Function}
    nsites(term_sum1) == nsites(term_sum2) || return false

    # we don't strictly need to check the length of the term sums
    # small values are allowed for terms that don't exist in both
    for (pstr, coeff) in term_sum1
        if !compfunc(getcoeff(term_sum2, pstr), coeff)
            return false
        end
    end
    for (pstr, coeff) in term_sum2
        if !compfunc(getcoeff(term_sum1, pstr), coeff)
            return false
        end
    end

    return true
end
