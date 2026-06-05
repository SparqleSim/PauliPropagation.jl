### generics.jl
##
# This file contains the foundational functions for the `propagation` function. 
# They can be overloaded to custom gate types or custom behaviour in `specializations.jl`.
##
###
"""
    propagate(circ, pstr::PauliString, thetas=nothing; min_abs_coeff=1e-10, max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, heisenberg=true, kwargs...)

Propagate a `PauliString` through the circuit `circ`.
By default this is done in the Heisenberg picture, as indicated by `heisenberg=true`. 
This means that the circuit is applied to the Pauli string in reverse order, and the action of each gate is its conjugate action.
Parameters for the parametrized gates in `circ` are given by `thetas`, and need to be passed as if the circuit was applied as written in the Schrödinger picture.
If thetas are not passed, the circuit must contain only non-parametrized `StaticGates`.
Default truncations are `min_abs_coeff`, `max_weight`, `max_freq`, and `max_sins`.
`max_freq`, and `max_sins` will lead to automatic conversion if the coefficients are not already wrapped in suitable `PathProperties` objects.
A custom truncation function can be passed as `customtruncfunc` with the signature customtruncfunc(pstr::PauliStringType, coefficient)::Bool.
Further `kwargs` are passed to the lower-level functions `applymergetruncate!`, `applytoall!`, and `apply`.
"""
function PropagationBase.propagate(circuit, pstr::PauliString, thetas=nothing; min_abs_coeff=1e-10, max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, heisenberg=true, kwargs...)
    psum = PauliSum(pstr)
    return propagate(circuit, psum, thetas; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, heisenberg, kwargs...)
end


# In-place version of `propagate()` for a `PauliString`.
# This is only a convenience function, because the `PauliString` is converted into a `PauliSum` internally.
# If `max_freq`, and `max_sins` are used without the coefficients already being wrapped in suitable `PathProperties` objects, an error is thrown.
function PropagationBase.propagate!(circuit, pstr::PauliString, thetas=nothing; max_weight=Inf, min_abs_coeff=1e-10, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, heisenberg=true, kwargs...)
    psum = PauliSum(pstr)
    # check that max_freq and max_sins are only used a PathProperties type tracking them
    _checkfreqandsinfields(psum, max_freq, max_sins)
    return propagate(circuit, psum, thetas; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, heisenberg, kwargs...)
end

"""
    propagate(circuit, psum::AbstractPauliSum, thetas=nothing; min_abs_coeff=1e-10, max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, heisenberg=true, kwargs...)
    propagate!(circuit, psum::AbstractPauliSum, thetas=nothing; min_abs_coeff=1e-10, max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, heisenberg=true, kwargs...)

Propagate a Pauli sum `psum` through the circuit `circ`. 
By default this is done in the Heisenberg picture, as indicated by `heisenberg=true`. 
This means that the circuit is applied to the Pauli sum in reverse order, and the action of each gate is its conjugate action.
In `propagate()` the Pauli sum `psum` is deepcopied and passed into the in-place propagation function `propagate!()`.
Parameters for the parametrized gates in `circ` are given by `thetas`, and need to be passed as if the circuit was applied as written in the Schrödinger picture.
If thetas are not passed, the circuit must contain only non-parametrized `StaticGates`.
Default truncations are `min_abs_coeff`, `max_weight`, `max_freq`, and `max_sins`.
`max_freq`, and `max_sins` will lead to automatic conversion if the coefficients are not already wrapped in suitable `PathProperties` objects.
A custom truncation function can be passed as `customtruncfunc` with the signature customtruncfunc(pstr::PauliStringType, coefficient)::Bool.
Further `kwargs` are passed to the lower-level functions `applymergetruncate!`, `applytoall!`, and `apply`.
"""
function PropagationBase.propagate(circuit, psum::AbstractPauliSum, thetas=nothing; max_weight=Inf, min_abs_coeff=1e-10, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, heisenberg=true, kwargs...)
    CT = coefftype(psum)

    # if max_freq and max_sins are used, and no PathProperties used, automatically wrap the coefficients in `PauliFreqTracker` 
    psum = _check_wrapping_into_paulifreqtracker(psum, max_freq, max_sins)

    # check that max_freq and max_sins are only used a PathProperties type tracking them
    _checkfreqandsinfields(psum, max_freq, max_sins)

    # run the in-place propagation function on a deepcopy of the input psum
    psum = propagate!(circuit, deepcopy(psum), thetas; max_weight, min_abs_coeff, max_freq, max_sins, customtruncfunc, heisenberg, kwargs...)

    # if the input psum was not a `PauliFreqTracker`, and the corresponding truncations were set,we need to unwrap the coefficients
    psum = _check_unwrap_from_paulifreqtracker(CT, psum)

    return psum
end


"""
    propagate!(circuit, prop_cache::AbstractPauliPropagationCache, thetas=nothing; min_abs_coeff=1e-10, max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, heisenberg=true, kwargs...)

In-place propagation of an `AbstractPauliPropagationCache` through the circuit `circ` in the Heisenberg picture.
"""
function PropagationBase.propagate!(circuit, prop_cache::AbstractPauliPropagationCache, thetas=nothing; max_weight=Inf, min_abs_coeff=1e-10, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, heisenberg=true, kwargs...)

    # if circuit is actually a single gate, promote it to a list [gate]
    # similarly the thetas if it is a single number
    circuit, thetas = PropagationBase._promotecircandparams(circuit, thetas)

    # if thetas is nothing, the circuit must contain only StaticGates
    # also check if the length of thetas equals the number of parametrized gates
    PropagationBase._checknumberofparams(circuit, thetas)

    if heisenberg
        # this usually just reverses circuit and parameter order
        circuit, thetas = toheisenberg(circuit, thetas)
    else
        # this usually entails a conversion of how gates act
        circuit, thetas = toschrodinger(circuit, thetas)
    end

    return PropagationBase._propagate!(circuit, prop_cache, thetas; max_weight, min_abs_coeff, max_freq, max_sins, customtruncfunc, kwargs...)
end


### TRUNCATE

"""
truncate!(psum::AbstractPauliSum; min_abs_coeff=1e-10, max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, kwargs...)    
truncate!(prop_cache::AbstractPauliPropagationCache; min_abs_coeff=1e-10, max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, kwargs...)

Truncation function for `AbstractPauliPropagationCache`s that combines multiple truncation criteria.
The default truncation criteria are:
- `min_abs_coeff`: Truncates Pauli strings with absolute coefficient below this value.
- `max_weight`: Truncates Pauli strings with weight (number of non-identity Paulis) above this value.
- `max_freq`: Truncates Pauli strings with frequency (number of cosine factors in coefficient) above this value.
- `max_sins`: Truncates Pauli strings with number of sine factors in coefficient above this value.
A custom truncation function can be passed as `customtruncfunc` with the signature customtruncfunc(pstr, coeff)::Bool.

This function combines all truncation criteria into a single truncation function `truncfunc()` calls PropagationBase.truncate!(truncfunc, prop_cache).
"""
function PropagationBase.truncate!(prop_cache::AbstractPauliPropagationCache; min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing, kwargs...)

    function truncfunc(pstr, coeff)
        is_truncated = false
        if truncateweight(pstr, max_weight)
            is_truncated = true
        elseif truncatemincoeff(coeff, min_abs_coeff)
            is_truncated = true
        elseif truncatefrequency(coeff, max_freq)
            is_truncated = true
        elseif truncatesins(coeff, max_sins)
            is_truncated = true
        elseif !isnothing(customtruncfunc) && customtruncfunc(pstr, coeff)
            is_truncated = true
        end

        return is_truncated
    end

    prop_cache = truncate!(truncfunc, prop_cache)

    return prop_cache
end

function PropagationBase.truncate!(psum::AbstractPauliSum; min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing, kwargs...)

    function truncfunc(pstr, coeff)
        is_truncated = false
        if truncateweight(pstr, max_weight)
            is_truncated = true
        elseif truncatemincoeff(coeff, min_abs_coeff)
            is_truncated = true
        elseif truncatefrequency(coeff, max_freq)
            is_truncated = true
        elseif truncatesins(coeff, max_sins)
            is_truncated = true
        elseif !isnothing(customtruncfunc) && customtruncfunc(pstr, coeff)
            is_truncated = true
        end

        return is_truncated
    end

    psum = truncate!(truncfunc, psum; kwargs...)

    return psum
end
