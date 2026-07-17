###
##
# Variants of `applymergetruncate!` for PauliSum that truncate during gate application.
# This may yield slightly different results compared to normal functionality.
# We also use Base.Dict internals for speedups
##
###

const PB = PauliPropagation.PropagationBase

# Runs once at load time: the Dict internals used below are undocumented and could change between
# Julia versions, silently corrupting data instead of erroring. Fail loudly if that ever happens.
function _check_dict_internals()
    required_fields = (:slots, :keys, :vals)
    dict_fields = fieldnames(Dict)
    missing_fields = setdiff(required_fields, dict_fields)
    if !isempty(missing_fields)
        error("Performance.fused_dict requires Base.Dict to have fields $required_fields, " *
              "but Dict is missing $missing_fields (has fields $dict_fields). This file manipulates " *
              "Dict's internal storage directly and is unsafe to use on this Julia version.")
    end

    for fname in (:isslotfilled, :ht_keyindex2_shorthash!, :_setindex!, :_delete!)
        if !isdefined(Base, fname)
            error("Performance.fused_dict requires Base.$fname to exist, but it was not found. " *
                  "This file manipulates Dict's internal storage directly and is unsafe to use " *
                  "on this Julia version.")
        end
    end

    # replicate the update/delete-by-index and find-or-insert sequence used below, then compare
    # against the public Dict API doing the same logical operations
    d = Dict{Int,Float64}(1 => 1.0, 2 => 2.0, 3 => 3.0)
    reference = copy(d)

    @inbounds for i in 1:length(d.slots)
        Base.isslotfilled(d, i) || continue
        if d.keys[i] == 1
            d.vals[i] = 10.0
        elseif d.keys[i] == 2
            Base._delete!(d, i)
        end
    end
    reference[1] = 10.0
    delete!(reference, 2)

    for (key, delta) in ((4, 4.0), (3, 30.0))
        index, sh = Base.ht_keyindex2_shorthash!(d, key)
        if index > 0
            d.vals[index] += delta
        else
            Base._setindex!(d, delta, key, -index, sh)
        end
    end
    reference[4] = get(reference, 4, 0.0) + 4.0
    reference[3] = get(reference, 3, 0.0) + 30.0

    if d != reference
        error("Performance.fused_dict internal Dict manipulation produced $d, expected $reference. " *
              "Base.Dict's internal layout or helper semantics have changed on this Julia version " *
              "and this file's internals-based fast paths are no longer safe to use.")
    end

    return nothing
end

_check_dict_internals()

"""
    applymergetruncate!(gate::PauliRotation, prop_cache::PauliPropagationCache, theta; fused::Bool=false, min_abs_coeff=1e-10, max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, kwargs...)

Fused overload that truncates during gate application by walking and mutating the backing `Dict`
through its internal slot array. Only used when `fused=true`; otherwise falls through unchanged to
stock `applymergetruncate!`.
"""
function PauliPropagation.applymergetruncate!(gate::PauliRotation, prop_cache::PauliPropagationCache, theta;
    fused::Bool=false,
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing, kwargs...)

    # invoke function from library
    if !fused
        return invoke(PauliPropagation.applymergetruncate!,
            Tuple{PauliRotation,PauliPropagation.AbstractPauliPropagationCache,typeof(theta)},
            gate, prop_cache, theta;
            min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, kwargs...)
    end

    psum = PB.storage(mainsum(prop_cache))
    gate_mask = symboltoint(paulitype(prop_cache), gate.symbols, gate.qinds)
    cos_val, sin_val = cos(theta), sin(theta)

    truncfunc(pstr, coeff) = _fusedtruncfunc(pstr, coeff; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc)

    touched = Tuple{keytype(psum),valtype(psum)}[]
    # bounded above by one push per filled, non-commuting slot -- a tighter bound than the raw
    # table size (nslots), which includes empty/tombstone slots
    sizehint!(touched, length(psum))

    # pass 1: walk existing slots directly, update/delete in place by known slot index
    nslots = length(psum.slots)
    @inbounds for i in 1:nslots
        Base.isslotfilled(psum, i) || continue

        pstr = psum.keys[i]
        commutes(gate_mask, pstr) && continue
        coeff = psum.vals[i]

        coeff1 = coeff * cos_val
        if truncfunc(pstr, coeff1)
            Base._delete!(psum, i)
        else
            psum.vals[i] = coeff1
        end

        new_pstr, sign = PauliPropagation.paulirotationproduct(gate_mask, pstr)
        push!(touched, (new_pstr, coeff * sin_val * sign))
    end

    # pass 2: accumulate each touched delta into its final value and truncate immediately.
    # `touched` never contains duplicate keys: for a fixed gate_mask, paulirotationproduct is a
    # bijection on Pauli strings (multiplication by a fixed Pauli group element), and pass 1 only
    # ever touches each dict key once -- so distinct source pstrs always produce distinct
    # new_pstrs. That means each touched key gets exactly one, complete update below (either
    # accumulated onto whatever value it already had before this gate ran, or freshly inserted) --
    # never a partial sum -- so it's safe to truncate right after that single update instead of
    # deferring to a second pass with a second hash lookup per entry.
    for (new_pstr, delta) in touched
        index, sh = Base.ht_keyindex2_shorthash!(psum, new_pstr)
        if index > 0
            new_val = psum.vals[index] + delta
            if truncfunc(new_pstr, new_val)
                Base._delete!(psum, index)
            else
                psum.vals[index] = new_val
            end
        elseif !truncfunc(new_pstr, delta)
            Base._setindex!(psum, delta, new_pstr, -index, sh)
        end
    end

    return prop_cache
end
