### Propagation necessities

"""
    propagate(circ, pstr::PauliString{<:Integer,NodePathProperties}; max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, kwargs...)

Construct a Pauli propagation surrogate of the propagated `PauliString` through the circuit `circ` in the Heisenberg picture. 
The circuit must only contain `CliffordGate`s and `PauliRotation`s.
Truncations based on any numerical coefficient value cannot be used.
Everything else is the same as in `propagate!()` for the non-Surrogate code.
"""
function PropagationBase.propagate(circ, pstr::PauliString{TT,NodePathProperties}; max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, kwargs...) where {TT<:PauliStringType}
    return propagate(circ, PauliSum(pstr); max_weight, max_freq, max_sins, customtruncfunc, kwargs...)
end

"""
    propagate(circ, psum::PauliSum{<:Integer,NodePathProperties}; max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, kwargs...)

Construct a Pauli propagation surrogate of the propagated `PauliSum` through the circuit `circ` in the Heisenberg picture.
The circuit must only contain `CliffordGate`s and `PauliRotation`s. 
Truncations based on any numerical coefficient value cannot be used.
Everything else is the same as in `propagate!()` for the non-Surrogate code.
"""
function PropagationBase.propagate(circ, psum::PauliSum{TT,NodePathProperties}; max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, kwargs...) where {TT<:PauliStringType}
    _checksurrogationconditions(circ)
    return propagate!(circ, deepcopy(psum); max_weight, max_freq, max_sins, customtruncfunc, kwargs...)
end

"""
    propagate!(circ, psum::PauliSum{<:Integer,NodePathProperties}; max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, kwargs...)

Construct a Pauli propagation surrogate of the propagated `PauliSum` through the circuit `circ` in the Heisenberg picture. 
The `PauliSum` `psum` is modified in place.
The circuit must only contain `CliffordGate`s and `PauliRotation`s.
Truncations based on any numerical coefficient value cannot be used.
Everything else is the same as in `propagate!()` for the non-Surrogate code.
"""
function PropagationBase.propagate!(circ, psum::PauliSum{TT,NodePathProperties}; max_weight=Inf, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, kwargs...) where {TT<:PauliStringType}
    _checksurrogationconditions(circ)

    # dummy parameters transport the parameter indices 
    dummy_thetas = collect(Int, 1:countparameters(circ))

    propagate!(circ, psum, dummy_thetas; max_weight=max_weight, max_freq=max_freq, max_sins=max_sins, customtruncfunc=customtruncfunc, kwargs...)
end

function _checksurrogationconditions(circ)
    if !all(isa(gate, CliffordGate) || isa(gate, PauliRotation) for gate in circ)
        throw(ArgumentError("The surrogate currently only accepts `CliffordGate`s and `PauliRotation`s."))
    end
    return
end


## For Pauli Rotations
# overloads for _applycos and _applysins defined in PathProperties/paulifreqtracker.jl
function _applycos(path::NodePathProperties, theta, sign=1; kwargs...)
    # the parameter encodes the parameter index 
    param_idx = theta
    return NodePathProperties(_buildcosnode(path.node, param_idx, sign), path.nsins, path.ncos + 1, path.freq + 1)
end

function _buildcosnode(node::CircuitNode, param_idx, sign=1; kwargs...)
    return PauliRotationNode(parents=[node], trig_inds=[1], signs=[sign], param_idx=param_idx)
end

function _applysin(path::NodePathProperties, theta, sign=1; kwargs...)
    # the parameter encodes the parameter index 
    param_idx = theta
    return NodePathProperties(_buildsinnode(path.node, param_idx, sign), path.nsins + 1, path.ncos, path.freq + 1)
end

function _buildsinnode(node::CircuitNode, param_idx, sign=1; kwargs...)
    return PauliRotationNode(parents=[node], trig_inds=[-1], signs=[sign], param_idx=param_idx)
end

function PropagationBase.mergefunc(pth1::NodePathProperties, pth2::NodePathProperties)
    return NodePathProperties(
        _mergenodes!(pth1.node, pth2.node),
        min(pth1.nsins, pth2.nsins),
        min(pth1.ncos, pth2.ncos),
        min(pth1.freq, pth2.freq)
    )
end

function _mergenodes!(node1::CircuitNode, node2::CircuitNode)
    append!(node1.parents, node2.parents)
    append!(node1.trig_inds, node2.trig_inds)
    append!(node1.signs, node2.signs)
    return node1
end

## For Clifford Gates


# Apply a `CliffordGate` to an integer Pauli string and `NodePathProperties` coefficient. 
# the only necessary overload is what happens when we multiply node with a sign

Base.:*(pth::NodePathProperties, sign::Number) = NodePathProperties(_multiplysign(pth.node, sign), pth.nsins, pth.ncos, pth.freq)

function _multiplysign(pauli_node::PauliRotationNode, sign; kwargs...)
    for ii in eachindex(pauli_node.signs)
        pauli_node.signs[ii] *= sign
    end
    return pauli_node
end

function _multiplysign(eval_endnode::EvalEndNode, sign; kwargs...)
    eval_endnode.coefficient *= sign
    return eval_endnode
end

## Truncation functions

# don't truncate on coefficients
function truncatemincoeff(path::NodePathProperties, min_abs_coeff::Real)
    return false
end