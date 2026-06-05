###
##
# This file contains the types with which we represent the surrogate graph.
##
###

# Abstract node type for the Pauli propagation Surrogate #
abstract type CircuitNode end


# Node type for the Pauli strings in the observable to be backpropagated.
Base.@kwdef mutable struct EvalEndNode <: CircuitNode
    pstr::Int
    coefficient::Float64
    cummulative_value::Float64 = 0.0
    is_evaluated::Bool = false
end


# Constructor for `EvalEndNode` with a default coefficient of 1.0.
EvalEndNode(pstr::Integer) = EvalEndNode(pstr, 1.0)


# Surrogate graph node for a Pauli rotation gate.
Base.@kwdef mutable struct PauliRotationNode <: CircuitNode
    parents::Vector{Union{EvalEndNode,PauliRotationNode}}
    trig_inds::Vector{Int}
    signs::Vector{Int}
    param_idx::Int
    cummulative_value::Float64 = 0.0  # This must be changed to Real for automatic differentiation libraries.
    is_evaluated::Bool = false
end


# Pretty print for `CircuitNode`
Base.show(io::IO, node::CircuitNode) = print(io, "$(typeof(node))($(length(node.parents)) parent(s), param_idx=$(node.param_idx))")

# Pretty print for `EvalEndNode`
Base.show(io::IO, node::EvalEndNode) = print(io, "$(typeof(node))(Pauli string=$(node.pstr), coefficient=$(node.coefficient))")

## PathProperties Type
"""
    NodePathProperties(node::CircuitNode)
    NodePathProperties(node::CircuitNode, nsins::Int, ncos::Int, freq::Int)

Surrogate `PathProperties` type. Carries `CircuitNode`s instead of numerical coefficients.
If `nsins`, `ncos`, and `freq` are not provided, they are initialized to 0.
"""
struct NodePathProperties <: PathProperties
    node::Union{EvalEndNode,PauliRotationNode}
    nsins::Int
    ncos::Int
    freq::Int
end

PropagationBase.numcoefftype(::Type{NodePathProperties}) = Float64


# Pretty print for PauliFreqTracker
Base.show(io::IO, pth::NodePathProperties) = print(io, "NodePathProperties($(typeof(pth.node)), nsins=$(pth.nsins), ncos=$(pth.ncos), freq=$(pth.freq))")



# One-argument constructor for `NodePathProperties`. Initializes `nsins`, `ncos`, and `freq` to 0.
NodePathProperties(node::CircuitNode) = NodePathProperties(node, 0, 0, 0)


"""
    tonumber(path::NodePathProperties)

Get the cummulative coefficient of a `NodePathProperties` node.
This assumes that the surrogate has already been evaluated.
"""
PropagationBase.tonumber(path::NodePathProperties) = path.node.cummulative_value


"""
    wrapcoefficients(pstr::PauliString, ::Type{NodePathProperties})

Wrap the coefficient of a `PauliString` into `NodePathProperties`.
"""
function wrapcoefficients(pstr::PauliString, ::Type{NodePathProperties})
    node = NodePathProperties(EvalEndNode(pstr.term, pstr.coeff, 0.0, false))
    return PauliString(pstr.nqubits, pstr.term, node)
end

"""
    wrapcoefficients(psum::PauliSum, ::Type{NodePathProperties})

Wrap the coefficients of a `PauliSum` into `NodePathProperties`.
"""
function wrapcoefficients(psum::PauliSum, ::Type{NodePathProperties})
    return PauliSum(psum.nqubits, Dict(pstr => NodePathProperties(EvalEndNode(pstr, coeff, 0.0, false)) for (pstr, coeff) in psum.terms))
end


parents(node::T) where {T<:CircuitNode} = node.parents

function getnodeval(node::T) where {T<:CircuitNode}
    return node.cummulative_value
end

function isevaluated(node::T)::Bool where {T<:CircuitNode}
    return node.is_evaluated
end

function setcummulativevalue(node::CircuitNode, val)
    node.cummulative_value = val
    return
end

function PropagationBase.set!(psum::Dict{TT,NodePathProperties}, pstr::TT, path::NodePathProperties) where {TT}
    psum[pstr] = path
    return psum
end
