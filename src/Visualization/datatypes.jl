### datatypes.jl
##
# Data types for tracking genealogy during propagation in the Visualization module.
# Contains TreeNode, TreeEdge, and PauliTreeTracker types.
##
###

using UUIDs

# Import necessary functions from parent modules
import ..PauliPropagation: PathProperties

"""
    TreeNode

Represents a node in the Pauli evolution tree.
Contains the Pauli string representation and a unique identifier.
"""
struct TreeNode
    id::String
    pauli_string::String
    gate_applied::Union{String,Nothing}  # The gate that created this node (if any)
end

"""
    TreeEdge

Represents an edge in the Pauli evolution tree.
Contains the coefficient that was applied during the transformation.
"""
struct TreeEdge
    parent_id::String
    child_id::String
    coefficient::String  # String representation of the coefficient (e.g., "cos(θ₁)", "-sin(θ₁)")
    gate::String
end

"""
    PauliTreeTracker(coeff::Number, node_id::String, parent_id::Union{String, Nothing})

Wrapper type for numerical coefficients in Pauli propagation that tracks the genealogy tree.
Each PauliTreeTracker corresponds to a node in the evolution tree.
"""
mutable struct PauliTreeTracker{T<:Number} <: PathProperties
    coeff::T
    node_id::String
    parent_id::Union{String,Nothing}

    # Constructor
    function PauliTreeTracker(coeff::T, node_id::String, parent_id::Union{String,Nothing}=nothing) where {T<:Number}
        new{T}(coeff, node_id, parent_id)
    end
end

"""
    PauliTreeTracker(coeff::Number)

Constructor for `PauliTreeTracker` from only a coefficient.
Creates a new unique node ID and no parent.
"""
function PauliTreeTracker(coeff::Number)
    node_id = string(uuid4())[1:8]  # Use first 8 characters of UUID for readability
    return PauliTreeTracker(float(coeff), node_id, nothing)
end