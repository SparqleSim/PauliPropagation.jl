### treetracker.jl
##
# Methods for tracking genealogy during propagation using PauliTreeTracker.
# It records parent-child relationships and edge coefficients during gate application.
##
###

using UUIDs

# Import necessary functions from parent modules
import ..PauliPropagation: PauliSum, PauliString, PauliStringType, PauliRotation, MaskedPauliRotation, CliffordGate
import ..PauliPropagation: _tomaskedpaulirotation, paulitype, paulirotationproduct, commutes, set!, add!
import ..PauliPropagation: inttostring, symboltoint, getpauli, setpauli, splitapply, applytoall!, apply
import Base: mergewith!

# Global storage for the evolution tree
EVOLUTION_TREE = Dict{String,TreeNode}()
EVOLUTION_EDGES = Vector{TreeEdge}()

"""
    reset_tree!()

Reset the global evolution tree storage.
"""
function reset_tree!()
    empty!(EVOLUTION_TREE)
    empty!(EVOLUTION_EDGES)
end

"""
    add_node!(node_id::String, pauli_string::String, gate_applied::Union{String, Nothing}=nothing)

Add a node to the evolution tree.
"""
function add_node!(node_id::String, pauli_string::String, gate_applied::Union{String,Nothing}=nothing)
    EVOLUTION_TREE[node_id] = TreeNode(node_id, pauli_string, gate_applied)
end

"""
    add_edge!(parent_id::String, child_id::String, coefficient::String, gate::String)

Add an edge to the evolution tree.
"""
function add_edge!(parent_id::String, child_id::String, coefficient::String, gate::String)
    push!(EVOLUTION_EDGES, TreeEdge(parent_id, child_id, coefficient, gate))
end

"""
    create_child_tracker(parent::PauliTreeTracker, coeff_expression::String, gate_name::String)

Create a new child tracker from a parent during gate application.
"""
function create_child_tracker(parent::PauliTreeTracker, coeff_value::Number, coeff_expression::String, gate_name::String)
    child_id = string(uuid4())[1:8]
    child_tracker = PauliTreeTracker(coeff_value, child_id, parent.node_id)

    # Add edge to the tree
    add_edge!(parent.node_id, child_id, coeff_expression, gate_name)

    return child_tracker
end

"""
    format_pauli_string(pstr)

Convert a Pauli string to a readable string format like "XXZ".
"""
function format_pauli_string(pstr::PauliStringType, nqubits::Int)
    return inttostring(pstr, nqubits)
end

function format_pauli_string(pstr::PauliString)
    return inttostring(pstr.term, pstr.nqubits)
end

"""
    mergewith!(combine, d1::PauliSum{TT,PauliTreeTracker{T}}, d2::PauliSum{TT,PauliTreeTracker{T}}) where {TT,T}

Specialized mergewith! for PauliSum with PauliTreeTracker coefficients.
This provides the context of which Pauli string is being merged and properly tracks the genealogy.
"""
function Base.mergewith!(combine, d1::PauliSum{TT,PauliTreeTracker{T}}, d2::PauliSum{TT,PauliTreeTracker{T}}) where {TT,T}
    # Ensure d1 is the larger dictionary for efficiency (similar to the base implementation)
    if length(d1) < length(d2)
        d1, d2 = d2, d1
    end

    # Iterate through all entries in d2
    for (pstr, tracker2) in d2
        if haskey(d1.terms, pstr)
            # This Pauli string exists in both sums - we need to merge the trackers
            tracker1 = d1.terms[pstr]
            merged_tracker = merge_trackers_with_context(tracker1, tracker2, pstr, d1.nqubits)
            d1.terms[pstr] = merged_tracker
        else
            # This Pauli string only exists in d2 - just add it to d1
            d1.terms[pstr] = tracker2
        end
    end

    return d1
end

"""
    merge_trackers_with_context(tracker1::PauliTreeTracker, tracker2::PauliTreeTracker, pstr, nqubits::Int)

Merge two PauliTreeTracker objects with context about which Pauli string they represent.
Creates a proper merge node in the evolution tree.
"""
function merge_trackers_with_context(tracker1::PauliTreeTracker, tracker2::PauliTreeTracker, pstr, nqubits::Int)
    # Calculate the merged coefficient
    merged_coeff = tracker1.coeff + tracker2.coeff
    merged_id = string(uuid4())[1:8]

    # Create the merged tracker
    merged_tracker = PauliTreeTracker(merged_coeff, merged_id, nothing)  # No single parent for merges

    # Format the Pauli string for display
    pstr_str = format_pauli_string(pstr, nqubits)

    # Add the merged node to the tree
    add_node!(merged_id, pstr_str, "MERGE")

    # Add edges from both parent trackers to the merged node
    add_edge!(tracker1.node_id, merged_id, "", "MERGE")
    add_edge!(tracker2.node_id, merged_id, "", "MERGE")

    return merged_tracker
end

### Specialized methods for gate applications

"""
    splitapply(gate::MaskedPauliRotation, pstr::PauliStringType, coeff::PauliTreeTracker, theta; nqubits::Int, kwargs...)

Specialized splitapply for PauliTreeTracker that tracks the tree evolution.
This creates two child nodes with cos and sin coefficients.
"""
function splitapply(gate::MaskedPauliRotation, pstr::PauliStringType, coeff::PauliTreeTracker, theta; nqubits::Int, kwargs...)
    # Get the gate name for labeling - extract first symbol from gate.symbols
    gate_symbol = isempty(gate.symbols) ? "?" : string(gate.symbols[1])
    gate_name = "R$(gate_symbol)"

    # Add the current node to the tree if not already there
    pstr_str = format_pauli_string(pstr, nqubits)


    # Create cos coefficient child
    cos_coeff_value = coeff.coeff * cos(theta)
    cos_multiplier = cos(theta)
    cos_child = create_child_tracker(coeff, cos_coeff_value, string(round(cos_multiplier, digits=3)), gate_name)
    add_node!(cos_child.node_id, pstr_str, gate_name)

    # Get new Pauli string and sign for sin coefficient
    new_pstr, sign = paulirotationproduct(gate, pstr)
    sin_coeff_value = coeff.coeff * sin(theta) * sign
    sin_multiplier = sin(theta) * sign
    sin_child = create_child_tracker(coeff, sin_coeff_value, string(round(sin_multiplier, digits=3)), gate_name)

    # Add the new Pauli string node
    new_pstr_str = format_pauli_string(new_pstr, nqubits)
    add_node!(sin_child.node_id, new_pstr_str, gate_name)

    return pstr, cos_child, new_pstr, sin_child
end

"""
    applytoall!(gate::PauliRotation, theta, psum::PauliSum{TT,PauliTreeTracker{T}}, aux_psum; kwargs...)

Specialized applytoall! for PauliRotation gates with PauliSum containing PauliTreeTracker coefficients.
This tracks every gate application in the evolution tree.
"""
function PropagationBase.applytoall!(gate::PauliRotation, prop_cache::PauliPropagationCache{PauliSum{TT,PauliTreeTracker{T}}}, theta; kwargs...) where {TT<:PauliStringType,T<:Number}
    psum = mainsum(prop_cache)
    aux_psum = auxsum(prop_cache)

    # Convert PauliRotation to MaskedPauliRotation for efficiency
    gate_mask = symboltoint(paulitype(psum), gate.symbols, gate.qinds)

    # Loop over all Pauli strings and their coefficients in the Pauli sum
    for (pstr, coeff) in psum
        pstr_str = format_pauli_string(pstr, psum.nqubits)
        if commutes(gate_mask, pstr)
            # If the gate commutes, create a new child node and edge
            pstr_str = format_pauli_string(pstr, psum.nqubits)
            gate_symbol = isempty(gate.symbols) ? "?" : string(gate.symbols[1])
            gate_name = "R$(gate_symbol)"

            # Create new child tracker (coefficient stays the same for commuting gates)
            new_child = create_child_tracker(coeff, coeff.coeff, "1", gate_name)
            add_node!(new_child.node_id, pstr_str, gate_name)

            # Update the coefficient in the sum with the new child tracker
            set!(psum, pstr, new_child)
            continue
        end

        # Apply the gate and track the split
        pstr, coeff1, new_pstr, coeff2 = splitapply(gate_mask, pstr, coeff, theta; nqubits=psum.nqubits, kwargs...)

        # Set the coefficient of the original Pauli string
        set!(psum, pstr, coeff1)

        # Set the coefficient of the new Pauli string in the aux_psum
        set!(aux_psum, new_pstr, coeff2)
    end

    return
end

"""
    applytoall!(gate::CliffordGate, prop_cache::PauliPropagationCache{PauliSum{TT,PauliTreeTracker{T}}}; kwargs...)

Specialized applytoall! for CliffordGate with PauliSum containing PauliTreeTracker coefficients.
Clifford gates deterministically transform Pauli strings without branching, so we create a single child node for each transformation.
"""
function PropagationBase.applytoall!(gate::CliffordGate, prop_cache::PauliPropagationCache{PauliSum{TT,PauliTreeTracker{T}}}; kwargs...) where {TT<:PauliStringType,T<:Number}
    psum = mainsum(prop_cache)
    aux_psum = auxsum(prop_cache)

    # Loop over all Pauli strings and their coefficients in the Pauli sum
    for (pstr, coeff) in psum
        # Apply the Clifford gate to get the new Pauli string and coefficient
        new_pstr, new_coeff_value = apply(gate, pstr, coeff.coeff; kwargs...)

        # Format the gate name for display
        gate_name = string(gate.symbol)

        # Create a new child tracker for the transformed Pauli string
        edge_num = new_coeff_value / coeff.coeff
        new_child = create_child_tracker(coeff, new_coeff_value, string(round(edge_num, digits=3)), gate_name)

        # Add the new node to the tree
        new_pstr_str = format_pauli_string(new_pstr, psum.nqubits)
        add_node!(new_child.node_id, new_pstr_str, gate_name)

        # Set the new Pauli string and its tracker in aux_psum
        # (Note: Clifford gates create non-overlapping Pauli strings so we can use set!)
        set!(aux_psum, new_pstr, new_child)
    end

    # Empty the original psum since everything was moved to aux_psum
    empty!(psum)

    return
end
"""
    applytoall!(gate::PauliNoise, prop_cache::PauliPropagationCache{PauliSum{TT,PauliTreeTracker{T}}}, p; kwargs...) where {TT<:PauliStringType,T<:Number}

Specialized applytoall! for PauliNoise with PauliSum containing PauliTreeTracker coefficients.
PauliNoise gates deterministically transform Pauli strings without branching, so we create a single child node for each transformation.
"""
function PropagationBase.applytoall!(gate::PauliNoise, prop_cache::PauliPropagationCache{PauliSum{TT,PauliTreeTracker{T}}}, p; kwargs...) where {TT<:PauliStringType,T<:Number}
    # check that the noise strength is in the correct range
    _check_noise_strength(PauliNoise, p)

    psum = mainsum(prop_cache)

    # loop over all Pauli strings and their coefficients in the Pauli sum
    for (pstr, coeff) in psum
        # the Pauli on the site that the noise acts on
        pauli = getpauli(pstr, gate.qind)

        # `isdamped` is defined in noisechannels.jl for each Pauli noise channel
        # I Paulis are never damped, but the others vary
        if !isdamped(gate, pauli)
            continue
        end

        new_coeff_value = coeff.coeff * (1 - p)

        # Format the gate name for display
        gate_name = string(typeof(gate).name.name)

        # Create a new child tracker for the transformed Pauli string
        edge_num = new_coeff_value / coeff.coeff
        new_child = create_child_tracker(coeff, new_coeff_value, string(round(edge_num, digits=3)), gate_name)

        # Add the new node to the tree
        pstr_str = format_pauli_string(pstr, psum.nqubits)
        add_node!(new_child.node_id, pstr_str, gate_name)

        # Set the new coefficient in psum
        set!(psum, pstr, new_child)
    end

    return
end

"""
    applytoall!(gate::AmplitudeDampingNoise, gamma, psum::PauliSum{TT,PauliTreeTracker{T}}, aux_psum; kwargs...)

Specialized applytoall! for AmplitudeDampingNoise with PauliSum containing PauliTreeTracker coefficients.
AmplitudeDampingNoise gates can transform Pauli strings in two ways:
1. X/Y Paulis get scaled by sqrt(1-gamma)
2. Z Paulis split into I with coefficient gamma and Z with coefficient (1-gamma)
"""
function PropagationBase.applytoall!(gate::AmplitudeDampingNoise, prop_cache::PauliPropagationCache{PauliSum{TT,PauliTreeTracker{T}}}, gamma; kwargs...) where {TT<:PauliStringType,T<:Number}
    psum = mainsum(prop_cache)
    aux_psum = auxsum(prop_cache)

    # check that the noise strength is in the correct range
    _check_noise_strength(AmplitudeDampingNoise, gamma)

    # loop over all Pauli strings and their coefficients in the Pauli sum
    for (pstr, coeff) in psum
        pauli = getpauli(pstr, gate.qind)
        if pauli == 0
            # Pauli is I, so the gate does not do anything
            continue

        elseif pauli == 1 || pauli == 2
            # Pauli is X or Y, so the gate will give a sqrt(1-gamma) prefactor
            new_coeff_value = coeff.coeff * sqrt(1 - gamma)

            # Format the gate name for display
            gate_name = string(typeof(gate).name.name)

            # Create a new child tracker for the transformed Pauli string
            edge_num = sqrt(1 - gamma)
            new_child = create_child_tracker(coeff, new_coeff_value, string(round(edge_num, digits=3)), gate_name)

            # Add the new node to the tree
            pstr_str = format_pauli_string(pstr, psum.nqubits)
            add_node!(new_child.node_id, pstr_str, gate_name)

            # Set the new coefficient in psum
            set!(psum, pstr, new_child)

        else
            # Pauli is Z, so the gate will split the Pauli string 
            new_pstr = setpauli(pstr, 0, gate.qind)
            coeff1_value = (1 - gamma) * coeff.coeff
            coeff2_value = gamma * coeff.coeff

            # Format the gate name for display
            gate_name = string(typeof(gate).name.name)

            # Create child trackers for both branches
            edge_num1 = 1 - gamma
            edge_num2 = gamma
            new_child1 = create_child_tracker(coeff, coeff1_value, string(round(edge_num1, digits=3)), gate_name)
            new_child2 = create_child_tracker(coeff, coeff2_value, string(round(edge_num2, digits=3)), gate_name)

            # Add the new nodes to the tree
            pstr_str1 = format_pauli_string(pstr, psum.nqubits)
            pstr_str2 = format_pauli_string(new_pstr, psum.nqubits)
            add_node!(new_child1.node_id, pstr_str1, gate_name)
            add_node!(new_child2.node_id, pstr_str2, gate_name)

            # Set the coefficients in psum and aux_psum
            set!(psum, pstr, new_child1)
            set!(aux_psum, new_pstr, new_child2)
        end
    end

    return
end
