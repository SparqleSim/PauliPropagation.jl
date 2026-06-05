### plotting.jl
##
# Plotting functionality for visualizing Pauli evolution trees.
# Supports GraphViz output and potentially other visualization backends.
##
###

# Import necessary functions
import ..PauliPropagation: propagate, PauliString, PauliSum, wrapcoefficients, unwrapcoefficients
using JSON
using PlotGraphviz: plot_graphviz, read_dot_file

"""
    export_to_graphviz(filename::String="pauli_evolution_tree.dot"; 
                      show_coefficients::Bool=true,
                      node_shape::String="ellipse",
                      edge_style::String="solid",
                      graph_title::String="Pauli Propagation Graph")

Export the current evolution tree to a GraphViz DOT file.
"""
function export_to_graphviz(filename::String="pauli_evolution_tree.dot";
    show_coefficients::Bool=true,
    node_shape::String="ellipse",
    edge_style::String="solid",
    graph_title::String="Pauli Propagation Graph")

    dot_string = format_tree("graphviz", node_shape, edge_style, graph_title, show_coefficients)
    open(filename, "w") do io
        println(io, dot_string)
    end
    println("Evolution tree exported to $filename")
    println("To visualize, run: dot -Tpng $filename -o pauli_tree.png")
    return filename
end

"""
    export_to_json(filename::String="pauli_evolution_tree.json")

Export the current evolution tree to JSON format for other visualization tools.
"""
function export_to_json(filename::String="pauli_evolution_tree.json")
    try
        # Try to load JSON package
        JSON = Base.require(@__MODULE__, :JSON)
    catch
        # Fallback if JSON package is not available
        println("Warning: JSON package not available. Install it using: Pkg.add(\"JSON\")")
        println("Exporting as basic text format instead...")

        open(filename, "w") do io
            println(io, "# Pauli Evolution Tree Data")
            println(io, "# Nodes:")
            for (node_id, node) in EVOLUTION_TREE
                println(io, "Node: $node_id, Pauli: $(node.pauli_string), Gate: $(node.gate_applied)")
            end
            println(io, "\n# Edges:")
            for edge in EVOLUTION_EDGES
                println(io, "Edge: $(edge.parent_id) -> $(edge.child_id), Coeff: $(edge.coefficient), Gate: $(edge.gate)")
            end
        end
        println("Evolution tree exported as text to $filename")
        return
    end

    # Convert tree data to JSON-friendly format
    tree_data = Dict(
        "nodes" => [
            Dict(
                "id" => node.id,
                "pauli_string" => node.pauli_string,
                "gate_applied" => node.gate_applied
            ) for node in values(EVOLUTION_TREE)
        ],
        "edges" => [
            Dict(
                "parent_id" => edge.parent_id,
                "child_id" => edge.child_id,
                "coefficient" => edge.coefficient,
                "gate" => edge.gate
            ) for edge in EVOLUTION_EDGES
        ]
    )

    open(filename, "w") do io
        JSON.print(io, tree_data, 4)  # Use indent=4 as per user rules
    end
    println("Evolution tree exported to $filename")
end

"""
    print_tree_summary()

Print a summary of the current evolution tree.
"""
function print_tree_summary()
    println("=== Pauli Evolution Tree Summary ===")
    println("Number of nodes: $(length(EVOLUTION_TREE))")
    println("Number of edges: $(length(EVOLUTION_EDGES))")

    # Count gates
    gate_counts = Dict{String,Int}()
    for edge in EVOLUTION_EDGES
        gate_counts[edge.gate] = get(gate_counts, edge.gate, 0) + 1
    end

    println("\nGates applied:")
    for (gate, count) in gate_counts
        println("  $gate: $count times")
    end

    # Show root nodes (nodes with no parents)
    root_nodes = [node_id for node_id in keys(EVOLUTION_TREE)
                  if !any(edge.child_id == node_id for edge in EVOLUTION_EDGES)]

    println("\nRoot nodes: $(length(root_nodes))")
    for root_id in root_nodes
        root_node = EVOLUTION_TREE[root_id]
        println("  $root_id: $(root_node.pauli_string)")
    end

    # Show leaf nodes (nodes with no children)
    leaf_nodes = [node_id for node_id in keys(EVOLUTION_TREE)
                  if !any(edge.parent_id == node_id for edge in EVOLUTION_EDGES)]

    println("\nLeaf nodes: $(length(leaf_nodes))")
    for leaf_id in leaf_nodes
        leaf_node = EVOLUTION_TREE[leaf_id]
        println("  $leaf_id: $(leaf_node.pauli_string)")
    end
end

"""
    remove_merge_nodes()

When exporting the tree to GraphViz, we don't need to show the merge nodes explicitly.
Thus this function will remove the merge nodes from the tree.
"""
function remove_merge_nodes()
    # Find all merge nodes (nodes with gate_applied == "MERGE")
    merge_nodes = [node_id for (node_id, node) in EVOLUTION_TREE if node.gate_applied == "MERGE"]

    # For each merge node, connect its parents directly to its children
    for merge_id in merge_nodes
        # Find all edges where this merge node is a parent
        child_edges = [edge for edge in EVOLUTION_EDGES if edge.parent_id == merge_id]
        # Find all edges where this merge node is a child
        parent_edges = [edge for edge in EVOLUTION_EDGES if edge.child_id == merge_id]

        # For each child of the merge node, connect it to all parents of the merge node
        for child_edge in child_edges
            for parent_edge in parent_edges
                # Create new edge from parent to child
                push!(EVOLUTION_EDGES, TreeEdge(
                    parent_edge.parent_id,
                    child_edge.child_id,
                    child_edge.coefficient,
                    child_edge.gate
                ))
            end
        end

        # Remove the merge node and its edges
        delete!(EVOLUTION_TREE, merge_id)
        filter!(edge -> edge.parent_id != merge_id && edge.child_id != merge_id, EVOLUTION_EDGES)
    end
end

"""
    format_tree(format::String, node_shape::String, edge_style::String, graph_title::String, show_coefficients::Bool)

Format the evolution tree into a string representation.
Currently supports "graphviz" format.
"""
function format_tree(format::String,
    node_shape::String="ellipse",
    edge_style::String="solid",
    graph_title::String="Pauli Propagation Graph",
    show_coefficients::Bool=true)

    if format == "graphviz"
        # Create the DOT string with graph properties
        dot_string = """digraph PauliEvolutionTree {
            rankdir=TB;
            node [shape=$node_shape, fontname="Arial"];
            edge [fontname="Arial", fontsize=10];
            label="$graph_title";
            
            // Nodes
        """

        # Add nodes
        for (node_id, node) in EVOLUTION_TREE
            label = node.pauli_string
            if !isnothing(node.gate_applied)
                label *= "\\n($(node.gate_applied))"
            end
            dot_string *= "    \"$node_id\" [label=\"$label\", fontsize=12];\n"
        end

        # Add edges
        dot_string *= "\n    // Edges\n"
        for edge in EVOLUTION_EDGES
            if show_coefficients
                label = edge.coefficient
                dot_string *= "    \"$(edge.parent_id)\" -> \"$(edge.child_id)\" [label=\"$label\", style=$edge_style];\n"
            else
                dot_string *= "    \"$(edge.parent_id)\" -> \"$(edge.child_id)\" [style=$edge_style];\n"
            end
        end

        dot_string *= "}\n"
        return dot_string
    else
        throw(ArgumentError("Unsupported format: $format"))
    end
end


"""
    visualize_tree(output_format::String="graphviz"; 
                  filename::Union{String,Nothing}=nothing, 
                  show_merge_nodes::Bool=false,
                  node_shape::String="ellipse",
                  edge_style::String="solid",
                  graph_title::String="Pauli Propagation Graph",
                  show_coefficients::Bool=true)

Main function for visualizing the evolution tree.
Supports "graphviz", "json", and "summary" formats.
If the filename is provided, the tree will be exported to the file.
Otherwise, the tree will be plotted directly (graphviz) or printed (summary).
"""
function visualize_tree(output_format::String="graphviz";
    filename::Union{String,Nothing}=nothing,
    show_merge_nodes::Bool=false,
    node_shape::String="ellipse",
    edge_style::String="solid",
    graph_title::String="Pauli Propagation Graph",
    show_coefficients::Bool=true,
    scale::Int=18)

    if isempty(EVOLUTION_TREE)
        println("Warning: Evolution tree is empty. Run propagation with PauliTreeTracker coefficients first.")
        return
    end

    if output_format == "graphviz"
        # Save current tree state
        tree_edges_copy = deepcopy(EVOLUTION_EDGES)
        tree_nodes_copy = deepcopy(EVOLUTION_TREE)
        if !show_merge_nodes
            remove_merge_nodes()
        end
        dot_string = format_tree(output_format, node_shape, edge_style, graph_title, show_coefficients)
        if !isnothing(filename)
            open(filename, "w") do io
                println(io, dot_string)
            end
            # restore the tree edge and node
            global EVOLUTION_EDGES = deepcopy(tree_edges_copy)
            global EVOLUTION_TREE = deepcopy(tree_nodes_copy)
            println("Evolution tree exported to $filename")
            println("To visualize, run: dot -Tpng $filename -o pauli_tree.png")
        else
            # save the string to a tmp file, load it and plot it
            tmp_file = tempname() * ".dot"
            open(tmp_file, "w") do io
                println(io, dot_string)
            end
            # restore the tree edge and node
            global EVOLUTION_EDGES = deepcopy(tree_edges_copy)
            global EVOLUTION_TREE = deepcopy(tree_nodes_copy)
            mk, attrs = read_dot_file(tmp_file)
            plot_graphviz(mk, attrs; scale=scale)
        end
    elseif output_format == "json"
        if isnothing(filename)
            throw(ArgumentError("filename is required for json output"))
        end
        export_to_json(filename)
    elseif output_format == "summary"
        print_tree_summary()
    else
        throw(ArgumentError("Unsupported output format: $output_format. Supported formats are 'graphviz', 'json', or 'summary'."))
    end
end

"""
    propagate_with_tree_tracking(circ, input, thetas=nothing; kwargs...)

Convenience function that wraps coefficients in PauliTreeTracker and runs propagation.
Returns both the result and exports the tree visualization.
Supports both PauliString and PauliSum inputs.
"""
function propagate_with_tree_tracking(circ, input, thetas=nothing;
    reset_tree_first::Bool=true,
    kwargs...)

    if reset_tree_first
        reset_tree!()
    end

    # Wrap the coefficients in PauliTreeTracker
    tracked_input = wrapcoefficients(input, PauliTreeTracker)

    # Add the initial nodes
    if input isa PauliString
        pstr_str = format_pauli_string(input)
        for (pstr_key, coeff) in PauliSum(tracked_input)
            add_node!(coeff.node_id, pstr_str, nothing)
            break  # Only one term for PauliString
        end
    else  # PauliSum
        for (pstr, coeff) in tracked_input
            pstr_str = inttostring(pstr, input.nqubits)
            add_node!(coeff.node_id, pstr_str, nothing)
        end
    end

    # Run propagation
    result = propagate(circ, tracked_input, thetas; kwargs...)
    return unwrapcoefficients(result)
end
