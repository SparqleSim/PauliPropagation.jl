# Pauli Evolution Visualization

This module provides tree-like visualization of Pauli string evolution during quantum circuit propagation.

## Overview

Tracks Pauli string transformations through gates as a genealogy tree:
- **Nodes**: Individual Pauli strings (e.g., "XXZ", "IYI")  
- **Edges**: Transformation coefficients (e.g., "cos(θ)", "-sin(θ)")

## Key Components

### `PauliTreeTracker`
A `PathProperties` type that wraps coefficients with tree tracking:
- `coeff`: Numerical coefficient
- `node_id`: Unique identifier  
- `parent_id`: Parent node reference

### Tree Storage
- `EVOLUTION_TREE`: Global storage for tree nodes
- `EVOLUTION_EDGES`: Global storage for tree edges

## Main Functions

- `propagate_with_tree_tracking()`: Run propagation with automatic tree tracking
- `visualize_tree()`: Export tree in GraphViz, JSON, or summary format
- `export_to_graphviz()`: Export to DOT format for GraphViz rendering
- `export_to_json()`: Export to JSON format  
- `print_tree_summary()`: Print tree statistics
- `reset_tree!()`: Clear tree storage

## Basic Usage

```julia
using PauliPropagation

# Create circuit and initial state
circ = [PauliRotation(:X, 1), PauliRotation(:Z, 1)]
pstr = PauliString(1, :Z, 1, 1.0)
thetas = [π/4, π/6]

# Run with tree tracking
result = propagate_with_tree_tracking(circ, pstr, thetas)

# Visualize the tree
visualize_tree("graphviz", "tree.dot")  # GraphViz format
visualize_tree("json", "tree.json")     # JSON format  
visualize_tree("summary")               # Print summary
```

## Manual Tree Tracking

```julia
# Wrap existing coefficients for tracking
tracked_psum = wrapcoefficients(my_psum, PauliTreeTracker)
result = propagate(circ, tracked_psum, thetas)

# Visualize the result
visualize_tree("summary")
```

## Output Formats

- **GraphViz (.dot)**: Can be rendered with `dot -Tpng file.dot -o image.png`
- **JSON**: Machine-readable format with 4-space indentation
- **Summary**: Text overview with statistics and node lists

## Notes

- Tree tracking adds computational overhead
- Supports Pauli rotations, Clifford gates, and noise channels
- Merge operations create special merge nodes in the tree
- Memory usage scales with circuit depth and Pauli string count 