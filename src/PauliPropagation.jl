module PauliPropagation

using LinearAlgebra

# for the VectorPauliSum operations
using AcceleratedKernels
const AK = AcceleratedKernels
using Base.Threads

include("./Base/Base.jl")
using .PropagationBase


include("./PauliDataTypes/PauliDataTypes.jl")
export
    PauliStringType,
    PauliType,
    PauliSum,
    PauliString,
    VectorPauliSum,
    VectorPauliPropagationCache,
    nqubits,
    paulis,
    coefficients,
    norm,
    paulitype,
    coefftype,
    numcoefftype,
    getcoeff,
    topaulistrings,
    mult!,
    add!,
    set!,
    mult!,
    empty!,
    similar,
    convertcoefftype

include("./PauliAlgebra/PauliAlgebra.jl")
export
    identitypauli,
    identitylike,
    inttosymbol,
    symboltoint,
    inttostring,
    ispauli,
    getpauli,
    setpauli,
    countweight,
    countxy,
    countyz,
    countx,
    county,
    countz,
    containsXorY,
    containsYorZ,
    pauliprod,
    commutes,
    commutator,
    trace,
    getinttype

include("PauliTransferMatrix/PauliTransferMatrix.jl")
export
    calculateptm,
    totransfermap

include("Gates/Gates.jl")
export
    Gate,
    ParametrizedGate,
    StaticGate,
    PauliRotation,
    MaskedPauliRotation,
    ImaginaryPauliRotation,
    CliffordGate,
    clifford_map,
    transposecliffordmap,
    reset_clifford_map!,
    createcliffordmap,
    composecliffordmaps,
    ParametrizedNoiseChannel,
    PauliNoise,
    DepolarizingNoise,
    DephasingNoise,
    AmplitudeDampingNoise,
    PauliXNoise,
    PauliYNoise,
    PauliZNoise,
    FrozenGate,
    freeze,
    TGate,
    TransferMapGate,
    tomatrix,
    toschrodinger,
    toheisenberg

include("Circuits/Circuits.jl")
export
    countparameters,
    getparameterindices,
    bricklayertopology,
    staircasetopology,
    rectangletopology,
    staircasetopology2d,
    ibmeagletopology,
    hardwareefficientcircuit,
    efficientsu2circuit,
    tfitrottercircuit,
    tiltedtfitrottercircuit,
    heisenbergtrottercircuit,
    su4circuit,
    qcnncircuit,
    appendSU4!,
    rxlayer!,
    rylayer!,
    rzlayer!,
    rxxlayer!,
    ryylayer!,
    rzzlayer!


include("Propagation/Propagation.jl")
export
    AbstractPauliPropagationCache,
    PauliPropagationCache,
    VectorPauliPropagationCache,
    PropagationCache,
    mainsum,
    auxsum,
    capacity,
    propagate,
    propagate!,
    applymergetruncate!,
    applytoall!,
    apply,
    truncate,
    truncate!,
    merge,
    merge!,
    mergefunc


include("PathProperties/PathProperties.jl")
export
    PathProperties,
    PauliFreqTracker,
    wrapcoefficients,
    unwrapcoefficients

include("truncations.jl")
export
    truncatedampingcoeff


include("stateoverlap.jl")
export
    overlapbyorthogonality,
    overlapwithzero,
    overlapwithplus,
    overlapwithones,
    overlapwithcomputational,
    overlapwithmaxmixed,
    overlapwithpaulisum,
    scalarproduct,
    filter,
    filter!,
    zerofilter,
    zerofilter!,
    plusfilter,
    plusfilter!,
    evaluateagainstdict,
    tonumber

include("numericalcertificates.jl")
export
    estimatemse,
    estimatemse!

include("Symmetry/Symmetry.jl")
export
    symmetrymerge,
    translationmerge

include("Surrogate/Surrogate.jl")
export
    NodePathProperties,
    evaluate!,
    reset!

include("Visualization/Visualization.jl")
export
    PauliTreeTracker,
    TreeNode,
    TreeEdge,
    EVOLUTION_TREE,
    EVOLUTION_EDGES,
    reset_tree!,
    add_node!,
    add_edge!,
    create_child_tracker,
    format_pauli_string,
    export_to_graphviz,
    export_to_json,
    print_tree_summary,
    visualize_tree,
    propagate_with_tree_tracking

# # experimental vector propagation 
# include("Propagation/VectorPropagate/VectorPropagate.jl")

end
