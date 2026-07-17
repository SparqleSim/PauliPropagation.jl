module PropagationBase
using LinearAlgebra
using AcceleratedKernels
const AK = AcceleratedKernels
using Base.Threads

include("./utils.jl")
export tonumber, maxtasks

include("./termsum.jl")
export
    AbstractTermSum,
    storage,
    StorageType,
    terms,
    coefficients,
    coeffs,
    termtype,
    coefftype,
    numcoefftype,
    getcoeff,
    getmergedcoeff,
    nsites,
    add!,
    mult!,
    set!,
    empty!,
    similar,
    capacity,
    sortedprefix,
    setsortedprefix!

include("./propagationcache.jl")
export
    AbstractPropagationCache,
    PropagationCache,
    mainsum,
    auxsum,
    extractsum!,
    setmainsum!,
    setauxsum!,
    swapsums!,
    copyswapsums!,
    activesize,
    setactivesize!,
    activesum,
    activeterms,
    activecoeffs,
    activeauxterms,
    activeauxcoeffs,
    flags,
    indices,
    activeflags,
    activeindices,
    lastactiveindex,
    resize!

include("./gates.jl")
export
    Gate,
    StaticGate,
    ParametrizedGate,
    countparameters

include("./propagate.jl")
export propagate,
    propagate!,
    applymergetruncate!,
    applytoall!,
    apply,
    requiresmerging

include("./merge.jl")
export merge, merge!, mergefunc

include("./truncate.jl")
export truncate, truncate!


include("./vectorbackend.jl")
export
    sortbyterm!,
    flag!,
    flagterms!,
    flagcoeffs!,
    flagstoindices!,
    permuteviaindices!,
    filterviaflags!

include("./sortedtailmerge.jl")

end