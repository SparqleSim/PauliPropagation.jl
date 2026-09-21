module PropagationBase
using LinearAlgebra
using AcceleratedKernels
const AK = AcceleratedKernels
using Base.Threads

include("./utils.jl")
export tonumber

include("./threading_utils.jl")
export maxtasks, withworkers

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
    mapcoeffs!,
    mapcoeffsbypair!,
    set!,
    empty!,
    similar,
    emptylike,
    capacity,
    sortedprefix,
    setsortedprefix!,
    mergefunc

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

# MultiSumStorage is a storage trait, so its specializations of the primitive
# operations below are loaded with those operations.
include("./MultiSum/MultiSum.jl")
export
    MultiSumStorage,
    ZoneMap,
    zones,
    zonemap,
    zonestorage,
    defaultnzones,
    zonecaches,
    outboxes,
    nzones,
    zonesizes,
    zoneof

include("./Primitives/Primitives.jl")
export
    mapterms,
    mapterms!,
    mapcoeffs,
    mapcoeffs!,
    mapcoeffsbypair!,
    sortterms,
    sortterms!,
    sortcoeffs,
    sortcoeffs!,
    filterterms,
    filterterms!,
    filtercoeffs,
    filtercoeffs!,
    mapreducecoeffs,
    maxabscoeff,
    flatmap,
    flatmap!,
    mapandtruncate!

include("./vectorbackend.jl")
export
    flag!,
    flagterms!,
    flagcoeffs!,
    flagstoindices!,
    permuteviaindices!,
    filterviaflags!,
    coeffcumsum,
    coeffcumsum!

include("./gates.jl")
export
    Gate,
    StaticGate,
    ParametrizedGate,
    countparameters

include("./countterms.jl")

include("./propagate.jl")
export propagate,
    propagate!,
    applymergetruncate!,
    applytoall!,
    apply,
    requiresmerging

include("./truncate.jl")
export truncate, truncate!

include("./xorbranch.jl")
export
    xorbranch,
    xorbranch!,
    Unchanged,
    Kept,
    Branch

include("./Merge/Merge.jl")
export
    merge,
    merge!,
    mergeandtruncate!,
    xormerge!,
    xormergeandtruncate!

include("./MonteCarlo/MonteCarlo.jl")
export
    mcpropagate,
    mcpropagate!,
    applymergetruncateresample!,
    mcsample,
    mcsample!,
    mcapplytoall!,
    resample,
    resample!,
    mapslots!,
    multinomial_resample!,
    systematic_resample!,
    semideterministic_systematic_resample!

end
