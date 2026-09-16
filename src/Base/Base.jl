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
    xorbranch,
    xorbranch!

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

include("./merge.jl")
export merge, merge!, mergefunc

include("./truncate.jl")
export truncate, truncate!


include("./sortedtailmerge.jl")

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
    zoneof,
    staysinzone,
    applytoallzones!

end
