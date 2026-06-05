# ext/PauliPropagationCUDA.jl
module PauliPropagationCUDA

using PauliPropagation
const PP = PauliPropagation
using PauliPropagation.PropagationBase
const PB = PauliPropagation.PropagationBase
using CUDA

# TODO: export utilities
const CUDAVectorPauliSum = VectorPauliSum{<:CuArray,<:CuArray}
const CUDAVectorPauliPropagationCache = VectorPauliPropagationCache{CUDAVectorPauliSum,<:CuArray,<:CuArray}

const _UNIFIED = true

function CUDA.cu(psum::VectorPauliSum; unified=_UNIFIED)
    cu_paulis = cu(paulis(psum); unified=unified)
    cu_coeffs = cu(coefficients(psum); unified=unified)
    return VectorPauliSum(nqubits(psum), cu_paulis, cu_coeffs)
end

function CUDA.cu(prop_cache::VectorPauliPropagationCache; unified=_UNIFIED)
    cu_mainsum = cu(mainsum(prop_cache); unified)
    cu_auxsum = cu(auxsum(prop_cache); unified)
    cu_flags = cu(PP.flags(prop_cache); unified)
    cu_indices = cu(PP.indices(prop_cache); unified)
    return VectorPauliPropagationCache(cu_mainsum, cu_auxsum, cu_flags, cu_indices, PP.activesize(prop_cache))
end

function Base.collect(psum::VectorPauliSum)
    return VectorPauliSum(nqubits(psum), collect(paulis(psum)), collect(coefficients(psum)))
end


function Base.show(io::IO, psum::CUDAVectorPauliSum)
    println(io, "CUDA VectorPauliSum ($(nqubits(psum)) qubits, $(length(coefficients(psum))) Paulis)")
end

function Base.show(io::IO, prop_cache::CUDAVectorPauliPropagationCache)
    println(io, "CUDA VectorPauliPropagationCache with active size $(length(prop_cache)) (capacity: $(capacity(prop_cache)))")
end

# TODO: This function is apparently
function PauliPropagation.lastactiveindex(prop_cache::CUDAVectorPauliPropagationCache)
    return CUDA.@allowscalar PP.indices(prop_cache)[PP.activesize(prop_cache)]
end



end