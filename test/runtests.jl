using PauliPropagation
using Test
using Random

const test_cuda = include("setup_cuda.jl")

@testset "PauliPropagation.jl" begin

    include("test_propagate.jl")

    include("test_schrodinger.jl")

    include("test_datatypes.jl")

    include("test_paulialgebra_utils.jl")

    include("test_noisechannels.jl")

    include("test_circuits.jl")

    include("test_cliffordgates.jl")

    include("test_frozengates.jl")

    include("test_miscgates.jl")

    include("test_overlaps.jl")

    include("test_paulirotations.jl")

    include("test_imaginary.jl")

    include("test_paulioperations.jl")

    include("test_paulitransfermaps.jl")

    include("test_pathproperties.jl")

    include("test_symmetries.jl")

    include("test_truncations.jl")

    include("test_inplace.jl")

    include("test_numericalcertificates.jl")

    include("test_visualization.jl")

    include("test_gates_against_yao.jl")

    include("test_yao_extension.jl")

    include("test_ntuple_pauli_string.jl")

    if test_cuda
        if CUDA.functional()
            include("test_cuda_extension.jl")
        else
            @warn "CUDA.jl is installed but not functional; skipping GPU tests."
        end
    end
end
