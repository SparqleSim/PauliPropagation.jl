using PauliPropagation
using Test
using Random

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

    # GPU tests. We only run them when CUDA.jl is installed AND functional.
    # Loading CUDA and the functional() check are wrapped in try/catch so a
    # missing or broken install only warns and skips. The include() that runs
    # the actual tests is deliberately OUTSIDE the try/catch, so a genuine
    # failure inside the GPU tests propagates and fails the suite rather than
    # being swallowed and reported as "CUDA not available".
    cuda_ready = try
        cuda = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
        if cuda.functional()
            true
        else
            @warn "CUDA.jl is installed but not functional; skipping GPU tests."
            false
        end
    catch
        @warn "CUDA.jl is not installed; skipping GPU tests."
        false
    end

    if cuda_ready
        include("test_ntuple_pauli_string_cuda.jl")
    end

end