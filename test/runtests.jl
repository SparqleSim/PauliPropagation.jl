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

end
