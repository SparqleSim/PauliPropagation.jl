using PauliPropagation
const PP = PauliPropagation
import PauliPropagation.PropagationBase
const PB = PauliPropagation.PropagationBase
using Test
using Random

@testset "PauliPropagation.jl" begin

    include("test_propagate.jl")

    include("test_schrodinger.jl")

    include("test_datatypes.jl")

    include("test_paulialgebra_utils.jl")

    include("test_wideintegers.jl")

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

    include("test_primitives.jl")

    include("test_mergeandtruncate.jl")

    include("test_gradient.jl")

    include("test_xortailmerge.jl")

    include("test_multipaulisum.jl")

    include("test_montecarlo.jl")

    include("test_performance.jl")

    include("test_numericalcertificates.jl")

    include("test_countpaulis.jl")

    include("test_visualization.jl")

    # the same tests on Base's public `Dict` interface, with the dictionary internals switched off,
    # each file in a module of its own so that its definitions can be made again,
    # and before the Yao files, whose methods the ambiguity check in test_primitives.jl would otherwise see
    @testset "Base Dict interface" begin
        Base.delete_method(which(PB._hasdictinternals, Tuple{Dict}))
        try
            for file in ("test_primitives.jl", "test_montecarlo.jl", "test_gates_against_yao.jl")
                rerun = Module()
                Core.eval(rerun, :(using Test, Random, PauliPropagation; const PP = PauliPropagation; const PB = PauliPropagation.PropagationBase))
                Base.include(rerun, joinpath(@__DIR__, file))
            end
        finally
            @eval PB _hasdictinternals(::Dict) = _DICT_INTERNALS
        end
    end

    include("test_gates_against_yao.jl")

    include("test_yao_extension.jl")

end
