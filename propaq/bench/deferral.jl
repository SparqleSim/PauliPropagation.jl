###
##
# What Clifford deferral is worth on the library's fused path: the same saved circuit propagated
# once as it is, with every Clifford gate rewriting every term, and once through a `CliffordFrame`
# that absorbs the Clifford gates and conjugates the rotations' generators instead.
#
#   julia --project=. -t4 propaq/bench/deferral.jl circuits/6x6cx_steps12.json
#
# `CUTOFF`, `ZONES` and `REPEATS` come from the environment.
##
###

using PauliPropagation
using Printf

const PERF = PauliPropagation.Performance

include(joinpath(@__DIR__, "papercircuits.jl"))
include(joinpath(@__DIR__, "..", "src", "cliffordframe.jl"))

# the symbols and qubit indices of an integer Pauli string, for a `PauliRotation` around it
function rotationaround(pstr::TT, nqubits::Int) where {TT}
    qinds = [q for q in 1:nqubits if getpauli(pstr, q) != 0]
    symbols = [(:X, :Y, :Z)[getpauli(pstr, q)] for q in qinds]
    return PauliRotation(symbols, qinds)
end

function propagatedeferred(circuit, thetas, obs, cutoff::Float64, n_zones::Int)
    nq = nqubits(obs)
    TT = getinttype(nq)
    frame = CliffordFrame(TT, nq)
    prop_cache = PropagationCache(MultiPauliSum(VectorPauliSum(obs), n_zones))

    # the circuit is applied back to front, and so are its parameters
    theta_index = length(thetas)
    for gate in Iterators.reverse(circuit)
        if gate isa CliffordGate
            compose!(frame, gate)
        elseif gate isa PauliRotation
            theta = thetas[theta_index]
            theta_index -= 1
            generator, sign = conjugategenerator(frame, symboltoint(TT, gate.symbols, gate.qinds))
            PERF.propagate!([rotationaround(generator, nq)], prop_cache, [sign * theta]; min_abs_coeff=cutoff)
        else
            throw(ArgumentError("$(typeof(gate)) is not handled here"))
        end
    end

    # readout through the frame: a term contributes if its image is diagonal
    value = 0.0
    for (pstr, coeff) in VectorPauliSum(prop_cache)
        image, sign = mapterm(frame, pstr)
        isdiagonal = all(q -> getpauli(image, q) in (0, 3), 1:nq)
        isdiagonal && (value += sign * coeff)
    end
    return value, length(prop_cache)
end

function main(files; cutoff::Float64, n_zones::Int, repeats::Int)
    for file in files
        circuit, thetas, obs, steps = loadproblem(file)
        n_clifford = count(g -> g isa CliffordGate, circuit)

        t_plain, psum = best((c, o, t) -> PERF.propagate(c, MultiPauliSum(VectorPauliSum(o), n_zones), t; min_abs_coeff=cutoff),
            circuit, obs, thetas; repeats)
        t_deferred, (value, n_terms) = best((c, o, t) -> propagatedeferred(c, t, o, cutoff, n_zones), circuit, obs, thetas; repeats)

        @printf("{\"path\":\"fusedmulti\",\"file\":\"%s\",\"steps\":%d,\"threads\":%d,\"zones\":%d,\"wall_s\":%.4f,\"n_terms\":%d,\"expectation_value\":%.12g,\"n_clifford\":%d}\n",
            basename(file), steps, Threads.nthreads(), n_zones, t_plain, length(psum), overlapwithzero(psum), n_clifford)
        @printf("{\"path\":\"fusedmulti deferred\",\"file\":\"%s\",\"steps\":%d,\"threads\":%d,\"zones\":%d,\"wall_s\":%.4f,\"n_terms\":%d,\"expectation_value\":%.12g,\"n_clifford\":%d}\n",
            basename(file), steps, Threads.nthreads(), n_zones, t_deferred, n_terms, value, n_clifford)
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS; cutoff=parse(Float64, get(ENV, "CUTOFF", "1e-6")), n_zones=parse(Int, get(ENV, "ZONES", "32")),
        repeats=parse(Int, get(ENV, "REPEATS", "2")))
end
