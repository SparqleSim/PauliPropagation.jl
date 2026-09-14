###
##
# What the transposed index and the term table cost to build and to read, per term, next to the
# scan they would replace.
#
# The library's merge writes the sum back in a new order after every rotation, so an index over
# positions is only usable there if it can be rebuilt every gate. This times a full rebuild of both
# indices over the final sum of a saved circuit against one scan of the fused path over the same
# sum, on one thread.
#
#   julia --project=. -t1 propaq/bench/indexcost.jl circuits/6x6_steps18.json
##
###

using PauliPropagation
using PauliPropagation.PropagationBase
using Printf

const PERF = PauliPropagation.Performance

include(joinpath(@__DIR__, "papercircuits.jl"))
include(joinpath(@__DIR__, "..", "src", "IndexedPropagation.jl"))
using .IndexedPropagation
using .IndexedPropagation: TransposedIndex, TermTable, appendterms!, refill!, columnsof, _markbranching!

function main(file::String; cutoff::Float64)
    circuit, thetas, obs, _ = loadproblem(file)
    psum = PERF.propagate(circuit, VectorPauliSum(obs), thetas; min_abs_coeff=cutoff, thread=false)
    n = length(psum)
    trms, coeffs = terms(psum), coefficients(psum)
    nq = nqubits(psum)

    # a ZZ rotation in the middle of the grid, as in the circuit
    gate = PauliRotation([:Z, :Z], [nq ÷ 2, nq ÷ 2 + 1])
    plain_mask = PauliPropagation.symboltoint(paulitype(psum), gate.symbols, gate.qinds)
    gate_mask = PERF._gatemask(plain_mask, trms)

    # the scan of the fused path, counting only, so nothing is appended or grown
    scan() = PERF._fusedbranchwrite!(trms, coeffs, 1, trms, coeffs, 1, n, gate_mask, 1.0, 0.0, Inf, Val(:PauliRotation), Val(false))
    scan()
    t_scan = @elapsed n_branching = scan()

    index = TransposedIndex(nq, n)
    appendterms!(index, trms, n)
    empty!(index)
    t_index = @elapsed appendterms!(index, trms, n)

    marks = zeros(UInt64, cld(n, 64))
    cols = columnsof(plain_mask)
    _markbranching!(marks, index, cols, n, Val(:PauliRotation))
    t_mark = @elapsed n_marked = _markbranching!(marks, index, cols, n, Val(:PauliRotation))
    n_marked == n_branching || error("the index marks $n_marked terms, the scan $n_branching")

    table = TermTable(n)
    refill!(table, trms, n)
    empty!(table)
    t_table = @elapsed refill!(table, trms, n)

    @printf("%s: %d terms, %d qubits, %.2f%% branch under %s\n", basename(file), n, nq, 100n_branching / n, gate)
    @printf("  fused scan, test every term       %7.3f s   %6.2f ns/term\n", t_scan, 1e9t_scan / n)
    @printf("  transposed index, build           %7.3f s   %6.2f ns/term   %d B/term\n", t_index, 1e9t_index / n, 8 * length(index.words) ÷ n)
    @printf("  transposed index, mark            %7.3f s   %6.2f ns/term\n", t_mark, 1e9t_mark / n)
    @printf("  term table, build                 %7.3f s   %6.2f ns/term   %d B/term\n", t_table, 1e9t_table / n, 8 * length(table.slots) ÷ n)
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1]; cutoff=parse(Float64, get(ENV, "CUTOFF", "1e-6")))
end
