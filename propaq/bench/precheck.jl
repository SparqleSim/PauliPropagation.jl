###
##
# Propaq's emit precheck tried on the library's fused path: a product whose coefficient is already
# below the cutoff, `|c sin θ| < min_abs_coeff`, is never written, sorted or merged.
#
# Propaq holds such a branch back and applies it after all if its partner turns out to exist; the
# fused path has no cheap way to do that, so this variant simply drops it. What that changes in
# the term count, in the expectation value and in the run time is what this measures, against the
# exact path on the same saved circuit.
#
#   julia --project=. -t8 propaq/bench/precheck.jl circuits/6x6_steps16.json circuits/6x6_steps18.json
#
# `CUTOFF`, `ZONES` and `REPEATS` come from the environment.
##
###

using PauliPropagation
using PauliPropagation.PropagationBase
using Printf

const PERF = PauliPropagation.Performance

include(joinpath(@__DIR__, "papercircuits.jl"))

# the cutoff the precheck applies, or zero for the exact path
const PRECHECK = Ref(0.0)

# `_fusedbranchwrite!` with the precheck; replaces the library's for the duration of this script
PERF.eval(quote
    @inline function _fusedbranchwrite!(out_terms, out_coeffs, new_start, terms, coeffs, lo, hi,
        gate_mask::TT, kept_val, new_val, max_weight, ::Val{GateType}, ::Val{DoWrite}) where {TT,GateType,DoWrite}

        new_pos = new_start
        precheck = $PRECHECK[]

        GC.@preserve terms begin
            bytes = _bytesof(terms, gate_mask)

            @inbounds for ii in lo:hi
                does_commute = _gatecommutes(gate_mask, terms, bytes, ii)
                _branchcondition(Val(GateType), does_commute) || continue

                pstr = terms[ii]
                coeff = coeffs[ii]
                DoWrite && (coeffs[ii] = coeff * kept_val)

                new_coeff = coeff * new_val
                abs(new_coeff) < precheck && continue

                new_pstr, sign = _gateproduct(gate_mask, pstr, bytes, ii)
                if !_truncateweight(new_pstr, max_weight)
                    new_pos = PropagationBase._writeandadvance!(out_terms, out_coeffs, new_pos, new_pstr, new_coeff * sign, Val(DoWrite))
                end
            end
        end

        return new_pos - new_start
    end
end)

function main(files; cutoff::Float64, n_zones::Int, repeats::Int)
    for file in files
        circuit, thetas, obs, steps = loadproblem(file)
        for (label, precheck) in (("exact", 0.0), ("precheck", cutoff))
            PRECHECK[] = precheck
            elapsed, psum = best((c, o, t) -> PERF.propagate(c, MultiPauliSum(VectorPauliSum(o), n_zones), t; min_abs_coeff=cutoff),
                circuit, obs, thetas; repeats)
            @printf("{\"path\":\"fusedmulti %s\",\"file\":\"%s\",\"steps\":%d,\"threads\":%d,\"zones\":%d,\"wall_s\":%.4f,\"n_terms\":%d,\"expectation_value\":%.12g}\n",
                label, basename(file), steps, Threads.nthreads(), n_zones, elapsed, length(psum), overlapwithzero(psum))
            flush(stdout)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS; cutoff=parse(Float64, get(ENV, "CUTOFF", "1e-6")), n_zones=parse(Int, get(ENV, "ZONES", "32")),
        repeats=parse(Int, get(ENV, "REPEATS", "2")))
end
