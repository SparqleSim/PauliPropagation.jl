using Test
using Random
using PauliPropagation
const PP = PauliPropagation
const PB = PauliPropagation.PropagationBase

# the value of a chunked integer as a BigInt, and the modulus of its width
_big(x::NTupleInteger) = BigInt(x)
_modulus(::Type{NTupleInteger{N}}) where {N} = BigInt(1) << (64 * N)

# the per-qubit truth of the Pauli operations, as a reference that shares no code with the limbs
function _paulisof(pstr, nq)
    return [Int(getpauli(pstr, q)) for q in 1:nq]
end

function _anticommutingpairs(ps, qs)
    return count(p != 0 && q != 0 && p != q for (p, q) in zip(ps, qs))
end

# the exponent of im in the product of two single Paulis, by the table of the Pauli algebra
const _IMPOWER = [0 0 0 0; 0 0 1 3; 0 3 0 1; 0 1 3 0]
_signexponent(ps, qs) = sum(_IMPOWER[p+1, q+1] for (p, q) in zip(ps, qs); init=0) & 3

@testset "NTupleInteger" begin
    @testset "getinttype" begin
        @test getinttype(32) == UInt64
        @test getinttype(33) == UInt128
        @test getinttype(64) == UInt128
        @test getinttype(65) == NTupleInteger{3}
        @test getinttype(96) == NTupleInteger{3}
        @test getinttype(97) == NTupleInteger{4}
        @test getinttype(2500) == NTupleInteger{79}
        @test isbitstype(getinttype(1000))
        @test sizeof(getinttype(1000)) == 8 * 32
        @test Base.aligned_sizeof(getinttype(1000)) == 8 * 32
    end

    @testset "the type is an unsigned integer" begin
        @test NTupleInteger{3} <: Unsigned
        @test PauliPropagation.PauliStringType === Integer
    end

    @testset "integer behaviour at N=$N" for N in (1, 3, 4, 29, 79)
        T = NTupleInteger{N}
        rng = MersenneTwister(N)
        modulus = _modulus(T)
        samples = [rand(rng, T) for _ in 1:40]
        push!(samples, zero(T), one(T), typemax(T), T(5), T(1) << (64 * N - 1), T(3) << 70, T(typemax(UInt64)) << 60, samples[1] ⊻ one(T))

        @test _big(zero(T)) == 0
        @test _big(one(T)) == 1
        @test _big(typemax(T)) == modulus - 1
        @test typemin(T) == zero(T)
        @test iszero(zero(T)) && !iszero(one(T)) && isone(one(T)) && !isone(zero(T))
        @test isodd(T(7)) && iseven(T(8))
        @test T(BigInt(3)) == T(3) == T(UInt8(3)) == T(true) + T(2) == T(UInt128(3))
        @test_throws InexactError T(-1)
        @test_throws InexactError T(modulus)
        @test_throws InexactError T(BigInt(-5))

        # every property is folded over all samples, or all pairs of them, and tested once
        bitwise = comparisons = arithmetic = equal_hashes = true
        for x in samples, y in samples
            bx, by = _big(x), _big(y)
            bitwise &= _big(x & y) == bx & by
            bitwise &= _big(x | y) == bx | by
            bitwise &= _big(x ⊻ y) == bx ⊻ by
            bitwise &= _big(~x) == (modulus - 1) - bx
            comparisons &= (x == y) == (bx == by)
            comparisons &= (x < y) == (bx < by)
            comparisons &= (x <= y) == (bx <= by)
            comparisons &= isless(x, y) == isless(bx, by)
            comparisons &= (x > y) == (bx > by)
            arithmetic &= _big(x + y) == mod(bx + by, modulus)
            arithmetic &= _big(x - y) == mod(bx - by, modulus)
            comparisons &= min(x, y) == T(min(bx, by))
            if bx == by
                equal_hashes &= hash(x) == hash(y)
                equal_hashes &= isequal(x, y)
            end
        end
        @test bitwise
        @test comparisons
        @test arithmetic
        @test equal_hashes

        conversions = hashes = bit_counts = strings = shifts = true
        for x in samples
            bx = _big(x)
            conversions &= T(bx) == x
            hashes &= hash(x) == hash(bx) && hash(x, UInt(17)) == hash(bx, UInt(17))
            hashes &= hash(x) == hash(NTupleInteger{N + 2}(x))
            if bx <= typemax(UInt128)
                hashes &= hash(x) == hash(UInt128(bx))
            end
            bit_counts &= count_ones(x) == count_ones(bx)
            bit_counts &= count_zeros(x) == 64 * N - count_ones(bx)
            bit_counts &= trailing_zeros(x) == (iszero(x) ? 64 * N : trailing_zeros(bx))
            bit_counts &= leading_zeros(x) == 64 * N - ndigits(bx; base=2) + (iszero(x) ? 1 : 0)
            conversions &= x % UInt64 == bx % UInt64
            conversions &= x % UInt8 == bx % UInt8
            conversions &= x % Int16 == bx % Int16
            conversions &= x % UInt128 == bx % UInt128
            conversions &= x % Int128 == bx % Int128
            strings &= string(x; base=16) == string(bx; base=16)
            strings &= string(x; base=2, pad=64 * N) == bitstring(x)
            strings &= string(x; base=10) == string(bx)
            strings &= length(bitstring(x)) == 64 * N
            strings &= parse(BigInt, bitstring(x); base=2) == bx
            strings &= occursin("NTupleInteger{$N}(0x", string(x))
            strings &= occursin(string(bx; base=16), string(x))
            for k in (0, 1, 7, 63, 64, 65, 130, 64 * N - 1, 64 * N, 64 * N + 3)
                shifts &= _big(x << k) == mod(bx << k, modulus)
                shifts &= _big(x >> k) == bx >> k
                shifts &= _big(x >>> k) == bx >> k
                shifts &= x << k == x << UInt(k) && x >> k == x >> UInt16(k)
                shifts &= _big(x >> -k) == mod(bx << k, modulus)
                shifts &= _big(x << -k) == bx >> k
            end
            shifts &= iszero(x >> typemin(Int)) && iszero(x << typemin(Int)) && iszero(x >> typemax(Int))
        end
        @test conversions
        @test hashes
        @test bit_counts
        @test strings
        @test shifts

        # values that fit a machine word hash and compare as that word
        @test hash(T(5)) == hash(5) == hash(UInt64(5)) == hash(T(5), zero(UInt))
        @test hash(T(5), UInt(17)) == hash(5, UInt(17))
        @test T(5) == 5 && 5 == T(5) && T(5) != 6 && T(5) == UInt8(5) && T(5) == UInt128(5) && T(5) == BigInt(5)
        @test isless(T(5), 6) && isless(4, T(5)) && !isless(T(5), 5) && !isless(T(5), -1) && isless(-1, T(5))
        @test T(5) < 6 && 4 < T(5) && T(5) <= 5 && 5 <= T(5) && !(T(5) < -3) && -3 < T(5)
        @test T(5) + 1 == 6 && 1 + T(5) == 6 && T(5) - 1 == 4 && 7 - T(5) == 2
        @test typemax(T) + 1 == 0 && zero(T) - 1 == typemax(T)
        @test max(T(5), T(9)) == 9 && min(T(5), T(9)) == 5 && abs(T(5)) === T(5) && cmp(T(5), T(9)) == -1
        @test Int(T(5)) == 5 && UInt8(T(5)) == 0x05 && UInt128(T(5)) == 5 && UInt64(T(5)) == 5
        @test convert(Int, T(5)) === 5 && convert(BigInt, T(5)) == 5
        @test convert(T, 5) === T(5) && convert(T, T(5)) === T(5)
        @test_throws InexactError UInt8(T(300))
        if N > 1
            wide = one(T) << 70
            @test_throws InexactError Int(wide)
            @test_throws InexactError UInt64(wide)
            @test wide != 0 && !(wide < 5) && !isless(wide, typemax(Int)) && isless(typemax(Int), wide) && !(wide == typemax(UInt128))
            @test UInt128(wide) == UInt128(1) << 70
            @test T(UInt128(1) << 70) == wide
            @test wide == T(BigInt(1) << 70) && BigInt(wide) == BigInt(1) << 70
            @test T(UInt128(1) << 70) % UInt64 == 0
            @test wide % UInt128 == UInt128(1) << 70
        end

        # wrapping remainders from machine integers
        @test (-1) % T == typemax(T)
        @test Int8(-2) % T == typemax(T) - 1
        @test (UInt64(3)) % T == T(3)
        @test (UInt128(1) << 70) % T == (N > 1 ? one(T) << 70 : zero(T))

        # repacking between widths
        M = N + 2
        x = rand(rng, T)
        @test NTupleInteger{M}(x) == x && x == NTupleInteger{M}(x)
        @test NTupleInteger{N}(NTupleInteger{M}(x)) === x
        @test NTupleInteger{M}(x) % T === x
        @test x + one(NTupleInteger{M}) == _big(x) + 1 && x < NTupleInteger{M}(_big(x) + 1)
        @test_throws InexactError NTupleInteger{N}(one(NTupleInteger{M}) << (64 * N))
        @test (one(NTupleInteger{M}) << (64 * N)) % T == zero(T)
        @test isbits(x)

        # the containers a Pauli sum uses
        xs = [rand(rng, T) for _ in 1:200]
        mask = rand(rng, T)
        @test (xs .⊻ mask) == [x ⊻ mask for x in xs]
        @test sort(xs) == T.(sort(_big.(xs)))
        @test issorted(sort(xs; rev=true); rev=true)
        @test length(Set(vcat(xs, xs))) == length(unique(_big.(xs)))
        counts = Dict{T,Int}()
        for x in vcat(xs, xs)
            counts[x] = get(counts, x, 0) + 1
        end
        @test all(==(2), values(counts))
        counts[0] = 1
        @test haskey(counts, zero(T)) && counts[zero(T)] == 1
        sorted = sort(xs)
        @test searchsortedfirst(sorted, 0) == 1
        @test searchsortedfirst(sorted, sorted[10]) == 10
        @test rand(MersenneTwister(1), T, 3) == rand(MersenneTwister(1), T, 3)
        @test length(unique(rand(rng, T, 5))) == 5
        @test occursin("NTupleInteger{$N}(0x$(lpad("2", 16N, '0')))", sprint(show, T[T(1), T(2)]))
    end

    @testset "Pauli operations on $nq qubits" for nq in (65, 100, 928, 2528)
        T = getinttype(nq)
        @test T <: NTupleInteger
        @test PauliPropagation.maxqubits(T) == 32 * (sizeof(T) ÷ 8)
        rng = MersenneTwister(nq)
        used_bits = (one(T) << (2nq)) - one(T)

        # every property is folded over all random strings and tested once
        algebra = counts = reads = writes = true
        for _ in 1:30
            a = rand(rng, T) & used_bits
            b = rand(rng, T) & used_bits
            ps, qs = _paulisof(a, nq), _paulisof(b, nq)

            algebra &= commutes(a, b) == iseven(_anticommutingpairs(ps, qs))
            algebra &= PauliPropagation._calculatesignexponent(a, b) == _signexponent(ps, qs)
            product, sign = pauliprod(a, b)
            algebra &= product == a ⊻ b
            algebra &= sign == im^_signexponent(ps, qs)
            algebra &= last(PauliPropagation.paulirotationproduct(a, b)) == (_signexponent(ps, qs) & 2) - 1
            counts &= countweight(a) == count(!=(0), ps)
            counts &= countxy(a) == count(p -> p == 1 || p == 2, ps)
            counts &= countyz(a) == count(p -> p == 2 || p == 3, ps)
            counts &= countx(a) == count(==(1), ps)
            counts &= county(a) == count(==(2), ps)
            counts &= countz(a) == count(==(3), ps)
            counts &= containsXorY(a) == any(p -> p == 1 || p == 2, ps)

            # the bits above the qubits are never read by a Pauli read
            high = a | ~used_bits
            reads &= _paulisof(high, nq) == ps

            q = rand(rng, 1:nq)
            reads &= getpauli(a, q) isa UInt64
            reads &= getpauli(a, q) == ps[q]
            for pauli in 0:3
                set = setpauli(a, pauli, q)
                writes &= getpauli(set, q) == pauli
                writes &= set ⊻ a == T(pauli ⊻ ps[q]) << (2 * (q - 1))
            end
            writes &= setpauli(a, :Y, q) == setpauli(a, 2, q)

            # several Paulis gathered into the low bits, up to 32 in one word and more in the full type, and windows
            # of up to 32 and of more Paulis
            qinds = shuffle(rng, 1:nq)[1:5]
            packed = getpauli(a, qinds)
            reads &= packed isa T
            reads &= [Int(getpauli(packed, i)) for i in 1:5] == ps[qinds]
            reads &= getpauli(a, Tuple(qinds)) == packed
            many_qinds = shuffle(rng, 1:nq)[1:40]
            reads &= [Int(getpauli(getpauli(a, many_qinds), i)) for i in 1:40] == ps[many_qinds]
            q1 = rand(rng, 1:nq-31)
            window = getpauli(a, q1, q1 + 31)
            reads &= window isa T
            reads &= [Int(getpauli(window, i)) for i in 1:32] == ps[q1:q1+31]
            reads &= getpauli(a, q1, q1) == ps[q1]
            long_window = getpauli(a, q1, nq)
            reads &= long_window isa T
            reads &= [Int(getpauli(long_window, i)) for i in 1:nq-q1+1] == ps[q1:nq]
            reads &= getpauli(a, 1, nq) == a

            set = setpauli(a, packed, qinds)
            writes &= [Int(getpauli(set, i)) for i in qinds] == ps[qinds]
            set = setpauli(a, [:X, :Y, :Z, :I, :X], qinds)
            writes &= [Int(getpauli(set, i)) for i in qinds] == [1, 2, 3, 0, 1]
            window_set = setpauli(a, UInt64(0b1101), q1, q1 + 1)
            writes &= getpauli(window_set, q1) == 1 && getpauli(window_set, q1 + 1) == 3
            writes &= setpauli(a, getpauli(a, q1, q1 + 31), q1, q1 + 31) == a
        end
        @test algebra
        @test counts
        @test reads
        @test writes

        # the two-limb window read never straddles more than the two limbs it uses
        edge = setpauli(setpauli(zero(T), 3, 32), 2, 33)
        @test getpauli(edge, 32, 33) == UInt64(3) | (UInt64(2) << 2)
        @test getpauli(edge, 1, 32) == UInt64(3) << 62
        @test getpauli(edge, 33, 64) == 2
        @test getpauli(edge, 1, 33) == T(3) << 62 | T(2) << 64 && getpauli(edge, 31, 64) == 3 << 2 | 2 << 4

        # symbols and strings
        symbols = rand(rng, [:I, :X, :Y, :Z], nq)
        pstr = symboltoint(symbols)
        @test pstr isa T
        @test inttosymbol(pstr, nq) == symbols
        @test inttostring(pstr, nq) == join(string.(symbols))
        @test symboltoint(nq, symbols[1:3], [1, 33, nq]) == setpauli(setpauli(setpauli(zero(T), symbols[1], 1), symbols[2], 33), symbols[3], nq)
        @test PauliString(nq, [:X, :Z], [64, 65]).term == (T(1) << 126) | (T(3) << 128)
        @test ispauli(getpauli(pstr, 65), symbols[65])
        @test identitylike(pstr) == zero(T)
        @test PauliPropagation.alternatingmask(pstr) == T(BigInt(sum(BigInt(1) << (2k) for k in 0:32*(sizeof(T) ÷ 8)-1)))
        @test PauliPropagation._paulimask(T, 40) == (one(T) << 80) - one(T)
        @test PauliPropagation._paulimask(T, 32 * (sizeof(T) ÷ 8)) == typemax(T)
        @test PauliPropagation._paulimask(T, 0) == zero(T)
        @test PauliPropagation._pauliwindowmask(T, 30, 40) == ((one(T) << 22) - one(T)) << 58
    end

    @testset "grid string with rows wider than a machine word" begin
        pstr = symboltoint(120, [:X, :Y, :Z], [1, 40, 120])
        lines = split(inttostring(pstr, 40, 3), "\n"; keepempty=false)
        @test length(lines) == 3 && all(==(40), length.(lines))
        @test lines[1][1] == 'X' && lines[1][40] == 'Y' && lines[3][40] == 'Z'
        @test count(==('I'), join(lines)) == 117
    end

    @testset "a gate that branches by a Pauli string decides from the limbs it acts on as from the whole string" begin
        # commutation, the sign and the Paulis under the mask add up over the limbs, so for every term, the rule built
        # for and asked about only the limbs a gate acts on, next to each other or far apart, has to decide as the rule
        # built for the whole mask
        wholerule(::PauliRotation, gate_mask) = PP._rotationrule(gate_mask, cos(0.3), sin(0.3))
        wholerule(::ImaginaryPauliRotation, gate_mask) = PP._imaginaryrotationrule(gate_mask, cosh(0.3), sinh(0.3))
        wholerule(::AmplitudeDampingNoise, gate_mask) = PP._dampingrule(gate_mask, 0.3)

        rng = MersenneTwister(5)
        for nq in (100, 1000)
            TT = getinttype(nq)
            terms = rand(rng, TT, 256)
            coeffs = randn(rng, length(terms))
            prop_cache = PP.VectorPauliPropagationCache(VectorPauliSum(nq, copy(terms), copy(coeffs)))

            gates = Any[AmplitudeDampingNoise(qind) for qind in (1, 32, 33, nq)]
            for (symbols, qinds) in (([:X], [1]), ([:Y, :Z], [2, 3]), ([:Z, :X], [32, 33]), ([:X, :Y], [5, nq]), ([:Y], [nq]))
                push!(gates, PauliRotation(symbols, qinds), ImaginaryPauliRotation(symbols, qinds))
            end

            for gate in gates
                gate_mask = PP._branchmask(gate, prop_cache)
                whole_rule = wholerule(gate, gate_mask)
                rule = PP._branchrule(gate, prop_cache, 0.3)
                @test (rule isa PB.OnLimbs) == !isnothing(PB.limbspan(gate_mask))

                # the array kernels ask through `ruleat`, every other storage with the whole term
                @test all(PB.ruleat(rule, terms, coeffs, ii) == whole_rule(terms[ii], coeffs[ii]) for ii in eachindex(terms))
                @test all(rule(terms[ii], coeffs[ii]) == whole_rule(terms[ii], coeffs[ii]) for ii in eachindex(terms))
            end
        end

        # a gate within one limb or across two is asked about those; one on three limbs, or on Pauli strings of at
        # most two limbs, is asked about the whole string
        @test PB.limbspan(symboltoint(getinttype(100), [:X, :Y], [2, 3])) == (1, 1)
        @test PB.limbspan(symboltoint(getinttype(100), [:Z, :X], [32, 33])) == (1, 2)
        @test isnothing(PB.limbspan(symboltoint(getinttype(100), [:X, :Y, :Z], [1, 50, 100])))
        @test isnothing(PB.limbspan(symboltoint(getinttype(40), [:X], [40])))
        @test isnothing(PB.limbspan(symboltoint(getinttype(30), [:X, :Y], [1, 30])))
    end

    @testset "a circuit across a limb boundary propagates as on the qubits of one word" begin
        # the same circuit on qubits 1 to 20 of a machine word, and on qubits 55 to 74 of four limbs, where it straddles
        # the boundary between the second and third limb, must give the same Pauli strings, shifted
        nq_small, nq_wide, offset = 20, 100, 54
        Random.seed!(7)
        circuit = Gate[]
        for _ in 1:3
            append!(circuit, (PauliRotation(:X, q) for q in 1:nq_small))
            append!(circuit, (PauliRotation(:Y, q) for q in 1:nq_small))
            append!(circuit, (PauliRotation([:Z, :Z], [q, q + 1]) for q in 1:nq_small-1))
            append!(circuit, (CliffordGate(:CNOT, [q, q + 1]) for q in 2:4:nq_small-1))
            push!(circuit, PauliRotation([:Y, :X], [1, nq_small]), AmplitudeDampingNoise(nq_small ÷ 2 + 1))
        end
        # in [0, 1), as the damping strength must be
        thetas = rand(countparameters(circuit))

        shiftgate(gate::PauliRotation) = PauliRotation(gate.symbols, gate.qinds .+ offset)
        shiftgate(gate::CliffordGate) = CliffordGate(gate.symbol, gate.qinds .+ offset)
        shiftgate(gate::AmplitudeDampingNoise) = AmplitudeDampingNoise(gate.qind + offset)
        towide(pstr) = getinttype(nq_wide)(pstr) << (2 * offset)

        # the shifted terms of `small` against those of `wide`, whose coefficients `samecoeff` compares
        function sameshifted(small, wide, samecoeff)
            small_dict = Dict(towide(pstr) => coeff for (pstr, coeff) in zip(paulis(small), coefficients(small)))
            wide_dict = Dict(zip(paulis(wide), coefficients(wide)))
            return keys(small_dict) == keys(wide_dict) &&
                   all(samecoeff(small_dict[pstr], wide_dict[pstr]) for pstr in keys(small_dict))
        end

        for T in (PauliSum, VectorPauliSum)
            small = propagate(circuit, T(PauliString(nq_small, :Z, 10)), thetas; min_abs_coeff=0.0)
            wide = propagate(shiftgate.(circuit), T(PauliString(nq_wide, :Z, 10 + offset)), thetas; min_abs_coeff=0.0)
            @test length(small) > 1000
            @test sameshifted(small, wide, (a, b) -> isapprox(a, b; rtol=1e-12, atol=1e-15))
        end

        # the rule of a rotation on path properties reads the limbs as well
        rotations = filter(gate -> gate isa PauliRotation, circuit)
        rotation_thetas = randn(countparameters(rotations))
        small = propagate(rotations, PauliString(nq_small, :Z, 10, PauliFreqTracker(1.0)), rotation_thetas; min_abs_coeff=0.0)
        wide = propagate(shiftgate.(rotations), PauliString(nq_wide, :Z, 10 + offset, PauliFreqTracker(1.0)), rotation_thetas; min_abs_coeff=0.0)
        @test length(small) > 500
        @test sameshifted(small, wide, (a, b) -> a.freq == b.freq && isapprox(tonumber(a), tonumber(b); rtol=1e-12, atol=1e-15))
    end
end
