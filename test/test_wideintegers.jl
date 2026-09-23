using Test
using Random
using PauliPropagation
using PauliPropagation.Performance

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

        for x in samples, y in samples
            bx, by = _big(x), _big(y)
            @test _big(x & y) == bx & by
            @test _big(x | y) == bx | by
            @test _big(x ⊻ y) == bx ⊻ by
            @test _big(~x) == (modulus - 1) - bx
            @test (x == y) == (bx == by)
            @test (x < y) == (bx < by)
            @test (x <= y) == (bx <= by)
            @test isless(x, y) == isless(bx, by)
            @test (x > y) == (bx > by)
            @test _big(x + y) == mod(bx + by, modulus)
            @test _big(x - y) == mod(bx - by, modulus)
            @test min(x, y) == T(min(bx, by))
            if bx == by
                @test hash(x) == hash(y)
                @test isequal(x, y)
            end
        end

        for x in samples
            bx = _big(x)
            @test T(bx) == x
            @test hash(x) == hash(bx) && hash(x, UInt(17)) == hash(bx, UInt(17))
            @test hash(x) == hash(NTupleInteger{N + 2}(x))
            if bx <= typemax(UInt128)
                @test hash(x) == hash(UInt128(bx))
            end
            @test count_ones(x) == count_ones(bx)
            @test count_zeros(x) == 64 * N - count_ones(bx)
            @test trailing_zeros(x) == (iszero(x) ? 64 * N : trailing_zeros(bx))
            @test leading_zeros(x) == 64 * N - ndigits(bx; base=2) + (iszero(x) ? 1 : 0)
            @test x % UInt64 == bx % UInt64
            @test x % UInt8 == bx % UInt8
            @test x % Int16 == bx % Int16
            @test x % UInt128 == bx % UInt128
            @test x % Int128 == bx % Int128
            @test string(x; base=16) == string(bx; base=16)
            @test string(x; base=2, pad=64 * N) == bitstring(x)
            @test string(x; base=10) == string(bx)
            @test length(bitstring(x)) == 64 * N
            @test parse(BigInt, bitstring(x); base=2) == bx
            @test occursin("NTupleInteger{$N}(0x", string(x))
            @test occursin(string(bx; base=16), string(x))
            for k in (0, 1, 7, 63, 64, 65, 130, 64 * N - 1, 64 * N, 64 * N + 3)
                @test _big(x << k) == mod(bx << k, modulus)
                @test _big(x >> k) == bx >> k
                @test _big(x >>> k) == bx >> k
                @test x << k == x << UInt(k) && x >> k == x >> UInt16(k)
                @test _big(x >> -k) == mod(bx << k, modulus)
                @test _big(x << -k) == bx >> k
            end
            @test iszero(x >> typemin(Int)) && iszero(x << typemin(Int)) && iszero(x >> typemax(Int))
        end

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

        for _ in 1:30
            a = rand(rng, T) & used_bits
            b = rand(rng, T) & used_bits
            ps, qs = _paulisof(a, nq), _paulisof(b, nq)

            @test commutes(a, b) == iseven(_anticommutingpairs(ps, qs))
            @test PauliPropagation._calculatesignexponent(a, b) == _signexponent(ps, qs)
            product, sign = pauliprod(a, b)
            @test product == a ⊻ b
            @test sign == im^_signexponent(ps, qs)
            @test last(PauliPropagation.paulirotationproduct(a, b)) == (_signexponent(ps, qs) & 2) - 1
            @test countweight(a) == count(!=(0), ps)
            @test countxy(a) == count(p -> p == 1 || p == 2, ps)
            @test countyz(a) == count(p -> p == 2 || p == 3, ps)
            @test countx(a) == count(==(1), ps)
            @test county(a) == count(==(2), ps)
            @test countz(a) == count(==(3), ps)
            @test containsXorY(a) == any(p -> p == 1 || p == 2, ps)

            # the bits above the qubits are never read by a Pauli read
            high = a | ~used_bits
            @test _paulisof(high, nq) == ps

            q = rand(rng, 1:nq)
            @test getpauli(a, q) isa UInt64
            @test getpauli(a, q) == ps[q]
            for pauli in 0:3
                set = setpauli(a, pauli, q)
                @test getpauli(set, q) == pauli
                @test set ⊻ a == T(pauli ⊻ ps[q]) << (2 * (q - 1))
            end
            @test setpauli(a, :Y, q) == setpauli(a, 2, q)

            # several Paulis gathered into the low bits, up to 32 in one word and more in the full type, and windows
            # of up to 32 and of more Paulis
            qinds = shuffle(rng, 1:nq)[1:5]
            packed = getpauli(a, qinds)
            @test packed isa T
            @test [Int(getpauli(packed, i)) for i in 1:5] == ps[qinds]
            @test getpauli(a, Tuple(qinds)) == packed
            many_qinds = shuffle(rng, 1:nq)[1:40]
            @test [Int(getpauli(getpauli(a, many_qinds), i)) for i in 1:40] == ps[many_qinds]
            q1 = rand(rng, 1:nq-31)
            window = getpauli(a, q1, q1 + 31)
            @test window isa T
            @test [Int(getpauli(window, i)) for i in 1:32] == ps[q1:q1+31]
            @test getpauli(a, q1, q1) == ps[q1]
            long_window = getpauli(a, q1, nq)
            @test long_window isa T
            @test [Int(getpauli(long_window, i)) for i in 1:nq-q1+1] == ps[q1:nq]
            @test getpauli(a, 1, nq) == a

            set = setpauli(a, packed, qinds)
            @test [Int(getpauli(set, i)) for i in qinds] == ps[qinds]
            set = setpauli(a, [:X, :Y, :Z, :I, :X], qinds)
            @test [Int(getpauli(set, i)) for i in qinds] == [1, 2, 3, 0, 1]
            window_set = setpauli(a, UInt64(0b1101), q1, q1 + 1)
            @test getpauli(window_set, q1) == 1 && getpauli(window_set, q1 + 1) == 3
            @test setpauli(a, getpauli(a, q1, q1 + 31), q1, q1 + 31) == a
        end

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

    # a circuit on the first qubits of a wide register gives the same sum as on a narrow one, under
    # a coefficient and a weight truncation
    function _propagated(nq, circuit, thetas, observable_qind, kind; thread=false)
        pstr = PauliString(nq, :Z, observable_qind)
        psum = kind == :dict ? PauliSum(pstr) : VectorPauliSum(pstr)
        if kind == :fused
            return Performance.propagate(circuit, psum, thetas; min_abs_coeff=1e-8, max_weight=5, thread)
        elseif kind == :multi
            return propagate(circuit, MultiPauliSum(psum, 4), thetas; min_abs_coeff=1e-8, max_weight=5, thread)
        end
        return propagate(circuit, psum, thetas; min_abs_coeff=1e-8, max_weight=5, thread)
    end

    function _samesum(narrow, wide)
        if length(narrow) != length(wide)
            return false
        end
        TT = paulitype(wide)
        for (term, coeff) in narrow
            if !(getcoeff(wide, TT(BigInt(term))) ≈ coeff)
                return false
            end
        end
        return true
    end

    @testset "propagation on a wide register matches the narrow one" begin
        rng = MersenneTwister(7)
        circuit = tfitrottercircuit(24, 4)
        push!(circuit, CliffordGate(:CNOT, [3, 4]), CliffordGate(:H, 5), TGate(6), DepolarizingNoise(7, 0.1), AmplitudeDampingNoise(8, 0.2))
        thetas = randn(rng, countparameters(circuit))

        for kind in (:dict, :vec, :fused, :multi), thread in (false, true)
            narrow = _propagated(24, circuit, thetas, 12, kind; thread)
            @test all(countweight(term) <= 5 for term in PB.terms(narrow))
            @test any(countweight(term) == 5 for term in PB.terms(narrow))
            for nq in (65, 100, 300)
                wide = _propagated(nq, circuit, thetas, 12, kind; thread)
                @test paulitype(wide) == getinttype(nq)
                @test _samesum(narrow, wide)
                @test overlapwithzero(wide) ≈ overlapwithzero(narrow)
                @test overlapwithplus(wide) ≈ overlapwithplus(narrow)
            end
        end
    end

    # the same circuit shifted up by an offset, so that gates and terms straddle limb boundaries
    _shiftgate(gate::PauliRotation, offset) = PauliRotation(gate.symbols, gate.qinds .+ offset)
    _shiftgate(gate::CliffordGate, offset) = CliffordGate(gate.symbol, gate.qinds .+ offset)
    _shiftgate(gate::TGate, offset) = TGate(gate.qind + offset)
    _shiftgate(gate::Union{PauliNoise,AmplitudeDampingNoise}, offset) = typeof(gate)(gate.qind + offset)
    _shiftgate(gate::FrozenGate, offset) = FrozenGate(_shiftgate(gate.gate, offset), gate.parameter)
    _shiftedcircuit(circuit, offset) = Gate[_shiftgate(gate, offset) for gate in circuit]

    function _shiftedsum(psum, offset, nq)
        TT = getinttype(nq)
        shifted = PauliSum(nq, Dict{TT,Float64}())
        for (term, coeff) in psum
            set!(shifted, TT(BigInt(term)) << (2 * offset), coeff)
        end
        return shifted
    end

    @testset "gates straddling limb boundaries" begin
        rng = MersenneTwister(11)
        circuit = tfitrottercircuit(24, 4)
        push!(circuit, CliffordGate(:CNOT, [3, 4]), CliffordGate(:H, 5), TGate(6), DepolarizingNoise(7, 0.1), AmplitudeDampingNoise(8, 0.2))
        thetas = randn(rng, countparameters(circuit))
        reference = _propagated(24, circuit, thetas, 12, :dict)

        for offset in (50, 62, 100, 190), kind in (:dict, :vec, :fused, :multi)
            nq = offset + 30
            wide = _propagated(nq, _shiftedcircuit(circuit, offset), thetas, 12 + offset, kind)
            @test _samesum(_shiftedsum(reference, offset, nq), wide)
        end
    end

    @testset "sums and strings on wide registers" begin
        nq = 200
        pstr1 = PauliString(nq, [:X, :Y], [63, 64])
        pstr2 = PauliString(nq, [:Z, :Z], [64, 65], 0.5)
        @test commutes(pstr1, pstr2) == false
        @test pauliprod(pstr1, pstr2) == PauliString(nq, [:X, :X, :Z], [63, 64, 65], 0.5im)
        @test commutator(pstr1, pstr2) == PauliString(nq, [:X, :X, :Z], [63, 64, 65], 2.0im)
        psum = PauliSum([pstr1, pstr2])
        @test length(psum) == 2
        @test getcoeff(psum, [:Z, :Z], [64, 65]) == 0.5
        @test getcoeff(psum, pstr1) == 1.0
        @test getcoeff(psum, 0) == 0.0
        @test getcoeff(VectorPauliSum(psum), pstr1.term) == 1.0
        @test PauliSum(VectorPauliSum(psum)) == psum
        @test psum * pstr2 == pauliprod(psum, PauliSum(pstr2))
        @test length(commutator(psum, psum)) == 0
        @test occursin("1 Pauli term", sprint(show, PauliSum(pstr1)))
        @test occursin("nqubits: 200", sprint(show, psum))
        @test occursin(inttostring(pstr1.term, 20), sprint(show, pstr1))
        @test isapprox(overlapwithzero(psum), 0.5)
        @test isapprox(overlapwithplus(psum), 0)
        @test isapprox(overlapwithcomputational(PauliSum(pstr2), [64, 65]), 0.5)
        @test truncatedampingcoeff(pstr1.term, 1.0, 0.5, 0.09) == false
        @test truncatedampingcoeff(pstr1.term, 1.0, 0.5, 0.2) == true
        @test countweight(psum) == [2, 2]

        # a transfer map gate on qubits of two limbs against the Pauli rotation it was built from
        rotation = PauliRotation([:X, :Y], [64, 65])
        tmap_gate = TransferMapGate(totransfermap(2, [PauliRotation([:X, :Y], [1, 2])], [0.7]), [64, 65])
        thetas = [0.7]
        @test _samesum(propagate([rotation], PauliSum(pstr2), thetas), propagate([tmap_gate], PauliSum(pstr2)))
        @test _samesum(propagate([rotation], VectorPauliSum(pstr2), thetas), propagate([tmap_gate], VectorPauliSum(pstr2)))

        # translation symmetry across limbs
        symmetric = PauliSum([PauliString(nq, :Z, q) for q in 1:nq])
        @test length(translationmerge(symmetric)) == 1
        @test only(coefficients(translationmerge(symmetric))) == nq
        grid = PauliSum([PauliString(nq, [:X, :X], [q, q + 1]) for q in 1:20:nq-1])
        @test length(translationmerge(grid, 20, 10)) == 1
    end

    @testset "multi sums and zones on wide registers" begin
        nq = 300
        T = getinttype(nq)
        psum = VectorPauliSum([PauliString(nq, :Z, q, Float64(q)) for q in 1:nq])
        msum = MultiPauliSum(psum, 8)
        @test nzones(msum) == 8
        @test sum(zonesizes(msum)) == nq
        @test all(zoneof(msum, term) == zone for zone in 1:8 for term in PB.terms(zones(msum)[zone]))
        @test Dict(msum) == Dict(psum)
        mask = T(1) << 100
        @test all(PB.zoneof(msum, term ⊻ mask) - 1 == (PB.zoneof(msum, term) - 1) ⊻ (PB.zoneof(msum, mask) - 1) for term in PB.terms(psum))
    end
end
