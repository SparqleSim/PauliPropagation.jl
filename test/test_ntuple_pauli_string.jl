using Test
using Random
using Bits: bitsize, mask as bits_mask

let dir = joinpath(@__DIR__, "..")
    dir ∉ LOAD_PATH && push!(LOAD_PATH, dir)
end

using PauliPropagation

import PauliPropagation: _countbitweight, _countbitxy, _countbityz,
    _countbitx, _countbity, _countbitz, _bitcommutes, _bitpaulimultiply,
    _setpaulibits, _getpaulibits

pauli_to_bits = Dict(:I => 0, :X => 1, :Y => 2, :Z => 3)

function ref_encode(paulis::Vector{Symbol})
    n = length(paulis)
    @assert n <= 32
    v = UInt64(0)
    for (i, p) in enumerate(paulis)
        v |= UInt64(pauli_to_bits[p]) << (2*(i-1))
    end
    return v
end

function ntuple_encode(::Type{NTupleUInt{N,W}}, paulis::Vector{Symbol}) where {N,W}
    T = NTupleUInt{N,W}
    pstr = zero(T)
    for (i, p) in enumerate(paulis)
        pstr = _setpaulibits(pstr, pauli_to_bits[p], i)
    end
    return pstr
end

function ntupleuint_encode(::Type{NTupleUInt{N,W}}, paulis::Vector{Symbol}) where {N,W}
    T = NTupleUInt{N,W}
    pstr = zero(T)
    for (i, p) in enumerate(paulis)
        pstr = _setpaulibits(pstr, pauli_to_bits[p], i)
    end
    return pstr
end


@testset "NTupleUInt" begin

    T32  = NTupleUInt{2,  UInt32}
    T64  = NTupleUInt{2,  UInt64}
    T128 = NTupleUInt{8,  UInt32}
    T256 = NTupleUInt{16, UInt32}

    @testset "Construction and basics" begin
        z = zero(T64)
        @test z == 0
        @test z == zero(T64)

        o = one(T64)
        @test o == 1
        @test o != z

        @test max_qubits(T32)  == 32
        @test max_qubits(T64)  == 64
        @test max_qubits(T128) == 128
        @test max_qubits(T256) == 256

        x = NTupleUInt{2,UInt32}(0xDEADBEEF)
        @test x.data[1] == 0xDEADBEEF
        @test x.data[2] == 0x00000000

        y = NTupleUInt{2,UInt32}(0xDEADBEEF_CAFEBABE)
        @test y.data[1] == 0xCAFEBABE
        @test y.data[2] == 0xDEADBEEF
    end

    @testset "Bitwise operations" begin
        a = NTupleUInt{2,UInt32}((0xAAAAAAAA, 0xAAAAAAAA))
        b = NTupleUInt{2,UInt32}((0x55555555, 0x55555555))

        @test (a & b) == 0
        @test (a | b) == typemax(NTupleUInt{2,UInt32})
        @test (a ⊻ b) == typemax(NTupleUInt{2,UInt32})
        @test (~a)     == b
    end

    @testset "Shift operations" begin
        one64 = NTupleUInt{2,UInt32}(1)
        @test (one64 << 3) == 8
        @test (one64 << 3 >> 3) == 1

        shifted = one64 << 32
        @test shifted.data[1] == 0x00000000
        @test shifted.data[2] == 0x00000001

        @test (shifted >> 32) == one64

        val = NTupleUInt{2,UInt32}((0xFFFFFFFF, 0x00000000))
        shifted2 = val << 1
        @test shifted2.data[1] == 0xFFFFFFFE
        @test shifted2.data[2] == 0x00000001

        @test (one64 << 64) == 0
        @test (one64 >> 64) == 0
    end

    @testset "Comparison and sorting" begin
        a = NTupleUInt{2,UInt32}(10)
        b = NTupleUInt{2,UInt32}(20)
        c = NTupleUInt{2,UInt32}(10)

        @test a < b
        @test b > a
        @test a <= c
        @test a >= c
        @test a == c
        @test a != b

        big   = NTupleUInt{2,UInt32}((0x00000000, 0x00000001))
        small = NTupleUInt{2,UInt32}((0xFFFFFFFF, 0x00000000))
        @test small < big

        arr    = [b, a, big, small, c]
        sorted = sort(arr)
        @test sorted == [a, c, b, small, big]
    end

    @testset "count_ones" begin
        z = zero(T64)
        @test count_ones(z) == 0

        all_ones = typemax(T64)
        @test count_ones(all_ones) == bitsize(T64)

        x = NTupleUInt{2,UInt64}((0xAAAAAAAAAAAAAAAA, 0x0000000000000000))
        @test count_ones(x) == 32
    end

    @testset "alternatingmask" begin
        mask = alternatingmask(T64)
        for i in 0:(bitsize(T64)-1)
            word_idx    = i ÷ 64 + 1
            bit_in_word = i % 64
            bit_val  = (mask.data[word_idx] >> bit_in_word) & one(UInt64)
            expected = (i % 2 == 0) ? one(UInt64) : zero(UInt64)
            @test bit_val == expected
        end
    end

    @testset "Pauli count functions" begin
        paulis7 = [:X, :Y, :Z, :I, :X, :Y, :Z]
        pstr  = ntuple_encode(NTupleUInt{1,UInt32}, paulis7)
        ref32 = UInt32(ref_encode(paulis7))

        @test _countbitweight(pstr) == _countbitweight(ref32)
        @test _countbitxy(pstr)     == _countbitxy(ref32)
        @test _countbityz(pstr)     == _countbityz(ref32)
        @test _countbitx(pstr)      == _countbitx(ref32)
        @test _countbity(pstr)      == _countbity(ref32)
        @test _countbitz(pstr)      == _countbitz(ref32)

        @test _countbitweight(pstr) == 6
        @test _countbitx(pstr)      == 2
        @test _countbity(pstr)      == 2
        @test _countbitz(pstr)      == 2
    end

    @testset "Commutation" begin
        pX = ntuple_encode(NTupleUInt{1,UInt32}, [:X])
        pY = ntuple_encode(NTupleUInt{1,UInt32}, [:Y])
        pZ = ntuple_encode(NTupleUInt{1,UInt32}, [:Z])
        pI = ntuple_encode(NTupleUInt{1,UInt32}, [:I])

        @test !_bitcommutes(pX, pY)
        @test !_bitcommutes(pY, pZ)
        @test !_bitcommutes(pX, pZ)
        @test  _bitcommutes(pX, pX)
        @test  _bitcommutes(pX, pI)

        pXX = ntuple_encode(NTupleUInt{1,UInt32}, [:X, :X])
        pYY = ntuple_encode(NTupleUInt{1,UInt32}, [:Y, :Y])
        @test _bitcommutes(pXX, pYY)

        pXZ = ntuple_encode(NTupleUInt{1,UInt32}, [:X, :Z])
        pZX = ntuple_encode(NTupleUInt{1,UInt32}, [:Z, :X])
        @test _bitcommutes(pXZ, pZX)
    end

    @testset "Pauli product" begin
        pX = ntuple_encode(NTupleUInt{1,UInt32}, [:X])
        pY = ntuple_encode(NTupleUInt{1,UInt32}, [:Y])
        pZ = ntuple_encode(NTupleUInt{1,UInt32}, [:Z])
        pI = ntuple_encode(NTupleUInt{1,UInt32}, [:I])

        @test _bitpaulimultiply(pX, pX) == pI
        @test _bitpaulimultiply(pY, pY) == pI
        @test _bitpaulimultiply(pZ, pZ) == pI
        @test _bitpaulimultiply(pX, pY) == pZ
    end

    @testset "getpauli / setpauli round-trip" begin
        T    = NTupleUInt{4,UInt32}
        pstr = zero(T)
        syms = [:X, :Y, :Z, :I, :X, :Z, :Y]

        for (qind, sym) in enumerate(syms)
            pstr = _setpaulibits(pstr, pauli_to_bits[sym], qind)
        end

        for (qind, sym) in enumerate(syms)
            got = _getpaulibits(pstr, qind)
            @test Int(UInt64(got)) == pauli_to_bits[sym]
        end
    end

    @testset "Cross-validation vs UInt64 for small qubit counts" begin
        nq         = 15
        paulis_ref = [:X, :Y, :Z, :I, :X, :Y, :Z, :I, :X, :Y, :Z, :I, :X, :Y, :Z]
        ref        = UInt64(ref_encode(paulis_ref))
        ntp        = ntuple_encode(NTupleUInt{2,UInt32}, paulis_ref)

        @test _countbitweight(ntp) == _countbitweight(ref)
        @test _countbitxy(ntp)     == _countbitxy(ref)
        @test _countbityz(ntp)     == _countbityz(ref)
        @test _countbitx(ntp)      == _countbitx(ref)
        @test _countbity(ntp)      == _countbity(ref)
        @test _countbitz(ntp)      == _countbitz(ref)
    end

    @testset "256-qubit smoke test" begin
        T  = NTupleUInt{16,UInt32}
        nq = 256
        @test max_qubits(T) == 256

        paulis_large = [isodd(i) ? :X : :Z for i in 1:nq]
        pstr = ntuple_encode(T, paulis_large)

        @test _countbitx(pstr)      == 128
        @test _countbitz(pstr)      == 128
        @test _countbity(pstr)      == 0
        @test _countbitweight(pstr) == 256

        pX_all = ntuple_encode(T, fill(:X, nq))
        pZ_all = ntuple_encode(T, fill(:Z, nq))
        @test _bitcommutes(pX_all, pZ_all)
    end

    @testset "getchunkedinttype factory" begin
        @test getchunkedinttype(16;  word=UInt32) == NTupleUInt{1,  UInt32}
        @test getchunkedinttype(64;  word=UInt32) == NTupleUInt{4,  UInt32}
        @test getchunkedinttype(256; word=UInt32) == NTupleUInt{16, UInt32}
        @test getchunkedinttype(128; word=UInt64) == NTupleUInt{4,  UInt64}
    end

    @testset "getinttype chunked flag" begin
        @test getinttype(64;  chunked=true, word=UInt32) == NTupleUInt{4,  UInt32}
        @test getinttype(256; chunked=true, word=UInt32) == NTupleUInt{16, UInt32}
    end

    @testset "propagate() matches BitIntegers baseline" begin
        Random.seed!(42)

        for nq in (32, 64, 128)
            nl = 2
            topo  = bricklayertopology(nq; periodic=false)
            circ  = hardwareefficientcircuit(nq, nl; topology=topo)
            thetas = randn(length(circ))

            pstr_ref   = PauliString(nq, :Z, div(nq, 2))
            result_ref = overlapwithzero(propagate(circ, pstr_ref, thetas))

            TT = getchunkedinttype(nq; word=UInt32)
            vpsum = VectorPauliSum(Float64, nq, TT)
            push!(paulis(vpsum), symboltoint(TT, [:Z], [div(nq, 2)]))
            push!(coefficients(vpsum), 1.0)
            result_ntuple = overlapwithzero(propagate(circ, vpsum, thetas))

            @test result_ref ≈ result_ntuple atol=1e-10 rtol=1e-10
        end
    end

    @testset "propagate() 256-qubit NTupleUInt completes" begin
        Random.seed!(99)

        nq = 256
        nl = 1
        topo  = bricklayertopology(nq; periodic=false)
        circ  = hardwareefficientcircuit(nq, nl; topology=topo)
        thetas = randn(length(circ))

        TT = getchunkedinttype(nq; word=UInt32)
        vpsum = VectorPauliSum(Float64, nq, TT)
        push!(paulis(vpsum), symboltoint(TT, [:Z], [div(nq, 2)]))
        push!(coefficients(vpsum), 1.0)

        result = overlapwithzero(propagate(circ, vpsum, thetas))
        @test isfinite(result)
    end

end


# ── NTupleUInt tests ────────────────────────────────────────────────────────────
# NTupleUInt{N,T} <: Unsigned integrates into getinttype automatically (no opt-in)
# and slots into the existing PauliStringType = Integer union without widening it.

@testset "NTupleUInt" begin

    M64  = NTupleUInt{4, UInt64}   # 256 bits via UInt64 words
    M32  = NTupleUInt{8, UInt32}   # 256 bits via UInt32 words (consumer GPU width)

    @testset "Construction and isbitstype" begin
        @test isbitstype(M64)
        @test isbitstype(M32)
        @test zero(M64).parts == ntuple(_ -> UInt64(0), 4)
        @test one(M64).parts[1] == UInt64(1)
        @test iszero(zero(M64)) && !iszero(one(M64))
        @test typemax(M64).parts == ntuple(_ -> typemax(UInt64), 4)
    end

    @testset "Bitwise operations" begin
        a = M64((UInt64(0xAAAA_AAAA_AAAA_AAAA), UInt64(0xAAAA_AAAA_AAAA_AAAA), 0x0, 0x0))
        b = M64((UInt64(0x5555_5555_5555_5555), UInt64(0x5555_5555_5555_5555), 0x0, 0x0))
        @test (a & b) == zero(M64)
        @test (~zero(M64)).parts == ntuple(_ -> typemax(UInt64), 4)
        @test (a ⊻ b) == M64((typemax(UInt64), typemax(UInt64), 0x0, 0x0))
    end

    @testset "Shifts across word boundaries" begin
        o = one(M64)
        @test (o << 64).parts  == (0x0, 0x1, 0x0, 0x0)
        @test (o << 255).parts == (0x0, 0x0, 0x0, UInt64(0x8000_0000_0000_0000))
        @test (o << 256) == zero(M64)
        b = M64((0x0, 0x0, 0x0, UInt64(1)))
        @test (b >> 192) == one(M64)
        @test (b >> 193) == zero(M64)
    end

    @testset "count_ones + ordering" begin
        @test count_ones(zero(M64)) == 0
        @test count_ones(~zero(M64)) == 256
        @test isless(one(M64), one(M64) << 1)
        @test !isless(M64((0x0, 0x1, 0x0, 0x0)), one(M64))
    end

    @testset "equality with integers" begin
        @test M64(42) == 42
        @test 42 == M64(42)
        @test !(M64(42) == 43)
        @test hash(zero(M64)) == hash(0)
    end

    @testset "alternatingmask (compile-time constant)" begin
        am = alternatingmask(M64)
        @test count_ones(am) == 128   # half of 256
        o = one(M64)
        @test (am & (o << 0)) != zero(M64)   # bit 0 (even) set
        @test (am & (o << 1)) == zero(M64)   # bit 1 (odd) clear
        @test (am & (o << 254)) != zero(M64) # bit 254 (even) set
        @test (am & (o << 255)) == zero(M64) # bit 255 (odd) clear
    end

    @testset "arithmetic (ripple-carry)" begin
        @test one(M64) + one(M64) == M64(2)
        # Carry across word boundary: typemax(UInt64) + 1 should carry into word 2
        @test M64(typemax(UInt64)) + one(M64) == M64((0x0, 0x1, 0x0, 0x0))
        @test M64(5) - M64(3) == M64(2)
    end

    @testset "Bits.bitsize + Bits.mask" begin
        @test bitsize(M64) == 256
        @test bitsize(M32) == 256
        m = bits_mask(M64, 130)
        @test count_ones(m) == 130
    end

    @testset "getinttype automatic routing (no opt-in needed)" begin
        @test getinttype(32)  === UInt64                   # unchanged
        @test getinttype(33)  === NTupleUInt{2, UInt64}     # was UInt66 (BitIntegers)
        @test getinttype(128) === NTupleUInt{4, UInt64}
        @test getinttype(256) === NTupleUInt{8, UInt64}     # partial-reward target
        # UInt32-word variant for consumer GPU register width
        @test getinttype(256; word=UInt32) === NTupleUInt{16, UInt32}
    end

    @testset "Pauli bit-ops match NTupleUInt at 256 qubits" begin
        nq = 256
        TM = getinttype(nq)                            # NTupleUInt{8, UInt64}
        TN = getchunkedinttype(nq; word=UInt64)        # NTupleUInt{4, UInt64}

        paulis_large = [isodd(i) ? :X : :Z for i in 1:nq]

        pM = ntupleuint_encode(TM, paulis_large)
        pN = ntuple_encode(TN, paulis_large)

        @test _countbitx(pM)      == _countbitx(pN)
        @test _countbitz(pM)      == _countbitz(pN)
        @test _countbitweight(pM) == _countbitweight(pN)

        pM_X = ntupleuint_encode(TM, fill(:X, nq))
        pM_Z = ntupleuint_encode(TM, fill(:Z, nq))
        pN_X = ntuple_encode(TN, fill(:X, nq))
        pN_Z = ntuple_encode(TN, fill(:Z, nq))
        @test _bitcommutes(pM_X, pM_Z) == _bitcommutes(pN_X, pN_Z)
    end

    @testset "getpauli / setpauli round-trip at $(nq) qubits" for nq in (50, 128, 256)
        T = getinttype(nq)
        @test T <: NTupleUInt
        pstr = zero(T)
        targets = [(1, 1), (17, 2), (33, 3), (nq, 2)]
        for (site, p) in targets
            pstr = setpauli(pstr, p, site)
        end
        for (site, p) in targets
            @test getpauli(pstr, site) == p
        end
        @test getpauli(pstr, 7) == 0
    end

    @testset "propagate() matches NTupleUInt baseline (nq=$nq)" for nq in (40, 64, 128)
        Random.seed!(2026 + nq)
        nl = 2
        topo   = bricklayertopology(nq; periodic=false)
        circ   = hardwareefficientcircuit(nq, nl; topology=topo)
        thetas = randn(length(circ))

        # NTupleUInt reference (existing tested path)
        TN = getchunkedinttype(nq; word=UInt32)
        vps_n = VectorPauliSum(Float64, nq, TN)
        push!(paulis(vps_n), symboltoint(TN, [:Z], [div(nq, 2)]))
        push!(coefficients(vps_n), 1.0)
        ref = overlapwithzero(propagate(circ, vps_n, thetas))

        # NTupleUInt path (automatic via getinttype, no opt-in)
        TM = getinttype(nq)
        vps_m = VectorPauliSum(Float64, nq, TM)
        push!(paulis(vps_m), symboltoint(TM, [:Z], [div(nq, 2)]))
        push!(coefficients(vps_m), 1.0)
        got = overlapwithzero(propagate(circ, vps_m, thetas))

        @test ref ≈ got atol=1e-10 rtol=1e-10
    end

    @testset "propagate() 256-qubit NTupleUInt completes" begin
        Random.seed!(31415)
        nq = 256
        TM = getinttype(nq)   # NTupleUInt{8, UInt64}
        @test isbitstype(TM)

        topo   = bricklayertopology(nq; periodic=false)
        circ   = hardwareefficientcircuit(nq, 1; topology=topo)
        thetas = randn(length(circ))

        vps = VectorPauliSum(Float64, nq, TM)
        push!(paulis(vps), symboltoint(TM, [:Z], [div(nq, 2)]))
        push!(coefficients(vps), 1.0)

        result = overlapwithzero(propagate(circ, vps, thetas))
        @test isfinite(result)
    end

end

println("\nAll NTupleUInt + NTupleUInt tests passed.")