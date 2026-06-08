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
        # 0xAAAAAAAA = ...1010 1010 1010 (every other bit set, starting from bit 1)
        # 0x55555555 = ...0101 0101 0101 (every other bit set, starting from bit 0)
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
        # Build the NTupleUInt reference using the production symboltoint() encoder
        # so that any mistake in the encoder will surface here, not be papered over.
        paulis7 = [:X, :Y, :Z, :I, :X, :Y, :Z]
        T1 = NTupleUInt{1,UInt32}
        pstr  = symboltoint(T1, paulis7, collect(1:length(paulis7)))
        ref32 = symboltoint(UInt32, paulis7, collect(1:length(paulis7)))

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

    @testset "Pauli product" begin
        T1 = NTupleUInt{1,UInt32}
        pX = symboltoint(T1, [:X], [1])
        pY = symboltoint(T1, [:Y], [1])
        pZ = symboltoint(T1, [:Z], [1])
        pI = symboltoint(T1, [:I], [1])

        @test _bitpaulimultiply(pX, pX) == pI
        @test _bitpaulimultiply(pY, pY) == pI
        @test _bitpaulimultiply(pZ, pZ) == pI
        @test _bitpaulimultiply(pX, pY) == pZ
    end

    @testset "Commutation, products via PauliString, PauliSum, VectorPauliSum" begin
        # Drive commutation and Pauli products through the high-level
        # PauliString / PauliSum / VectorPauliSum API to show the new type
        # is compatible end-to-end.
        T1 = NTupleUInt{1,UInt32}

        pX = symboltoint(T1, [:X], [1])
        pY = symboltoint(T1, [:Y], [1])
        pZ = symboltoint(T1, [:Z], [1])

        @test !commutes(pX, pY)
        @test !commutes(pY, pZ)
        @test !commutes(pX, pZ)
        @test  commutes(pX, pX)
        @test  commutes(pX, symboltoint(T1, [:I], [1]))

        # Pauli product via PauliString.
        # pauliprod tracks phase, so X*X = (1+0im)*I and X*Y = (0+1im)*Z.
        # We only check the Pauli string (term), not the coefficient.
        ps_X = PauliString(1, :X, 1)
        ps_Y = PauliString(1, :Y, 1)
        @test pauliprod(ps_X, ps_X).term == symboltoint(getinttype(1), [:I], [1])
        @test pauliprod(ps_X, ps_Y).term == symboltoint(getinttype(1), [:Z], [1])

        # Via PauliSum
        sum_xy = PauliSum(ps_X) + PauliSum(ps_Y)
        @test length(topaulistrings(sum_xy)) == 2
        @test !commutes(sum_xy, PauliSum(ps_X))

        # Via VectorPauliSum.
        # X and Z on the *same* qubit anticommute; on different qubits they commute.
        nq = 4
        TT = getinttype(nq)
        vps  = VectorPauliSum(nq, TT[], Float64[])
        push!(paulis(vps),       symboltoint(TT, [:X], [1]))
        push!(coefficients(vps), 1.0)
        push!(paulis(vps),       symboltoint(TT, [:Z], [1]))   # same qubit 1
        push!(coefficients(vps), 2.0)
        @test countweight(paulis(vps)[1]) == 1
        @test countweight(paulis(vps)[2]) == 1
        @test !commutes(paulis(vps)[1], paulis(vps)[2])
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
        ref        = symboltoint(UInt64, paulis_ref, collect(1:length(paulis_ref)))
        ntp        = symboltoint(NTupleUInt{2,UInt32}, paulis_ref, collect(1:length(paulis_ref)))

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
        pstr = symboltoint(T, paulis_large, collect(1:nq))

        @test _countbitx(pstr)      == 128
        @test _countbitz(pstr)      == 128
        @test _countbity(pstr)      == 0
        @test _countbitweight(pstr) == 256

        pX_all = symboltoint(T, fill(:X, nq), collect(1:nq))
        pZ_all = symboltoint(T, fill(:Z, nq), collect(1:nq))
        @test _bitcommutes(pX_all, pZ_all)
    end

    @testset "getchunkedinttype factory" begin
        @test getchunkedinttype(16;  word=UInt32) == NTupleUInt{1,  UInt32}
        @test getchunkedinttype(64;  word=UInt32) == NTupleUInt{4,  UInt32}
        @test getchunkedinttype(256; word=UInt32) == NTupleUInt{16, UInt32}
        @test getchunkedinttype(128; word=UInt64) == NTupleUInt{4,  UInt64}
    end

    @testset "propagate() matches BitIntegers baseline" begin
        Random.seed!(42)

        for nq in (32, 64, 128, 256)
            nl = 2
            topo  = bricklayertopology(nq; periodic=false)
            circ  = hardwareefficientcircuit(nq, nl; topology=topo)
            thetas = randn(length(circ))

            pstr_ref   = PauliString(nq, :Z, div(nq, 2))
            result_ref = overlapwithzero(propagate(circ, pstr_ref, thetas))

            TT = getchunkedinttype(nq; word=UInt32)
            vpsum = VectorPauliSum(nq, TT[], Float64[])
            push!(paulis(vpsum), symboltoint(TT, [:Z], [div(nq, 2)]))
            push!(coefficients(vpsum), 1.0)
            result_ntuple = overlapwithzero(propagate(circ, vpsum, thetas))

            @test result_ref ≈ result_ntuple atol=1e-10 rtol=1e-10
        end
    end

end