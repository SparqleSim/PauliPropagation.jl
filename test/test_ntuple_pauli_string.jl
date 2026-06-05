using Test

let dir = joinpath(@__DIR__, "..")
    dir ∉ LOAD_PATH && push!(LOAD_PATH, dir)
end

using PauliPropagation
using PauliPropagation.GPUTypes

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

function ntuple_encode(::Type{NTuplePauliString{N,W}}, paulis::Vector{Symbol}) where {N,W}
    T = NTuplePauliString{N,W}
    pstr = zero(T)
    for (i, p) in enumerate(paulis)
        pstr = _setpaulibits(pstr, pauli_to_bits[p], i)
    end
    return pstr
end


@testset "NTuplePauliString" begin

    T32  = NTuplePauliString{2,  UInt32}
    T64  = NTuplePauliString{2,  UInt64}
    T128 = NTuplePauliString{8,  UInt32}
    T256 = NTuplePauliString{16, UInt32}

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

        x = NTuplePauliString{2,UInt32}(0xDEADBEEF)
        @test x.data[1] == 0xDEADBEEF
        @test x.data[2] == 0x00000000

        y = NTuplePauliString{2,UInt32}(0xDEADBEEF_CAFEBABE)
        @test y.data[1] == 0xCAFEBABE
        @test y.data[2] == 0xDEADBEEF
    end

    @testset "Bitwise operations" begin
        a = NTuplePauliString{2,UInt32}((0xAAAAAAAA, 0xAAAAAAAA))
        b = NTuplePauliString{2,UInt32}((0x55555555, 0x55555555))

        @test (a & b) == 0
        @test (a | b) == typemax(NTuplePauliString{2,UInt32})
        @test (a ⊻ b) == typemax(NTuplePauliString{2,UInt32})
        @test (~a)     == b
    end

    @testset "Shift operations" begin
        one64 = NTuplePauliString{2,UInt32}(1)
        @test (one64 << 3) == 8
        @test (one64 << 3 >> 3) == 1

        shifted = one64 << 32
        @test shifted.data[1] == 0x00000000
        @test shifted.data[2] == 0x00000001

        @test (shifted >> 32) == one64

        val = NTuplePauliString{2,UInt32}((0xFFFFFFFF, 0x00000000))
        shifted2 = val << 1
        @test shifted2.data[1] == 0xFFFFFFFE
        @test shifted2.data[2] == 0x00000001

        @test (one64 << 64) == 0
        @test (one64 >> 64) == 0
    end

    @testset "Comparison and sorting" begin
        a = NTuplePauliString{2,UInt32}(10)
        b = NTuplePauliString{2,UInt32}(20)
        c = NTuplePauliString{2,UInt32}(10)

        @test a < b
        @test b > a
        @test a <= c
        @test a >= c
        @test a == c
        @test a != b

        big   = NTuplePauliString{2,UInt32}((0x00000000, 0x00000001))
        small = NTuplePauliString{2,UInt32}((0xFFFFFFFF, 0x00000000))
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

        x = NTuplePauliString{2,UInt64}((0xAAAAAAAAAAAAAAAA, 0x0000000000000000))
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
        pstr  = ntuple_encode(NTuplePauliString{1,UInt32}, paulis7)
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
        pX = ntuple_encode(NTuplePauliString{1,UInt32}, [:X])
        pY = ntuple_encode(NTuplePauliString{1,UInt32}, [:Y])
        pZ = ntuple_encode(NTuplePauliString{1,UInt32}, [:Z])
        pI = ntuple_encode(NTuplePauliString{1,UInt32}, [:I])

        @test !_bitcommutes(pX, pY)
        @test !_bitcommutes(pY, pZ)
        @test !_bitcommutes(pX, pZ)
        @test  _bitcommutes(pX, pX)
        @test  _bitcommutes(pX, pI)

        pXX = ntuple_encode(NTuplePauliString{1,UInt32}, [:X, :X])
        pYY = ntuple_encode(NTuplePauliString{1,UInt32}, [:Y, :Y])
        @test _bitcommutes(pXX, pYY)

        pXZ = ntuple_encode(NTuplePauliString{1,UInt32}, [:X, :Z])
        pZX = ntuple_encode(NTuplePauliString{1,UInt32}, [:Z, :X])
        # XZ and ZX: qubit 1 (X,Z) anti-commutes, qubit 2 (Z,X) anti-commutes.
        # Two anti-commuting sites → even parity → the strings commute.
        @test _bitcommutes(pXZ, pZX)
    end

    @testset "Pauli product" begin
        pX = ntuple_encode(NTuplePauliString{1,UInt32}, [:X])
        pY = ntuple_encode(NTuplePauliString{1,UInt32}, [:Y])
        pZ = ntuple_encode(NTuplePauliString{1,UInt32}, [:Z])
        pI = ntuple_encode(NTuplePauliString{1,UInt32}, [:I])

        @test _bitpaulimultiply(pX, pX) == pI
        @test _bitpaulimultiply(pY, pY) == pI
        @test _bitpaulimultiply(pZ, pZ) == pI
        @test _bitpaulimultiply(pX, pY) == pZ
    end

    @testset "getpauli / setpauli round-trip" begin
        T    = NTuplePauliString{4,UInt32}
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
        ntp        = ntuple_encode(NTuplePauliString{2,UInt32}, paulis_ref)

        @test _countbitweight(ntp) == _countbitweight(ref)
        @test _countbitxy(ntp)     == _countbitxy(ref)
        @test _countbityz(ntp)     == _countbityz(ref)
        @test _countbitx(ntp)      == _countbitx(ref)
        @test _countbity(ntp)      == _countbity(ref)
        @test _countbitz(ntp)      == _countbitz(ref)
    end

    @testset "256-qubit smoke test" begin
        T  = NTuplePauliString{16,UInt32}
        nq = 256
        @test max_qubits(T) == 256

        paulis_large = [isodd(i) ? :X : :Z for i in 1:nq]
        pstr = ntuple_encode(T, paulis_large)

        @test _countbitx(pstr)      == 128
        @test _countbitz(pstr)      == 128
        @test _countbity(pstr)      == 0
        @test _countbitweight(pstr) == 256

        # 256 qubit pairs each anticommuting => even count => overall commutation
        pX_all = ntuple_encode(T, fill(:X, nq))
        pZ_all = ntuple_encode(T, fill(:Z, nq))
        @test _bitcommutes(pX_all, pZ_all)
    end

    @testset "getntupleinttype factory" begin
        @test getntupleinttype(16;  word=UInt32) == NTuplePauliString{1,  UInt32}
        @test getntupleinttype(64;  word=UInt32) == NTuplePauliString{4,  UInt32}
        @test getntupleinttype(256; word=UInt32) == NTuplePauliString{16, UInt32}
        @test getntupleinttype(128; word=UInt64) == NTuplePauliString{4,  UInt64}
    end

    @testset "getinttype gpu_compatible flag" begin
        @test getinttype(64;  gpu_compatible=true, gpu_word=UInt32) == NTuplePauliString{4,  UInt32}
        @test getinttype(256; gpu_compatible=true, gpu_word=UInt32) == NTuplePauliString{16, UInt32}
    end

end

println("\nAll NTuplePauliString tests passed.")
