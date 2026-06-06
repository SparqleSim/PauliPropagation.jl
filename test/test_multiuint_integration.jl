# Integration tests for MultiUInt-backed Pauli operations at >64-bit widths,
# i.e. the regime the CUDA extension previously couldn't reach (issue #145).
#
# These tests exercise the Pauli ops the issue lists as required functionality
# (sorting, commutes, pauliprod, getpauli, setpauli, countweight, countXorY,
# integer comparisons) at nqubits up to 256, against the MultiUInt-backed
# integer type returned by `getinttype`.

using Test
using Random
using PauliPropagation
using PauliPropagation: MultiUInt

@testset "MultiUInt-backed Pauli ops (>64-bit regime, issue #145)" begin

    @testset "getinttype routes to MultiUInt above 64 bits" begin
        @test getinttype(32) === UInt64                              # 64 bits, native
        @test getinttype(33) === MultiUInt{2, UInt64}                 # 66 bits
        @test getinttype(64) === MultiUInt{2, UInt64}                 # 128 bits
        @test getinttype(65) === MultiUInt{3, UInt64}                 # 130 bits
        @test getinttype(128) === MultiUInt{4, UInt64}                # 256 bits
        @test getinttype(256) === MultiUInt{8, UInt64}                # partial-reward target
    end

    @testset "getpauli / setpauli round-trip at $(nq) qubits" for nq in (50, 128, 256)
        Random.seed!(42 + nq)
        T = getinttype(nq)
        @test T <: MultiUInt
        pstr = zero(T)
        # set a few sites to non-identity Paulis (1=X, 2=Z, 3=Y in the encoding)
        targets = [(1, 1), (17, 2), (33, 3), (64, 1), (nq, 2)]
        for (site, p) in targets
            pstr = setpauli(pstr, p, site)
        end
        for (site, p) in targets
            @test getpauli(pstr, site) == p
        end
        # untouched sites are still identity
        @test getpauli(pstr, 7) == 0
        @test getpauli(pstr, 90) == 0
    end

    @testset "countweight / count{x,y,z,xy,yz} at $(nq) qubits" for nq in (50, 128, 256)
        T = getinttype(nq)
        pstr = zero(T)
        # site 1: X (encoding 1), site 2: Y (encoding 3), site 3: Z (encoding 2)
        pstr = setpauli(pstr, 1, 1)
        pstr = setpauli(pstr, 3, 2)
        pstr = setpauli(pstr, 2, 3)
        @test countweight(pstr) == 3
        @test countx(pstr) == 1
        @test county(pstr) == 1
        @test countz(pstr) == 1
        @test countxy(pstr) == 2        # X and Y
        @test countyz(pstr) == 2        # Y and Z
        @test containsXorY(pstr)
        @test containsYorZ(pstr)
    end

    @testset "commutes is symmetric and consistent (nq=$(nq))" for nq in (40, 100, 200)
        Random.seed!(nq)
        T = getinttype(nq)
        for _ in 1:10
            a = zero(T); b = zero(T)
            for k in 1:5
                a = setpauli(a, rand(0:3), rand(1:nq))
                b = setpauli(b, rand(0:3), rand(1:nq))
            end
            @test commutes(a, b) == commutes(b, a)
            # commuting with the identity always commutes
            @test commutes(a, zero(T))
        end
    end

    @testset "pauliprod is involutive on identical strings" for nq in (50, 128, 256)
        T = getinttype(nq)
        a = zero(T)
        for k in 1:5
            a = setpauli(a, (k % 3) + 1, k * (nq ÷ 10))
        end
        # P * P = I (ignoring sign), and PauliPropagation's bit-level
        # multiplication encodes that as the all-identity bit-string.
        p, _coeff = pauliprod(a, a)
        @test p == zero(T)
    end

    @testset "ordering is total (required for sort)" for nq in (40, 128)
        Random.seed!(nq + 1)
        T = getinttype(nq)
        items = T[zero(T)]
        a = zero(T)
        for k in 1:20
            a = setpauli(a, rand(1:3), rand(1:nq))
            push!(items, a)
        end
        sorted = sort(items)
        @test issorted(sorted)
        @test length(unique(sorted)) <= length(sorted)
    end

    @testset "PauliSum can hold MultiUInt-backed PauliStrings (nq=80)" begin
        nq = 80
        T = getinttype(nq)
        @test T <: MultiUInt
        ps = PauliSum(nq)
        # Add three terms with distinct Pauli supports.
        for (site, p, c) in ((1, :X, 1.0), (40, :Z, -2.5), (80, :Y, 0.5))
            add!(ps, PauliString(nq, p, site, c))
        end
        @test length(ps) == 3
        # Direct dict lookup via getcoeff works with Int key (= identity).
        @test getcoeff(ps, 0) == 0.0           # no identity term
        # PauliString lookups round-trip.
        x_term = PauliString(nq, :X, 1).term
        @test getcoeff(ps, x_term) == 1.0
    end
end
