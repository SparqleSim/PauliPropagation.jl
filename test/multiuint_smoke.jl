using Test
using Bits     # mask + bitsize used by the module under test
using PauliPropagation: MultiUInt

@testset "MultiUInt smoke" begin
    M = MultiUInt{4, UInt64}
    z = zero(M); o = one(M)

    @testset "construction + identity" begin
        @test z.parts == (0x0,0x0,0x0,0x0)
        @test o.parts == (0x1,0x0,0x0,0x0)
        @test M(0xABCD) == M((0xABCD, 0x0, 0x0, 0x0))
        @test iszero(z) && !iszero(o)
    end

    @testset "pointwise bitops" begin
        a = M((0xFF00, 0x0F0F, 0x0, 0x0))
        b = M((0xF0F0, 0xF0F0, 0x0, 0x0))
        @test (a & b) == M((0xF000, 0x0000, 0x0, 0x0))
        @test (a | b) == M((0xFFF0, 0xFFFF, 0x0, 0x0))
        @test (a ⊻ b) == M((0x0FF0, 0xFFFF, 0x0, 0x0))
        @test (~z).parts == ntuple(_ -> typemax(UInt64), 4)         # all-ones
    end

    @testset "shifts across word boundaries" begin
        a = M(UInt64(1))
        @test (a << 63).parts == (0x8000_0000_0000_0000, 0x0, 0x0, 0x0)
        @test (a << 64).parts == (0x0, 0x1, 0x0, 0x0)
        @test (a << 65).parts == (0x0, 0x2, 0x0, 0x0)
        @test (a << 191).parts == (0x0, 0x0, 0x0, 0x8000_0000_0000_0000) ||
              (a << 191).parts == (0x0, 0x0, 0x8000_0000_0000_0000, 0x0)
        @test (a << 255).parts == (0x0, 0x0, 0x0, 0x8000_0000_0000_0000)  # bit 255 = top
        @test (a << 256) == z                                              # past the end

        b = M((0x0, 0x0, 0x0, 0x1))                                        # bit 192
        @test (b >> 192) == one(M)                                         # → bit 0
        @test (b >> 193) == z                                              # → past low
        @test (b >> 64).parts == (0x0, 0x0, 0x1, 0x0)
        @test (b >> 128).parts == (0x0, 0x1, 0x0, 0x0)
    end

    @testset "shift round-trip" begin
        x = M((0xDEAD_BEEF, 0xCAFE_BABE, 0x12345678, 0xABCD))
        @test (x << 64) >> 64 == M((0xDEAD_BEEF, 0xCAFE_BABE, 0x12345678, 0x0))
        @test (x << 1) >> 1 == M((x.parts[1] & ~UInt64(0), x.parts[2], x.parts[3], x.parts[4] & ((UInt64(1) << 63) - 1)))
    end

    @testset "count_ones + ordering" begin
        @test count_ones(z) == 0
        @test count_ones(~z) == 256
        @test count_ones(o << 200) == 1
        @test isless(o, o << 1)
        @test isless(o, M((0x0, 0x1, 0x0, 0x0)))         # high word wins
        @test !isless(M((0x0, 0x1, 0x0, 0x0)), o)
    end

    @testset "hash + equality" begin
        @test M(0xABCD) == M(0xABCD)
        @test hash(M(0xABCD)) == hash(M(0xABCD))
        @test hash(M(0xABCD)) != hash(M(0xABCE))
    end

    @testset "Bits.bitsize + mask" begin
        @test Bits.bitsize(M) == 256
        @test Bits.bitsize(z) == 256
        m = Bits.mask(M, 130)
        @test count_ones(m) == 130
        @test (m & (one(M) << 129)) != zero(M)            # bit 129 in
        @test (m & (one(M) << 130)) == zero(M)            # bit 130 out
    end

    @testset "isbits — required for GPU" begin
        @test isbits(z)
        @test isbits(o)
    end

    println("MultiUInt smoke: OK")
end
