using Test
using PauliPropagation

@testset "GPULongInt" begin
    a = GPULongInt{2}()
    @test a.blocks == (0x0, 0x0)
    b = GPULongInt{2}((0x1, 0x0))
    @test b.blocks == (0x1, 0x0)
    x = GPULongInt{2}((0b1010, 0x0))
    y = GPULongInt{2}((0b1100, 0x0))
    @test (x & y).blocks[1] == 0b1000
    @test (x | y).blocks[1] == 0b1110
    @test xor(x, y).blocks[1] == 0b0110
    a = set_bit(a, 1, 1)
    a = set_bit(a, 65, 1)
    @test get_bit(a, 1) == 1
    @test get_bit(a, 65) == 1
    @test get_bit(a, 2) == 0
    x1 = GPULongInt{2}((0b01, 0x0))
    z1 = GPULongInt{2}((0x0, 0x0))
    x2 = GPULongInt{2}((0x0, 0x0))
    z2 = GPULongInt{2}((0b01, 0x0))
    @test !commutes(x1, z1, x2, z2)
    nx, nz = pauliprod(x1, z1, x2, z2)
    @test nx.blocks[1] == 0b01
    @test nz.blocks[1] == 0b01
end
