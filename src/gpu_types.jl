struct GPULongInt{N}
    blocks::NTuple{N, UInt64}
end

# Empty constructor
GPULongInt{N}() where N =
    GPULongInt{N}(ntuple(i -> zero(UInt64), N))

# Overloading logical operators
import Base: &, |, xor, ~

Base.:~(a::GPULongInt{N}) where N =
    GPULongInt{N}(map(~, a.blocks))

Base.:&(a::GPULongInt{N}, 
         b::GPULongInt{N}) where N =
    GPULongInt{N}(map(&, a.blocks, b.blocks))

Base.:|(a::GPULongInt{N}, 
         b::GPULongInt{N}) where N =
    GPULongInt{N}(map(|, a.blocks, b.blocks))

Base.:xor(a::GPULongInt{N}, 
          b::GPULongInt{N}) where N =
    GPULongInt{N}(map(xor, a.blocks, b.blocks))

# Get bit value for a given qubit
function get_bit(a::GPULongInt{N}, 
                 bit_idx::Int) where N
    block_idx = div(bit_idx - 1, 64) + 1
    local_idx = mod(bit_idx - 1, 64)
    if block_idx > N || block_idx < 1
        return zero(UInt64)
    end
    return (a.blocks[block_idx] >> local_idx) & 
           one(UInt64)
end

# Set bit value in a new tuple
function set_bit(a::GPULongInt{N}, 
                 bit_idx::Int, val::Int) where N
    block_idx = div(bit_idx - 1, 64) + 1
    local_idx = mod(bit_idx - 1, 64)

    new_blocks = ntuple(i -> begin
        if i == block_idx
            old = a.blocks[i]
            if val == 1
                old | (one(UInt64) << local_idx)
            else
                old & ~(one(UInt64) << local_idx)
            end
        else
            a.blocks[i]
        end
    end, N)

    return GPULongInt{N}(new_blocks)
end

# Check commutation on GPU
function commutes(x1::GPULongInt{N}, 
                  z1::GPULongInt{N},
                  x2::GPULongInt{N}, 
                  z2::GPULongInt{N}) where N
    match1 = map(&, x1.blocks, z2.blocks)
    match2 = map(&, z1.blocks, x2.blocks)
    conflicts = map(xor, match1, match2)
    total_ones = sum(map(count_ones, conflicts))
    return iseven(total_ones)
end

# Pauli multiplication on GPU
function pauliprod(x1::GPULongInt{N}, 
                   z1::GPULongInt{N},
                   x2::GPULongInt{N}, 
                   z2::GPULongInt{N}) where N
    new_x = map(xor, x1.blocks, x2.blocks)
    new_z = map(xor, z1.blocks, z2.blocks)
    return GPULongInt{N}(new_x), 
           GPULongInt{N}(new_z)
end
