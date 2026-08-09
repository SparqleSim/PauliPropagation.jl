### Circuits/topologies.jl
##
# A file containing functions to generate topologies for circuits.
# Topologies are represented as a list of tuples, where each tuple represents a connection between two qubits.
##
###

"""
    bricklayertopology(nqubits::Integer; periodic=false)

Create the topology of a so-called 1D bricklayer circuit on `nqubits` qubits. It consists of two sublayers connecting odd-even and eve-odd qubit indices, respectively.
If `periodic` is set to `true`, the last qubit is connected to the first qubit.
"""
function bricklayertopology(nqubits::Integer; periodic=false)
    return bricklayertopology(1:nqubits; periodic=periodic)
end

"""
    bricklayertopology(qindices; periodic=false)

Create the topology of a so-called 1D bricklayer circuit on a subset of qubits indicated by `qindices`.
If `periodic` is set to `true`, the last qubit is connected to the first qubit.
"""
function bricklayertopology(qindices; periodic=false)
    nqubits = length(qindices)

    topology = Tuple{Int,Int}[]
    if nqubits == 1
        return topology
    elseif nqubits == 2
        push!(topology, (qindices[1], qindices[2]))
        return topology
    else
        for ii in 1:2:nqubits-1
            push!(topology, (qindices[ii], qindices[ii+1]))
        end
        if periodic && (qindices[end] % 2 == 1)  # odd layer
            push!(topology, (qindices[end], qindices[1]))
        end
        for ii in 2:2:nqubits-1
            push!(topology, (qindices[ii], qindices[ii+1]))
        end
        if periodic && (qindices[end] % 2 == 0)  # even layer
            push!(topology, (qindices[end], qindices[1]))
        end

        return topology
    end
end

"""
    staircasetopology(nqubits::Integer; periodic=false)

Create a 1D staircase topology on `nqubits` qubits. The qubits are connected in a staircase pattern, where qubit `i` is connected to qubit `i+1`.
If `periodic` is set to `true`, the last qubit is connected to the first qubit.
"""
function staircasetopology(nqubits::Integer; periodic=false)
    topology = [(ii, ii + 1) for ii in 1:nqubits-1]
    if periodic && nqubits > 2
        push!(topology, (nqubits, 1))
    end
    return topology
end

"""
    rectangletopology(nx::Integer, ny::Integer; periodic=false)

Create a 2D topology on a grid of `nx` by `ny` qubits. The order is none in particular and may need to be adapted for specific purposes.
If `periodic` is set to `true`, the grid is connected periodically in both directions.
"""
function rectangletopology(nx::Integer, ny::Integer; periodic=false)
    topology = Tuple{Int,Int}[]

    for jj in 1:ny
        for ii in 1:nx

            if jj <= ny - 1
                push!(topology, ((jj - 1) * nx + ii, jj * nx + ii))
            end

            if ii + 1 <= nx
                push!(topology, ((jj - 1) * nx + ii, (jj - 1) * nx + ii + 1))
            end
        end
    end

    if periodic
        nq = nx * ny
        for ii in 1:nx
            push!(topology, (ii, nq - nx + ii))
        end


        for ii in 0:ny-1
            push!(topology, (ii * nx + 1, ii * nx + nx))
        end

        topology = [pair for pair in unique(topology) if pair[1] != pair[2]]
    end

    return topology

end


"""
    rectanglebricktopology(nx::Integer, ny::Integer; periodic_x=false, periodic_y=false)

Create a 2D brick-layer topology on a grid of `nx` by `ny` qubits, indexed column-major via `LinearIndices((nx, ny))`.
The topology is split into four sequential layers of non-overlapping two-qubit connections:
horizontal bonds starting at odd columns, horizontal bonds starting at even columns,
vertical bonds starting at odd rows, and vertical bonds starting at even rows.
If `periodic_x` (`periodic_y`) is set to `true`, the grid is connected periodically along the `x` (`y`) direction.
"""
function rectanglebricktopology(nx::Integer, ny::Integer; periodic_x::Bool=false, periodic_y::Bool=false)
    LI = LinearIndices((nx, ny))

    # Layer A: Horizontal edges starting at odd columns
    layer_A = [(LI[x, y], LI[x+1, y]) for x in 1:2:(nx-1) for y in 1:ny]
    if periodic_x && isodd(nx) && nx > 1
        append!(layer_A, [(LI[nx, y], LI[1, y]) for y in 1:ny])
    end

    # Layer B: Horizontal edges starting at even columns
    layer_B = [(LI[x, y], LI[x+1, y]) for x in 2:2:(nx-1) for y in 1:ny]
    if periodic_x && iseven(nx) && nx > 2
        append!(layer_B, [(LI[nx, y], LI[1, y]) for y in 1:ny])
    end

    # Layer C: Vertical edges starting at odd rows
    layer_C = [(LI[x, y], LI[x, y+1]) for x in 1:nx for y in 1:2:(ny-1)]
    if periodic_y && isodd(ny) && ny > 1
        append!(layer_C, [(LI[x, ny], LI[x, 1]) for x in 1:nx])
    end

    # Layer D: Vertical edges starting at even rows
    layer_D = [(LI[x, y], LI[x, y+1]) for x in 1:nx for y in 2:2:(ny-1)]
    if periodic_y && iseven(ny) && ny > 2
        append!(layer_D, [(LI[x, ny], LI[x, 1]) for x in 1:nx])
    end

    return vcat(layer_A, layer_B, layer_C, layer_D)
end

"""
    staircasetopology2d(nx::Integer, ny::Integer)

Create a 2D staircase topology on a grid of `nx` by `ny` qubits.
Mind the order of the topology, which forms a staircase spanning the grid -> in the Schrödinger picture <-. 
An observable acting on qubits index `nqubits` may interact non-trivially with every gate on the topology. 
Can topology can either be pathological or the most simple, depending on which index observables are non-identity.
"""
function staircasetopology2d(nx::Integer, ny::Integer)
    next_inds = [1]
    temp_inds = []

    topology = Tuple{Int,Int}[]
    while length(next_inds) > 0
        for ind in next_inds
            if ind % nx != 0
                next_ind = ind + 1
                push!(topology, (ind, next_ind))
                push!(temp_inds, next_ind)
            end
            if ceil(Int, ind / nx) < ny
                next_ind = ind + nx
                push!(topology, (ind, next_ind))
                push!(temp_inds, next_ind)
            end
        end
        next_inds = temp_inds
        temp_inds = []

    end
    return unique(topology)
end


"""
    ibmeagletopology

Topology of the IBM Eagle device with 127 qubits.
Also called the heave-hex topology on 127 qubits.
"""
const ibmeagletopology = [
    (1, 2), (1, 15), (2, 3), (3, 4), (4, 5), (5, 6), (5, 16), (6, 7), (7, 8), (8, 9), (9, 10), (9, 17),
    (10, 11), (11, 12), (12, 13), (13, 14), (13, 18), (15, 19), (16, 23), (17, 27), (18, 31), (19, 20),
    (20, 21), (21, 22), (21, 34), (22, 23), (23, 24), (24, 25), (25, 26), (25, 35), (26, 27), (27, 28),
    (28, 29), (29, 30), (29, 36), (30, 31), (31, 32), (32, 33), (33, 37), (34, 40), (35, 44), (36, 48),
    (37, 52), (38, 39), (38, 53), (39, 40), (40, 41), (41, 42), (42, 43), (42, 54), (43, 44), (44, 45),
    (45, 46), (46, 47), (46, 55), (47, 48), (48, 49), (49, 50), (50, 51), (50, 56), (51, 52), (53, 57),
    (54, 61), (55, 65), (56, 69), (57, 58), (58, 59), (59, 60), (59, 72), (60, 61), (61, 62), (62, 63),
    (63, 64), (63, 73), (64, 65), (65, 66), (66, 67), (67, 68), (67, 74), (68, 69), (69, 70), (70, 71),
    (71, 75), (72, 78), (73, 82), (74, 86), (75, 90), (76, 77), (76, 91), (77, 78), (78, 79), (79, 80),
    (80, 81), (80, 92), (81, 82), (82, 83), (83, 84), (84, 85), (84, 93), (85, 86), (86, 87), (87, 88),
    (88, 89), (88, 94), (89, 90), (91, 95), (92, 99), (93, 103), (94, 107), (95, 96), (96, 97), (97, 98),
    (97, 110), (98, 99), (99, 100), (100, 101), (101, 102), (101, 111), (102, 103), (103, 104), (104, 105),
    (105, 106), (105, 112), (106, 107), (107, 108), (108, 109), (109, 113), (110, 115), (111, 119), (112, 123),
    (113, 127), (114, 115), (115, 116), (116, 117), (117, 118), (118, 119), (119, 120), (120, 121), (121, 122),
    (122, 123), (123, 124), (124, 125), (125, 126), (126, 127)
]
