### symmetry_utils.jl
##
# Low-level machinery behind the symmetry merging functions in `symmetries.jl`.
# Nothing in this file is exported. It contains:
#   - grid helpers (site <-> coordinate, grid size validation)
#   - periodic bit-shifts for translational symmetry in 1D and 2D
#   - site permutations, used for reflection symmetry
#   - the canonical form under all permutations of the sites
##
###


## Grid helpers

# Sites on an (nx, ny) grid are numbered row by row, so that site (x, y) has
# integer index (y - 1) * nx + x. This is the same convention as
# `rectangletopology`, so symmetry merging composes with those circuits.
_coordtoindex(x::Integer, y::Integer, nx::Integer) = (y - 1) * nx + x

function _indextocoord(ind::Integer, nx::Integer)
    y, x = divrem(ind - 1, nx)
    return x + 1, y + 1
end

# Every 2D symmetry needs the sum to actually live on the grid.
function _checkgridsize(psum, nx::Integer, ny::Integer)
    if nqubits(psum) != nx * ny
        throw(ArgumentError(
            "Number of qubits $(nqubits(psum)) does not match grid size $(nx) x $(ny)."
        ))
    end
    return nothing
end


## Translational symmetry, 1D

# a function for 1D symmetric merging that does not check for existing terms
# and instead shifts through to find the lowest integer representation
# that is the representative that we merge to
function _translatetolowestinteger(pstr::PauliStringType, nq)
    if pstr == 0
        return pstr
    end

    lowest_pstr = pstr
    for _ in 1:nq
        # shift periodically by one
        pstr = _periodicshiftright(pstr, nq)

        # if the shifted Pauli is lower, record lowest int
        lowest_pstr = min(lowest_pstr, pstr)
    end

    return lowest_pstr
end

# For 1d case, it is easier to shift right and set the first pauli to the last position
function _periodicshiftright(pstr::PauliStringType, nq)
    first_pauli = getpauli(pstr, 1)
    pstr = _paulishiftright(pstr)
    pstr = setpauli(pstr, first_pauli, nq)
    return pstr
end


## Translational symmetry, 2D

# the same strategy for the 2D case
function _translatetolowestinteger(pstr::PauliStringType, nx, ny, main_mask, wrap_mask)
    if pstr == 0
        return pstr
    end

    lowest_pstr = pstr
    for _ in 1:ny
        for _ in 1:nx
            # shift periodically by one column
            pstr = _periodicshiftleft(pstr, nx, main_mask, wrap_mask)

            # if the shifted Pauli is lower, record lowest int
            lowest_pstr = min(lowest_pstr, pstr)
        end

        pstr = _periodicshiftup(pstr, nx, ny) # shift periodically by one row

    end

    return lowest_pstr
end

# Precomputes the two masks used by `_periodicshiftleft`.
# main_mask: mask for all bits except the first column
# wrap_mask: mask for the first column
function _computeshiftleftmasks(::Type{TT}, nx::Integer, ny::Integer) where TT
    main_mask = zero(TT)
    wrap_mask = zero(TT)

    for col in 1:nx
        for row in 1:ny
            site_index = _coordtoindex(col, row, nx)
            bit_index = 2 * (site_index - 1)

            if col == 1
                # first column -> wrap mask
                wrap_mask |= (TT(3) << bit_index)
            else
                main_mask |= (TT(3) << bit_index)
            end
        end
    end

    return main_mask, wrap_mask
end

# Shifts a `pstr` left one column in a (`nx`, _ ) 2D grid of `nq` qubits.
# This function shifts the entire bitstring left one column, 
# and sets the first column of Paulis to the last column of the Paulis.
function _periodicshiftleft(pstr::PauliStringType, nx, main_mask, wrap_mask)
    # main_mask: mask for all bits except the first column
    # wrap_mask: mask for the first column

    shift_size = 2 * nx - 2
    first_col_paulis = pstr & wrap_mask
    main_shift = (pstr & main_mask) >> 2
    pstr = main_shift | (first_col_paulis << shift_size)

    return pstr
end

# Shifts a `pstr` up one row in a (`nx`, `ny`) 2D grid of `nq` qubits on a 
# cylindrical lattice.
# This function shifts the entire bitstring up one row, 
# and sets the first row of Paulis to the last row of the Paulis.
function _periodicshiftup(pstr::PauliStringType, nx, ny)

    n_bits = nx * ny * 2
    shift_size = 2 * nx
    first_row_paulis = pstr & _pauliwindowmask(typeof(pstr), 1, nx)
    pstr = pstr >> shift_size
    pstr = pstr | (first_row_paulis << (n_bits - shift_size))

    return pstr
end


## Site permutations

# A permutation `perm` of the sites is applied to a Pauli string as a gather:
# the Pauli on site `perm[i]` is moved to site `i`. This is the same operation
# as `getpauli(pstr, perm)`, but with the bit offsets `2 * (perm[i] - 1)`
# precomputed and the loop unrolled over an `NTuple`, which is roughly twice
# as fast for large strings.
@inline function _permutesites(pstr::TT, src_shifts::NTuple{N,Int}) where {TT,N}
    permuted = zero(TT)
    for ii in 1:N
        pauli = (pstr >> src_shifts[ii]) & TT(3)
        permuted |= pauli << (2 * (ii - 1))
    end
    return permuted
end

_permutationtoshifts(perm) = Tuple(2 * (site - 1) for site in perm)

# Returns a function mapping a Pauli string to the lowest integer among itself
# and its images under each permutation in `perms`. For this to be a valid
# canonical form for `symmetrymerge`, `perms` together with the identity must be
# closed under composition, e.g. both mirrors of a rectangle plus their product.
function _lowestpermutationmapper(perms)
    all_src_shifts = Tuple(_permutationtoshifts(perm) for perm in perms)
    return pstr -> _lowestpermuted(pstr, pstr, all_src_shifts)
end

# Recursion on the tuple of permutations rather than a `for` loop, so that
# the compiler fully unrolls it (measurably faster for large Pauli strings).
@inline _lowestpermuted(pstr, lowest_pstr, ::Tuple{}) = lowest_pstr

@inline function _lowestpermuted(pstr, lowest_pstr, all_src_shifts::Tuple)
    lowest_pstr = min(lowest_pstr, _permutesites(pstr, first(all_src_shifts)))
    return _lowestpermuted(pstr, lowest_pstr, Base.tail(all_src_shifts))
end


## Reflection permutations

# Reflection of a 1D chain of `nq` sites: site i <-> site nq - i + 1.
_chainreflection(nq::Integer) = collect(nq:-1:1)

# Reflection of an (nx, ny) grid as a site permutation. `flip_x` reflects the
# x coordinate (x -> nx - x + 1), `flip_y` the y coordinate; both together
# give the rotation by 180 degrees.
function _gridreflection(nx::Integer, ny::Integer; flip_x::Bool, flip_y::Bool)
    perm = Vector{Int}(undef, nx * ny)
    for ind in eachindex(perm)
        x, y = _indextocoord(ind, nx)
        src_x = flip_x ? nx - x + 1 : x
        src_y = flip_y ? ny - y + 1 : y
        perm[ind] = _coordtoindex(src_x, src_y, nx)
    end
    return perm
end

# The non-identity elements of the reflection group selected by `axes`:
# {Rx}, {Ry}, or {Rx, Ry, Rx*Ry} for both. Together with the identity each set
# is closed under composition, as `_lowestpermutationmapper` requires.
function _gridreflections(axes, nx::Integer, ny::Integer)
    axes = axes isa Symbol ? (axes,) : Tuple(axes)
    if isempty(axes) || any(axis -> axis ∉ (:x, :y), axes)
        throw(ArgumentError("Reflection axes must be :x, :y or both. Got $(axes)."))
    end

    flip_x = :x in axes
    flip_y = :y in axes
    perms = Vector{Int}[]
    flip_x && push!(perms, _gridreflection(nx, ny; flip_x=true, flip_y=false))
    flip_y && push!(perms, _gridreflection(nx, ny; flip_x=false, flip_y=true))
    flip_x && flip_y && push!(perms, _gridreflection(nx, ny; flip_x=true, flip_y=true))
    return perms
end


## Full permutation symmetry

# Canonical representative of `pstr` under all permutations of the sites.
# The orbit of a Pauli string under the full symmetric group is fixed by how
# many X, Y and Z it contains, so the representative is the sorted string
# `X...X Y...Y Z...Z I...I`. It is assembled from three masked runs of
# `01` (X), `10` (Y) and `11` (Z) bit pairs, so the cost does not grow with
# the number of qubits.
function _permutationcanonicalform(pstr::TT) where {TT<:PauliStringType}
    if pstr == 0
        return pstr
    end

    num_x = countx(pstr)
    num_y = county(pstr)
    num_z = countz(pstr)

    x_mask = alternatingmask(pstr)  # ...0101 = XXX...
    y_mask = x_mask << 1            # ...1010 = YYY...

    xs = x_mask & _paulimask(TT, num_x)
    ys = (y_mask & _paulimask(TT, num_y)) << (2 * num_x)
    zs = _paulimask(TT, num_z) << (2 * (num_x + num_y))

    return xs | ys | zs
end


## Block-wise permutation symmetry (residual subsymmetry)

# Canonical representative under S_{B_1} x ... x S_{B_k}, where the blocks are contiguous
# site ranges (lo, hi) that partition 1:nq in order (empty blocks, hi < lo, are allowed).
# Each block is sorted independently as in the single-block form above.
function _permutationcanonicalform(pstr::TT, blocks) where {TT<:PauliStringType}
    canonical = zero(TT)
    for (lo, hi) in blocks
        hi < lo && continue
        block_paulis = _getpaulibits(pstr, lo, hi)          # Paulis of the block, shifted to sites 1..hi-lo+1
        canonical |= _permutationcanonicalform(block_paulis) << (2 * (lo - 1))
    end
    return canonical
end

# Blocks must be contiguous, ordered, non-overlapping and cover 1:nq exactly; otherwise
# sites would be dropped or double counted by the canonical form.
function _checkblocks(nq::Integer, blocks)
    next_site = 1
    for (lo, hi) in blocks
        lo == next_site || throw(ArgumentError(
            "Blocks must be contiguous and ordered: expected a block starting at site $(next_site), got ($(lo), $(hi))."))
        hi >= lo - 1 || throw(ArgumentError("Invalid block ($(lo), $(hi))."))
        next_site = hi + 1
    end
    next_site == nq + 1 || throw(ArgumentError(
        "Blocks $(blocks) do not cover all $(nq) qubits."))
    return nothing
end

"""
    residualpermutationblocks(i, j, nq)

Site blocks of the symmetry that survives inside a block of commuting all-to-all two-qubit
gates applied in lexicographic order of their qubit pairs. After the gate on `(i, j)` has been
applied, the gates still to come are invariant under 
`S_{[1, i-1]} x S_{i} x S_{[i+1, j]} x S_{[j+1, nq]}`, so the Pauli sum may be merged with
`permutationmerge!(psum, residualpermutationblocks(i, j, nq))` after every gate.
Returns `((1, i-1), (i, i), (i+1, j), (j+1, nq))`.
"""
function residualpermutationblocks(i::Integer, j::Integer, nq::Integer)
    1 <= i < j <= nq || throw(ArgumentError("Need 1 <= i < j <= nq, got i=$(i), j=$(j), nq=$(nq)."))
    return ((1, i - 1), (i, i), (i + 1, j), (j + 1, nq))
end
