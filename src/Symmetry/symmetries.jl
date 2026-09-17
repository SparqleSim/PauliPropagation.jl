### symmetries.jl
##
# This file contains functions to merge Pauli strings by symmetries.
# Currently it supports translational symmetry in 1d and 2d.
##
###


"""
    symmetrymerge(mapfunc, psum::AbstractPauliSum; thread=true)
    symmetrymerge(mapfunc, prop_cache::AbstractPauliPropagationCache; thread=true)

Merge equivalent Pauli strings under a symmetry mapping, returning a copy.
Each Pauli string is transformed using `mapfunc(pstr)` to its canonical representative,
and identical representatives are combined.
`mapfunc` must be constant on symmetry orbits, i.e. equivalent Pauli strings must map to the same integer.
On array-backed sums, `thread=false` turns off multithreading, the same as in `propagate`.

# Example
```julia
psum = PauliSum(6)
add!(psum, :Z, 3)
add!(psum, :Z, 6)
# merge under 1D translations, the same as `translationmerge(psum)`
symmetrymerge(pstr -> _translatetolowestinteger(pstr, nqubits(psum)), psum)
```
"""
symmetrymerge(mapfunc::F, thing::Union{AbstractPauliSum,AbstractPauliPropagationCache}; thread::Bool=true) where {F} =
    symmetrymerge!(mapfunc, deepcopy(thing); thread)

"""
    symmetrymerge!(mapfunc, psum::AbstractPauliSum; thread=true)
    symmetrymerge!(mapfunc, prop_cache::AbstractPauliPropagationCache; thread=true)

In-place version of [`symmetrymerge`](@ref).
Returns the merged `psum` or `prop_cache`, which is the same object that was passed in.
"""
function symmetrymerge!(mapfunc::F, psum::AbstractPauliSum; thread::Bool=true) where {F}
    prop_cache = PropagationCache(psum)
    symmetrymerge!(mapfunc, prop_cache; thread)
    return extractsum!(prop_cache, psum)
end

function symmetrymerge!(mapfunc::F, prop_cache::AbstractPauliPropagationCache; thread::Bool=true) where {F}
    mapterms!(mapfunc, prop_cache; thread)
    return merge!(prop_cache; thread)
end


"""
    translationmerge(psum::AbstractPauliSum; thread=true)

Shift and merge of a `psum` in a system with 1D translational symmetry.
```
psum = PauliSum(6)
add!(psum, :Z, 3)
add!(psum, :Z, 6)
translationmerge(psum)
>>> PauliSum(nqubits: 6, 1 Pauli term: 
 2.0 * ZIIIII
)
```
"""
translationmerge(psum::AbstractPauliSum; thread::Bool=true) = symmetrymerge(_translationmapper(psum), psum; thread)

"""
    translationmerge!(psum::AbstractPauliSum; thread=true)
    translationmerge!(prop_cache::AbstractPauliPropagationCache; thread=true)

In-place version of [`translationmerge`](@ref) for a periodic 1D chain.
"""
translationmerge!(thing; thread::Bool=true) = symmetrymerge!(_translationmapper(thing), thing; thread)

function _translationmapper(thing)
    nq = nqubits(thing)
    return pstr -> _translatetolowestinteger(pstr, nq)
end


"""
    translationmerge(psum::AbstractPauliSum, nx::Integer, ny::Integer; thread=true)

Shift and merge of a `psum` in a system with 2D translational symmetry.
```
psum = PauliSum(6)
add!(psum, :Z, 3)
add!(psum, :Z, 6)
translationmerge(psum, 2, 3)
```
"""
function translationmerge(psum::AbstractPauliSum, nx::Integer, ny::Integer; thread::Bool=true)
    return symmetrymerge(_translationmapper(psum, nx, ny), psum; thread)
end

"""
    translationmerge!(psum::AbstractPauliSum, nx::Integer, ny::Integer; thread=true)
    translationmerge!(prop_cache::AbstractPauliPropagationCache, nx::Integer, ny::Integer; thread=true)

In-place version of [`translationmerge`](@ref) for a periodic `nx` x `ny` grid.
"""
function translationmerge!(thing, nx::Integer, ny::Integer; thread::Bool=true)
    return symmetrymerge!(_translationmapper(thing, nx, ny), thing; thread)
end

function _translationmapper(thing, nx::Integer, ny::Integer)
    if nqubits(thing) != nx * ny
        throw(
            ArgumentError("Number of qubits $(nqubits(thing)) does not
                match grid size $(nx) x $(ny)"
            )
        )
    end

    # precompute masks once to accelerate shifting
    # main_mask: mask for all bits except the first column
    # wrap_mask: mask for the first column
    main_mask, wrap_mask = _computeshiftleftmasks(paulitype(thing), nx, ny)

    return pstr -> _translatetolowestinteger(pstr, nx, ny, main_mask, wrap_mask)
end

function _computeshiftleftmasks(::Type{TT}, nx::Integer, ny::Integer) where TT
    main_mask = zero(TT)  # main_mask: mask for all bits except the first column
    wrap_mask = zero(TT)  # wrap_mask: mask for the first column    

    for col in 1:nx
        for row in 1:ny
            site_index = (row - 1) * nx + col
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


# For 1d case, it is easier to shift right and set the first pauli to the last position
function _periodicshiftright(pstr::PauliStringType, nq)
    first_pauli = getpauli(pstr, 1)
    pstr = PauliPropagation._paulishiftright(pstr)
    pstr = setpauli(pstr, first_pauli, nq)
    return pstr
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

