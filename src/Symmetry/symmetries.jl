### symmetries.jl
##
# This file contains functions to merge Pauli strings by symmetries.
# Currently it supports translational symmetry in 1d and 2d.
##
###


"""
    symmetrymerge(psum::AbstractPauliSum, mapfunc::Function) -> AbstractPauliSum

Merge equivalent Pauli strings in `psum` under a symmetry mapping.
Each Pauli string is transformed using `mapfunc(pstr)` to its canonical
representative, and identical representatives are combined.

# Arguments
- `psum`: A `PauliSum` containing Pauli strings and coefficients.
- `mapfunc`: A function mapping each `PauliString` to its canonical representative.

# Returns
A new `PauliSum` where symmetric terms have been merged.

# Example
```julia
psum = PauliSum(6)
add!(psum, :Z, 3)
add!(psum, :Z, 6)
symmetrymerge(psum, pstr -> _translatetolowestinteger(pstr, psum.nqubits))
```
"""
function symmetrymerge(psum::PauliSum, mapfunc::F) where F<:Function
    merged_psum = PauliPropagation.similar(psum)
    # TODO: mage this work for mapfuc that also want to modify the coefficient
    for (pstr, coeff) in psum
        pstr = mapfunc(pstr)
        add!(merged_psum, pstr, coeff)
    end

    return merged_psum
end

function symmetrymerge(psum::VectorPauliSum, mapfunc::F) where F<:Function
    cache = symmetrymerge!(PropagationCache(psum), mapfunc)
    return VectorPauliSum(cache)
end

function symmetrymerge!(prop_cache::VectorPauliPropagationCache, mapfunc::F) where F<:Function
    AK.map!(pstr -> mapfunc(pstr), activeterms(prop_cache), activeterms(prop_cache))
    merge!(prop_cache)
    return prop_cache
end


"""
    translationmerge(psum::AbstractPauliSum)

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
translationmerge(psum::AbstractPauliSum) = symmetrymerge(
    psum, (pstr -> _translatetolowestinteger(pstr, nqubits(psum)))
)


"""
    translationmerge(psum::AbstractPauliSum, nx::Integer, ny::Integer)

Shift and merge of a `psum` in a system with 2D translational symmetry.
```
psum = PauliSum(6)
add!(psum, :Z, 3)
add!(psum, :Z, 6)
translationmerge(psum, 2, 3)
```
"""
function translationmerge(psum::AbstractPauliSum, nx::Integer, ny::Integer)
    if nqubits(psum) != nx * ny
        throw(
            ArgumentError("Number of qubits $(nqubits(psum)) does not 
                match grid size $(nx) x $(ny)"
            )
        )
    end

    # precompute masks once to accelerate shifting
    # main_mask: mask for all bits except the first column
    # wrap_mask: mask for the first column    
    main_mask, wrap_mask = _computeshiftleftmasks(paulitype(psum), nx, ny)

    mergefunc(pstr) = _translatetolowestinteger(
        pstr, nx, ny, main_mask, wrap_mask
    )

    return symmetrymerge(psum, mergefunc)
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

