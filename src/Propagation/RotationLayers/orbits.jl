###
##
# The rotations of a sublayer on one orbit.
# The Pauli strings of an orbit are numbered by their coordinate, so that the rotation of bit i pairs up
# the entries whose numbers differ in bit i, and mixes the two coefficients of every pair.
# The truncations are applied after every rotation, as they are between the gates of a circuit.
##
###

# no Pauli string is heavier than this
const _UNLIMITED_WEIGHT = typemax(Int)

struct LayerTruncation{F}
    # the truncation of `buildtruncfunc` without the weight, which is known for every entry of an orbit
    truncfunc::F
    max_weight::Int
end

function LayerTruncation(truncfunc::F, max_weight::Real) where {F}
    if isinf(max_weight)
        return LayerTruncation{F}(truncfunc, _UNLIMITED_WEIGHT)
    else
        return LayerTruncation{F}(truncfunc, floor(Int, max_weight))
    end
end

_nevertruncate(pstr, coeff) = false

@inline _limitsweight(truncation::LayerTruncation) = truncation.max_weight != _UNLIMITED_WEIGHT

@inline function _istruncated(truncation::LayerTruncation, pstr, coeff)
    if _limitsweight(truncation) && countweight(pstr) > truncation.max_weight
        return true
    end
    return @inline truncation.truncfunc(pstr, coeff)
end

@inline function _istruncated(truncation::LayerTruncation, pstr, weight::Int, coeff)
    return weight > truncation.max_weight || @inline truncation.truncfunc(pstr, coeff)
end

# Without a limit on the weight, the weights are not counted.
@inline function _weightiflimited(truncation::LayerTruncation, pstr)
    if _limitsweight(truncation)
        return countweight(pstr)
    else
        return 0
    end
end

const _NO_WEIGHT_CHANGES = ntuple(_ -> Int8(0), 16)

@inline function _weightchangesiflimited(truncation::LayerTruncation, plan::SubLayerPlan)
    if _limitsweight(truncation)
        return plan.weight_changes
    else
        return _NO_WEIGHT_CHANGES
    end
end

# Applies the rotations of the sublayer to the orbits that `_grouporbits!` found, and writes their Pauli strings to `sink`.
# The orbits of every number of stages are transformed by a method of their own, in which that number is known to the compiler.
@eval function _transformorbits!(sink, task::TaskWorkspace, plan::SubLayerPlan, truncation, record_terms, record_coeffs, record_labels)
    Base.Cartesian.@nexprs $_MAX_BLOCK_STAGES n_stages -> begin
        if task.n_blocks[n_stages] > 0
            _transformblocks!(sink, task, plan, truncation, record_terms, Val(n_stages))
        end
    end

    for long_orbit in 1:task.n_long_orbits
        _transformlongorbit!(sink, task, plan, truncation, record_terms, record_coeffs, record_labels, task.long_orbit_records[long_orbit])
    end
    return sink
end

function _transformblocks!(sink, task::TaskWorkspace{TT,CT}, plan::SubLayerPlan{TT}, truncation::LayerTruncation,
    record_terms::Vector{TT}, ::Val{K}) where {TT,CT,K}

    n_blocks = task.n_blocks[K]
    block_records = task.block_records[K]
    block_coeffs = task.block_coeffs[K]
    block_present = task.block_present[K]
    _checkblocks(task, n_blocks, block_records, block_coeffs, block_present, K)

    if plan.overlapping
        for block in 1:n_blocks
            representative = record_terms[block_records[block]]
            _transformsharedblock!(sink, task, plan, truncation, representative, block_coeffs, block_present, (block - 1) << K, Val(K))
        end
    else
        for block in 1:n_blocks
            representative = record_terms[block_records[block]]
            _transformblock!(sink, task, plan, truncation, representative, block_coeffs, block_present, (block - 1) << K, Val(K))
        end
    end
    return sink
end

# The block of an orbit of a sublayer whose rotations share no qubits.
# A rotation finds the same Paulis in every pair, so its signs and its weight change are those of the representative.
@inline function _transformblock!(sink, task::TaskWorkspace{TT,CT}, plan::SubLayerPlan{TT}, truncation::LayerTruncation,
    representative::TT, block_coeffs::Vector{CT}, block_present::Vector{Bool}, block_start::Int, ::Val{K}) where {TT,CT,K}

    positions = task.positions
    local_paulis = task.local_paulis
    orbit_terms = task.orbit_terms
    orbit_weights = task.orbit_weights

    if _orbitrotations!(positions, plan, representative) != K
        _throwwrongblock()
    end
    n_entries = 1 << K

    # the rotation of every bit doubles the Pauli strings of the orbit
    weight_changes = _weightchangesiflimited(truncation, plan)
    orbit_terms[1] = representative
    orbit_weights[1] = _weightiflimited(truncation, representative)

    for bit in 0:K-1
        position = positions[bit+1]
        paulis = _localpaulis(plan, representative, position)
        local_paulis[bit+1] = paulis

        mask = plan.masks[position]
        weight_change = Int(weight_changes[paulis+1])
        n_lower = 1 << bit
        @inbounds for entry in 1:n_lower
            orbit_terms[n_lower+entry] = orbit_terms[entry] ⊻ mask
            orbit_weights[n_lower+entry] = orbit_weights[entry] + weight_change
        end
    end

    n_pairs = n_entries >> 1
    for stage in 1:K
        bit = _stagebit(positions, K, stage, plan.order)
        position = positions[bit+1]
        cos_val = plan.cosines[position]
        sin_val = plan.sines[position]

        # the sign of the Pauli string a rotation creates, from the lower and from the upper entry of a pair
        paulis = local_paulis[bit+1]
        sign_from_lower = plan.signs[paulis+1]
        sign_from_upper = plan.signs[(paulis⊻plan.local_mask)+1]

        n_lower = 1 << bit
        below_bit = n_lower - 1
        @inbounds for pair in 0:n_pairs-1
            # the number of the pair with a 0 put in at the bit
            lower = (((pair & ~below_bit) << 1) | (pair & below_bit)) + 1
            upper = lower + n_lower

            if block_present[block_start+lower] || block_present[block_start+upper]
                _mixpair!(block_coeffs, block_present, block_start, lower, upper, cos_val, sin_val, sign_from_lower, sign_from_upper,
                    orbit_terms, orbit_weights, truncation)
            end
        end
    end

    _emitblock!(sink, orbit_terms, block_coeffs, block_present, block_start, n_entries)
    return sink
end

# The block of an orbit of a sublayer whose rotations share qubits.
@inline function _transformsharedblock!(sink, task::TaskWorkspace{TT,CT}, plan::SubLayerPlan{TT}, truncation::LayerTruncation,
    representative::TT, block_coeffs::Vector{CT}, block_present::Vector{Bool}, block_start::Int, ::Val{K}) where {TT,CT,K}

    positions = task.positions
    shared_stages = task.shared_stages
    orbit_terms = task.orbit_terms
    orbit_weights = task.orbit_weights

    if _orbitrotations!(positions, plan, representative) != K
        _throwwrongblock()
    end
    n_entries = 1 << K

    limits_weight = _limitsweight(truncation)
    weight_changes = _weightchangesiflimited(truncation, plan)
    orbit_terms[1] = representative
    orbit_weights[1] = _weightiflimited(truncation, representative)

    for bit in 0:K-1
        position = positions[bit+1]
        shared_stage = _sharedstage(plan, positions, K, bit, representative, limits_weight)
        shared_stages[bit+1] = shared_stage

        mask = plan.masks[position]
        n_lower = 1 << bit
        @inbounds for entry in 1:n_lower
            orbit_terms[n_lower+entry] = orbit_terms[entry] ⊻ mask
            orbit_weights[n_lower+entry] = orbit_weights[entry] + shared_stage.weight_changes[_neighborsselected(shared_stage, entry - 1)]
        end

        if !shared_stage.is_tabulated && limits_weight
            for entry in 1:n_lower
                entry_paulis = _localpaulis(plan, orbit_terms[entry], position)
                orbit_weights[n_lower+entry] = orbit_weights[entry] + weight_changes[entry_paulis+1]
            end
        end
    end

    n_pairs = n_entries >> 1
    for stage in 1:K
        bit = _stagebit(positions, K, stage, plan.order)
        position = positions[bit+1]
        cos_val = plan.cosines[position]
        sin_val = plan.sines[position]
        shared_stage = shared_stages[bit+1]

        n_lower = 1 << bit
        below_bit = n_lower - 1
        @inbounds for pair in 0:n_pairs-1
            # the number of the pair with a 0 put in at the bit
            lower = (((pair & ~below_bit) << 1) | (pair & below_bit)) + 1
            upper = lower + n_lower

            if block_present[block_start+lower] || block_present[block_start+upper]
                if shared_stage.is_tabulated
                    selected = _neighborsselected(shared_stage, lower - 1)
                    sign_from_lower = shared_stage.signs_from_lower[selected]
                    sign_from_upper = shared_stage.signs_from_upper[selected]
                else
                    sign_from_lower, sign_from_upper = _pairsigns(plan, orbit_terms[lower], position)
                end
                _mixpair!(block_coeffs, block_present, block_start, lower, upper, cos_val, sin_val, sign_from_lower, sign_from_upper,
                    orbit_terms, orbit_weights, truncation)
            end
        end
    end

    _emitblock!(sink, orbit_terms, block_coeffs, block_present, block_start, n_entries)
    return sink
end

# The two coefficients of a pair after the rotation, and whether the truncations keep them.
Base.@propagate_inbounds function _mixpair!(block_coeffs::Vector{CT}, block_present, block_start::Int, lower::Int, upper::Int, cos_val, sin_val,
    sign_from_lower, sign_from_upper, orbit_terms, orbit_weights, truncation::LayerTruncation) where {CT}

    lower_coeff = block_coeffs[block_start+lower]
    upper_coeff = block_coeffs[block_start+upper]
    new_lower_coeff = mergefunc(lower_coeff * cos_val, upper_coeff * sin_val * sign_from_upper)
    new_upper_coeff = mergefunc(upper_coeff * cos_val, lower_coeff * sin_val * sign_from_lower)

    keep_lower = !_istruncated(truncation, orbit_terms[lower], orbit_weights[lower], new_lower_coeff)
    keep_upper = !_istruncated(truncation, orbit_terms[upper], orbit_weights[upper], new_upper_coeff)
    block_coeffs[block_start+lower] = ifelse(keep_lower, new_lower_coeff, zero(CT))
    block_coeffs[block_start+upper] = ifelse(keep_upper, new_upper_coeff, zero(CT))
    block_present[block_start+lower] = keep_lower
    block_present[block_start+upper] = keep_upper
    return
end

# a bit that no coordinate of a block has set
const _UNSET_BIT = 8 * sizeof(Int) - 2

"""
    _sharedstage(plan, positions, n_stages, bit, representative, limits_weight)

The signs and the weight change of the rotation of `bit` for each of the four ways in which the rotations on its two qubits are selected.
`positions` holds the rotations of the orbit in the order of the bits of a coordinate.
"""
@inline function _sharedstage(plan::SubLayerPlan, positions::Vector{Int32}, n_stages::Int, bit::Int, representative, limits_weight::Bool)
    position = positions[bit+1]
    paulis = _localpaulis(plan, representative, position)
    first_neighbor, second_neighbor = plan.neighbors[position]
    first_bit = _neighborbit(positions, n_stages, bit, first_neighbor)
    second_bit = _neighborbit(positions, n_stages, bit, second_neighbor)

    weight_changes = if limits_weight
        plan.shared_weight_changes[paulis+1]
    else
        (Int8(0), Int8(0), Int8(0), Int8(0))
    end
    return SharedStage((max(first_bit, 0), max(second_bit, 0)), plan.shared_signs_from_lower[paulis+1],
        plan.shared_signs_from_upper[paulis+1], weight_changes, first_bit >= 0 && second_bit >= 0)
end

# The bit of the coordinate that tells whether the rotation at `neighbor` is selected.
# No bit that is ever set if there is no such rotation in the orbit, and -1 if several rotations share the qubit.
# Along a chain, the rotations on the qubits of a rotation are the ones next to it.
@inline function _neighborbit(positions::Vector{Int32}, n_stages::Int, bit::Int, neighbor::Int32)
    if neighbor == 0
        return _UNSET_BIT
    elseif neighbor < 0
        return -1
    elseif bit > 0 && positions[bit] == neighbor
        return bit - 1
    elseif bit + 2 <= n_stages && positions[bit+2] == neighbor
        return bit + 1
    end

    for other_bit in 0:n_stages-1
        if positions[other_bit+1] == neighbor
            return other_bit
        end
    end
    return _UNSET_BIT
end

# which of the rotations on the two qubits are selected in the entry of the coordinate, as an index from 1 to 4
@inline function _neighborsselected(shared_stage::SharedStage, coordinate::Int)
    first_bit, second_bit = shared_stage.neighbor_bits
    return (((coordinate >> (first_bit & 63)) & 1) | (((coordinate >> (second_bit & 63)) & 1) << 1)) + 1
end

# the signs of the Pauli strings that the rotation at `position` creates from `lower_pstr` and from its product with the generator
@inline function _pairsigns(plan::SubLayerPlan, lower_pstr, position::Integer)
    paulis = _localpaulis(plan, lower_pstr, position)
    return plan.signs[paulis+1], plan.signs[(paulis⊻plan.local_mask)+1]
end

# The bit of the rotation that is applied as number `stage` of the `n_stages` rotations of an orbit.
function _stagebit(positions::Vector{Int32}, n_stages::Int, stage::Int, order::Int)
    if order == 1
        return stage - 1
    elseif order == -1
        return n_stages - stage
    end

    # the rotation with `stage - 1` rotations before it
    for bit in 0:n_stages-1
        n_before = 0
        for other_bit in 0:n_stages-1
            n_before += positions[other_bit+1] < positions[bit+1]
        end
        if n_before == stage - 1
            return bit
        end
    end
    return stage - 1
end

# the loops over the entries of a block index the arrays of the task without bounds checks
function _checkblocks(task::TaskWorkspace, n_blocks::Int, block_records, block_coeffs, block_present, n_stages::Int)
    n_entries = 1 << n_stages
    if !(1 <= n_stages <= _MAX_BLOCK_STAGES) || n_blocks > length(block_records) ||
       n_blocks * n_entries > min(length(block_coeffs), length(block_present)) ||
       n_entries > min(length(task.orbit_terms), length(task.orbit_weights)) ||
       n_stages > min(length(task.positions), length(task.local_paulis))
        throw(ArgumentError("the $n_blocks blocks of the orbits with $n_stages stages do not fit the workspace"))
    end
    return
end

@noinline _throwwrongblock() = throw(ArgumentError("the orbit of a block has another number of stages than the block"))
