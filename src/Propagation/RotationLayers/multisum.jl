###
##
# A sublayer applied to a multi sum.
# The Pauli strings of an orbit belong to many zones, so the records of an orbit are collected in one zone,
# which the hash of the representative picks. Every zone reads its Pauli strings twice: to count the records for every zone,
# and to write them there, each zone into a range of its own. It then applies the rotations to the orbits collected in it
# and parks what it makes in its outbox. At last the zones are emptied and take delivery of the outboxes.
##
###

function _propagatesinorbits(storage::PropagationBase.MultiSumStorage, prop_cache)
    return all(zonecache -> _propagatesinorbits(storage.zonestorage, zonecache), zonecaches(prop_cache))
end

# Returns `false`, with the sum of the cache as it was, if the sum holds an orbit of more stages than a label holds.
function _applysublayer!(::PropagationBase.MultiSumStorage, prop_cache::AbstractPauliPropagationCache, plan::SubLayerPlan{TT}, truncation,
    workspace::LayerWorkspace{TT,CT}; thread::Bool=true) where {TT,CT}

    PropagationBase._checkauxempty(prop_cache)
    n_terms = length(prop_cache)
    if n_terms == 0
        return true
    end

    zone_caches = zonecaches(prop_cache)
    zone_storage = PropagationBase.zonestorage(prop_cache)
    n_zones = nzones(prop_cache)
    tasks = _taskworkspaces!(workspace, n_zones, length(plan.masks))

    # the highest bits of the hash pick the zone, and the bits below them a partition within the zone
    zone_bits = trailing_zeros(n_zones)
    n_bits = zone_bits + _partitionbits(cld(n_terms, n_zones), sizeof(TT) + sizeof(CT) + sizeof(Int))
    n_partitions_per_zone = 1 << (n_bits - zone_bits)
    partition_counts = _partitioncounts!(workspace, 1 << n_bits, n_zones)

    is_counted = Vector{Bool}(undef, n_zones)
    function count_zone!(zone_id)
        is_counted[zone_id] = _countrecords!(view(partition_counts, :, zone_id), n_bits, tasks[zone_id].positions, plan,
            _zonesource(zone_storage, zone_caches[zone_id]))
    end
    PropagationBase._eachzone(count_zone!, prop_cache, thread)

    if !all(is_counted)
        return false
    end

    # the records of a zone are numbered from 1 on
    partition_starts = _partitionstarts!(workspace.partition_starts, partition_counts)
    zone_starts = [partition_starts[(zone_id-1)*n_partitions_per_zone+1] for zone_id in 1:n_zones+1]
    zone_terms, zone_coeffs, zone_labels = _zonerecords!(zone_storage, workspace, zone_caches, diff(zone_starts))

    function write_zone!(zone_id)
        _writezonerecords!(zone_terms, zone_coeffs, zone_labels, zone_starts, view(partition_counts, :, zone_id), n_bits, n_bits - zone_bits,
            tasks[zone_id].positions, plan, _zonesource(zone_storage, zone_caches[zone_id]))
    end
    PropagationBase._eachzone(write_zone!, prop_cache, thread)

    function transform_zone!(zone_id)
        sink = OutboxSink(outboxes(prop_cache)[zone_id])
        first_record = zone_starts[zone_id]
        for partition in (zone_id-1)*n_partitions_per_zone+1:zone_id*n_partitions_per_zone
            _transformpartition!(sink, tasks[zone_id], plan, truncation, zone_terms[zone_id], zone_coeffs[zone_id], zone_labels[zone_id],
                partition_starts[partition] - first_record + 1, partition_starts[partition+1] - first_record)
        end
    end
    PropagationBase._eachzone(transform_zone!, prop_cache, thread)

    # the records hold all of the sum, so the zones are emptied before they take what the sublayer made for them
    function deliver_to_zone!(zone_id)
        empty!(zone_caches[zone_id])
        PropagationBase._deliverto!(prop_cache, zone_id)
    end
    PropagationBase._eachzone(deliver_to_zone!, prop_cache, thread)

    PropagationBase._syncsums!(prop_cache)
    return true
end

# the pairs of Pauli string and coefficient of a zone
function _zonesource(::PropagationBase.ArrayStorage, zonecache)
    return zip(PropagationBase.activeterms(zonecache), PropagationBase.activecoeffs(zonecache))
end
_zonesource(::PropagationBase.StorageType, zonecache) = mainsum(zonecache)

# Arrays for the records that every zone collects, `n_records[zone_id]` of them.
# A zone of arrays has them in the auxiliary arrays of its cache, any other zone in the workspace.
function _zonerecords!(::PropagationBase.ArrayStorage, workspace::LayerWorkspace{TT,CT}, zone_caches, n_records::Vector{Int}) where {TT,CT}
    zone_terms = Vector{Vector{TT}}(undef, length(zone_caches))
    zone_coeffs = Vector{Vector{CT}}(undef, length(zone_caches))

    for (zone_id, zonecache) in enumerate(zone_caches)
        PropagationBase._ensurecapacity!(zonecache, n_records[zone_id])
        _, _, aux_terms, aux_coeffs = PropagationBase._mainauxarrays(zonecache)
        PropagationBase._checkfits(n_records[zone_id], aux_terms, aux_coeffs)
        zone_terms[zone_id] = aux_terms
        zone_coeffs[zone_id] = aux_coeffs
    end

    return zone_terms, zone_coeffs, _zonelabels!(workspace, n_records)
end

function _zonerecords!(::PropagationBase.StorageType, workspace::LayerWorkspace{TT,CT}, zone_caches, n_records::Vector{Int}) where {TT,CT}
    zone_terms = workspace.zone_terms
    zone_coeffs = workspace.zone_coeffs
    while length(zone_terms) < length(zone_caches)
        push!(zone_terms, TT[])
        push!(zone_coeffs, CT[])
    end

    for zone_id in eachindex(zone_caches)
        _ensurelength!(zone_terms[zone_id], n_records[zone_id])
        _ensurelength!(zone_coeffs[zone_id], n_records[zone_id])
    end

    return zone_terms, zone_coeffs, _zonelabels!(workspace, n_records)
end

function _zonelabels!(workspace::LayerWorkspace, n_records::Vector{Int})
    zone_labels = workspace.zone_labels
    while length(zone_labels) < length(n_records)
        push!(zone_labels, Int[])
    end

    for zone_id in eachindex(n_records)
        _ensurelength!(zone_labels[zone_id], n_records[zone_id])
    end
    return zone_labels
end

"""
    _countrecords!(partition_counts, n_bits, positions, plan, source)

Counts the records of each partition that the pairs of Pauli string and coefficient of `source` make.
Returns `false` if an orbit spans more stages than a label holds.
"""
function _countrecords!(partition_counts, n_bits::Int, positions::Vector{Int32}, plan::SubLayerPlan, source)
    for (pstr, _) in source
        representative, _, n_stages = _locateinorbit!(positions, plan, pstr)
        if n_stages > _MAX_STAGES
            return false
        end
        partition_counts[_partitionof(_hashbits(representative), n_bits)] += 1
    end
    return true
end

"""
    _writezonerecords!(zone_terms, zone_coeffs, zone_labels, zone_starts, cursors, n_bits, n_zone_bits, positions, plan, source)

Writes a record for every pair of Pauli string and coefficient of `source` to where `cursors` points for its partition,
in the arrays of the zone that the partition belongs to.
"""
function _writezonerecords!(zone_terms::Vector{Vector{TT}}, zone_coeffs::Vector{Vector{CT}}, zone_labels::Vector{Vector{Int}},
    zone_starts::Vector{Int}, cursors, n_bits::Int, n_zone_bits::Int, positions::Vector{Int32}, plan::SubLayerPlan{TT}, source) where {TT,CT}

    for (pstr, coeff) in source
        representative, coordinate, n_stages = _locateinorbit!(positions, plan, pstr)
        partition = _partitionof(_hashbits(representative), n_bits)
        zone_id = ((partition - 1) >> n_zone_bits) + 1

        index = cursors[partition] - zone_starts[zone_id] + 1
        zone_terms[zone_id][index] = representative
        zone_coeffs[zone_id][index] = coeff
        zone_labels[zone_id][index] = _label(representative, coordinate, n_stages)
        cursors[partition] += 1
    end

    return zone_terms
end

# the outbox of a zone, which sorts what it is given by the zones that own it
struct OutboxSink{MS}
    outbox::MS
end

@inline function _emit!(sink::OutboxSink, pstr, coeff)
    push!(sink.outbox, pstr, coeff)
    return
end

function _emitblock!(sink::OutboxSink, orbit_terms, block_coeffs, block_present, block_start::Int, n_entries::Int)
    for entry in 1:n_entries
        if block_present[block_start+entry]
            push!(sink.outbox, orbit_terms[entry], block_coeffs[block_start+entry])
        end
    end
    return
end
