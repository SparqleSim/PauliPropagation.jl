###
##
# The classes of a `RotationLayer`. The Pauli strings that anticommute with the same rotations of the layer, and agree on
# what those rotations leave unchanged, form a class, and those rotations keep a string within its class, since they
# commute with each other. The strings of a class differ only in its key bits, the bits that its rotations flip.
# Within a partition the records are grouped by class, and every class is rotated one rotation after the other in the
# order of the layer, each rotation mixing every two strings that its generator turns into each other and truncating
# both, as the rotations one by one do: a class of few key bits as a dense block (rotationlayerblocks.jl), any other in a
# table of its own.
##
###

# The rotations of the layer that one pass rotates class by class: all of them, unless a qubit is acted on with two
# different Paulis. The key of a class keeps, on every qubit that its rotations act on with one Pauli, whether the string
# anticommutes with that Pauli there, and so tells apart strings that anticommute with different rotations. A qubit acted
# on with two Paulis keeps neither, so strings with different rotations could share a key there. In a layer of commuting
# rotations that happens only for a rotation and its swapped pair, as XZ on (1, 2) and on (2, 1), whose lowest qubit is the
# same, so that they lie in different sublayers: such a layer takes one pass per sublayer.
function _classpasses(layer::RotationLayer)
    if _actswithtwopaulis(layer)
        return layer.sublayers
    else
        return (eachindex(layer.qinds),)
    end
end

# whether a qubit is the first qubit of one rotation and the second of another, with a different Pauli
function _actswithtwopaulis(layer::RotationLayer)
    if length(layer.symbols) < 2 || layer.symbols[1] == layer.symbols[2]
        return false
    end
    first_qubits = Set(qinds[1] for qinds in layer.qinds)
    return any(qinds -> qinds[2] in first_qubits, layer.qinds)
end


### The plan of a pass

"""
    _prepareclasses(layer::RotationLayer, theta, TT, CT, nqubits, rotations=eachindex(layer.qinds); min_abs_coeff=0)

The `rotations` of the `layer` with the parameter `theta`, by default all of them, prepared for rotating the classes of Pauli
strings of the type `TT` with coefficients of the type `CT` on `nqubits` qubits, where coefficients below `min_abs_coeff`
are truncated.
Every rotation has a position in the order in which the layer applies them, and a reader finds the rotations that
anticommute with a string as the set of their positions.
"""
function _prepareclasses(layer::RotationLayer, theta, ::Type{TT}, ::Type{CT}, nqubits::Int, rotations=eachindex(layer.qinds);
    min_abs_coeff::Real=0) where {TT,CT}

    symbols = layer.symbols
    n_rotations = length(layer.qinds)

    masks = zeros(TT, n_rotations)
    qinds = fill((0, 0), n_rotations)
    for rotation in rotations
        rotation_qinds = layer.qinds[rotation]
        _check_qind_range(nqubits, rotation_qinds)
        masks[rotation] = symboltoint(TT, symbols, rotation_qinds)
        qinds[rotation] = (rotation_qinds[1], get(rotation_qinds, 2, 0))
    end
    angles = [_rotationangle(theta, index) for index in 1:n_rotations]
    sines = sin.(angles)

    # the place of every rotation in the order in which the layer applies them
    positions = zeros(Int, n_rotations)
    rotation_at_position = zeros(Int32, n_rotations)
    position = 0
    for sublayer in layer.sublayers, rotation in sublayer
        position += 1
        positions[rotation] = position
        rotation_at_position[position] = rotation
    end

    symbol_codes = (UInt8(symboltoint(symbols[1])), UInt8(symboltoint(get(symbols, 2, :I))))
    local_mask = symbol_codes[1] | (symbol_codes[2] << 2)
    signs = _signsfor(CT, _localsigns(local_mask))
    reader = _classreader(layer, rotations, masks, qinds, positions, symbol_codes, TT, nqubits)

    # A rotation on one qubit leaves every Pauli string of its class with one of two Paulis there, the lower entry of a
    # pair with the one whose key bit is clear, so its signs are the same for every pair of every class.
    lower_paulis = symbol_codes[1] == 0x02 ? 0x01 : 0x02
    lower_signs = (signs[lower_paulis+1], signs[(lower_paulis⊻local_mask)+1])

    return (; masks, qinds, cosines=cos.(angles), sines, positions, rotation_at_position, local_mask, signs,
        acts_on_one_qubit=symbol_codes[2] == 0x00, lower_signs, min_coeffs_to_make=_mincoefftomake.(min_abs_coeff, sines), reader)
end

# the rotations of the class of `pstr`, those that anticommute with it, in the order of the layer, and their number
@inline function _rotationsinorder!(rotations::Vector{Int32}, plan, positions::NTuple{W,UInt64}) where {W}
    if length(rotations) < length(plan.rotation_at_position)
        throw(ArgumentError("the rotations do not fit the workspace"))
    end
    n_found = 0
    for word in 1:W
        bits = positions[word]
        while bits != 0
            position = 64 * (word - 1) + trailing_zeros(bits)
            bits &= bits - one(UInt64)
            n_found += 1
            @inbounds rotations[n_found] = plan.rotation_at_position[position+1]
        end
    end
    return n_found
end

# the sign a rotation gives the Pauli string it creates from each combination of Paulis on its qubits, 0 where it commutes with them
function _localsigns(local_mask::UInt8)
    function sign_of(index)
        paulis = UInt8(index - 1)
        if commutes(local_mask, paulis)
            return Int8(0)
        end
        _, sign = paulirotationproduct(local_mask, paulis)
        return Int8(sign)
    end
    return ntuple(sign_of, 16)
end

# The signs as the real type of floating-point coefficients, which they multiply as the integers do, so that they are not
# converted for every entry.
_signsfor(::Type{CT}, signs) where {CT<:Union{AbstractFloat,Complex{<:AbstractFloat}}} = map(sign -> convert(real(CT), sign), signs)
_signsfor(::Type, signs) = signs

# the Paulis of `pstr` on the qubits of `rotation`, the first qubit in the low bits
@inline function _localpaulis(plan, pstr, rotation::Integer)
    first_qind, second_qind = plan.qinds[rotation]
    paulis = Int(_getpaulibits(pstr, first_qind))
    if second_qind != 0
        paulis |= Int(_getpaulibits(pstr, second_qind)) << 2
    end
    return paulis
end


### Readers: the rotations that anticommute with a string, and the qubits they act on

# A layer whose rotations fall into more groups than this is read through byte tables instead.
const _MAX_ROTATION_GROUPS = 4

# whether the rotations are read in groups, from the whole Pauli string at once
_readsingroups(groups) = length(groups) <= _MAX_ROTATION_GROUPS

# The rotations whose qubits are the same distance apart with the same Paulis on them: the low bit of the lower qubit of
# every rotation, the distance from the lower to the upper qubit in bits (0 for rotations on one qubit), the Paulis of the
# generator on the lower and on the upper qubit, and for every qubit the position of the rotation of which it is the lower qubit.
struct _RotationGroup{TT}
    lower_mask::TT
    shift::Int
    lower_pauli::UInt8
    upper_pauli::UInt8
    position_at_lower::Vector{Int32}
end

# Finds the rotations that anticommute with a string, as the bits of their positions in a tuple of `W` words, and the
# qubits they act on. A layer whose rotations fall into few groups is read group by group from the whole string at once.
# Any other layer is read through byte tables: a string anticommutes with a rotation if the bits where the string and
# the rotation's generator with its two bits swapped on every qubit are both set are odd in number, so the rotations a
# string anticommutes with are the sum of those of its bits. For every byte of a string and each of its values, `tables`
# holds the positions of the rotations that a string with only those bits anticommutes with.
struct _ClassReader{TT,W}
    reads_in_groups::Bool

    # the Paulis of the generator, the low bit of every qubit a rotation acts on with its first and with its second, and the groups
    symbol_codes::NTuple{2,UInt8}
    role_masks::NTuple{2,TT}
    groups::Vector{_RotationGroup{TT}}

    # the tables, and by position the low bits of the qubits the rotation acts on with X, with Y and with Z
    tables::Vector{NTuple{W,UInt64}}
    touched::Vector{NTuple{3,TT}}
end

function _classreader(layer::RotationLayer, rotations, masks::Vector{TT}, qinds, positions, symbol_codes, ::Type{TT}, nqubits::Int) where {TT}
    n_words = cld(length(layer.qinds), 64)
    role_masks = [zero(TT), zero(TT)]
    group_ids = Dict{Tuple{Int,UInt8,UInt8},Int}()
    groups = _RotationGroup{TT}[]
    shared_lower_qubit = false

    for rotation in rotations
        first_qind, second_qind = qinds[rotation]
        role_masks[1] |= symboltoint(TT, :X, first_qind)
        if second_qind != 0
            role_masks[2] |= symboltoint(TT, :X, second_qind)
        end

        shift, lower_pauli, upper_pauli, lower_qind = if second_qind == 0
            0, symbol_codes[1], 0x00, first_qind
        elseif first_qind < second_qind
            2 * (second_qind - first_qind), symbol_codes[1], symbol_codes[2], first_qind
        else
            2 * (first_qind - second_qind), symbol_codes[2], symbol_codes[1], second_qind
        end

        group_id = get(group_ids, (shift, lower_pauli, upper_pauli), 0)
        if group_id == 0
            push!(groups, _RotationGroup{TT}(zero(TT), shift, lower_pauli, upper_pauli, zeros(Int32, nqubits)))
            group_id = length(groups)
            group_ids[(shift, lower_pauli, upper_pauli)] = group_id
        end
        group = groups[group_id]

        # a rotation that comes twice cannot be found through its lower qubit
        shared_lower_qubit |= group.position_at_lower[lower_qind] != 0
        group.position_at_lower[lower_qind] = positions[rotation]
        groups[group_id] = _RotationGroup{TT}(group.lower_mask | symboltoint(TT, :X, lower_qind), shift, lower_pauli, upper_pauli,
            group.position_at_lower)
    end

    # A qubit that is the first qubit of one rotation and the second of another is read with either Pauli,
    # which the groups cannot tell apart when the two Paulis differ.
    mixed_roles = symbol_codes[2] != 0x00 && symbol_codes[1] != symbol_codes[2] && !iszero(role_masks[1] & role_masks[2])
    reads_in_groups = _readsingroups(groups) && !shared_lower_qubit && !mixed_roles

    tables, touched = if reads_in_groups
        NTuple{n_words,UInt64}[], NTuple{3,TT}[]
    else
        _anticommutationtables(rotations, masks, qinds, positions, symbol_codes, TT, Val(n_words))
    end
    return _ClassReader{TT,n_words}(reads_in_groups, symbol_codes, (role_masks[1], role_masks[2]), groups, tables, touched)
end

# the byte tables of the rotations, and the qubits that each rotation acts on, by position
function _anticommutationtables(rotations, masks::Vector{TT}, qinds, positions, symbol_codes, ::Type{TT}, ::Val{W}) where {TT,W}
    no_positions = ntuple(_ -> zero(UInt64), Val(W))
    n_bits = 8 * sizeof(TT)
    columns = fill(no_positions, n_bits)
    touched = fill((zero(TT), zero(TT), zero(TT)), length(positions))

    for rotation in rotations
        position = positions[rotation] - 1
        position_bits = _withposition(no_positions, position)
        mask = masks[rotation]
        low_bits = alternatingmask(mask)
        swapped = _limbs(((mask >> 1) & low_bits) | ((mask & low_bits) << 1))
        for limb_index in eachindex(swapped)
            limb = swapped[limb_index]
            while limb != 0
                bit = 64 * (limb_index - 1) + trailing_zeros(limb)
                limb &= limb - one(UInt64)
                columns[bit+1] = _xorwords(columns[bit+1], position_bits)
            end
        end

        first_qind, second_qind = qinds[rotation]
        touched_here = _touch((zero(TT), zero(TT), zero(TT)), symbol_codes[1], symboltoint(TT, :X, first_qind))
        if second_qind != 0
            touched_here = _touch(touched_here, symbol_codes[2], symboltoint(TT, :X, second_qind))
        end
        touched[position+1] = touched_here
    end

    # the entry of every value of a byte is that of the value without its lowest bit, plus the column of that bit
    n_bytes = sizeof(TT)
    tables = Vector{NTuple{W,UInt64}}(undef, 256 * n_bytes)
    for byte in 0:n_bytes-1
        tables[256*byte+1] = no_positions
        for value in 1:255
            column = columns[8*byte+trailing_zeros(value)+1]
            tables[256*byte+value+1] = _xorwords(tables[256*byte+(value&(value-1))+1], column)
        end
    end
    return tables, touched
end

@inline _xorwords(a::NTuple{W,UInt64}, b::NTuple{W,UInt64}) where {W} = ntuple(word -> a[word] ⊻ b[word], Val(W))

# the positions with the bit of `position` set as well
@inline function _withposition(positions::NTuple{W,UInt64}, position::Int) where {W}
    word = (position >> 6) + 1
    return ntuple(w -> positions[w] | ifelse(w == word, one(UInt64) << (position & 63), zero(UInt64)), Val(W))
end

"""
    _anticommuting(reader, pstr)

The positions of the rotations that anticommute with `pstr`, as bits, and the low bits of the qubits that those
rotations act on with X, with Y and with Z.
"""
@inline function _anticommuting(reader::_ClassReader{TT,W}, pstr::TT) where {TT,W}
    if !reader.reads_in_groups
        return _anticommutingbytables(reader, pstr)
    end

    candidates = _candidates(reader, pstr)
    positions = ntuple(_ -> zero(UInt64), Val(W))
    touched = (zero(TT), zero(TT), zero(TT))
    for group in reader.groups
        anticommuting = _groupanticommuting(group, candidates)
        touched = _touch(touched, group.lower_pauli, anticommuting)
        if group.shift != 0
            touched = _touch(touched, group.upper_pauli, _shiftup(anticommuting, group.shift))
        end
        limbs = _limbs(anticommuting)
        for limb_index in eachindex(limbs)
            limb = limbs[limb_index]
            while limb != 0
                qind = 32 * (limb_index - 1) + (trailing_zeros(limb) >> 1) + 1
                limb &= limb - one(UInt64)
                positions = _withposition(positions, Int(group.position_at_lower[qind]) - 1)
            end
        end
    end
    return positions, touched
end

# The positions, from the entries of the bytes of the string that are not zero, and the qubits from the positions.
# Kept out of line, so that the readers of layers read in groups stay small where they are inlined.
@noinline function _anticommutingbytables(reader::_ClassReader{TT,W}, pstr::TT) where {TT,W}
    positions = ntuple(_ -> zero(UInt64), Val(W))
    limbs = _limbs(pstr)
    for limb_index in eachindex(limbs)
        limb = limbs[limb_index]
        while limb != 0
            byte_in_limb = trailing_zeros(limb) >> 3
            value = (limb >> (8 * byte_in_limb)) & 0xff
            limb &= ~(UInt64(0xff) << (8 * byte_in_limb))
            positions = _xorwords(positions, reader.tables[256*(8*(limb_index-1)+byte_in_limb)+(value%Int)+1])
        end
    end

    touched = (zero(TT), zero(TT), zero(TT))
    for word in 1:W
        bits = positions[word]
        while bits != 0
            position = 64 * (word - 1) + trailing_zeros(bits)
            bits &= bits - one(UInt64)
            touched_here = reader.touched[position+1]
            touched = (touched[1] | touched_here[1], touched[2] | touched_here[2], touched[3] | touched_here[3])
        end
    end
    return positions, touched
end

# the low bits of the qubits that the rotations anticommuting with `pstr` act on with X, with Y and with Z
@inline function _touchedqubits(reader::_ClassReader{TT}, pstr::TT) where {TT}
    if !reader.reads_in_groups
        return last(_anticommutingbytables(reader, pstr))
    end

    candidates = _candidates(reader, pstr)
    touched = (zero(TT), zero(TT), zero(TT))
    for group in reader.groups
        anticommuting = _groupanticommuting(group, candidates)
        touched = _touch(touched, group.lower_pauli, anticommuting)
        if group.shift != 0
            touched = _touch(touched, group.upper_pauli, _shiftup(anticommuting, group.shift))
        end
    end
    return touched
end

# the low bit of every qubit in `low_bits`, added to the qubits that rotations act on with the Pauli `pauli`
@inline function _touch(touched::NTuple{3,TT}, pauli::UInt8, low_bits::TT) where {TT}
    touched_x, touched_y, touched_z = touched
    if pauli == 0x01
        touched_x |= low_bits
    elseif pauli == 0x02
        touched_y |= low_bits
    elseif pauli == 0x03
        touched_z |= low_bits
    end
    return (touched_x, touched_y, touched_z)
end

# the low bit of every qubit on which `pstr` anticommutes with the Pauli `symbol_code`
@inline function _locallyanticommuting(symbol_code::UInt8, pstr::TT) where {TT}
    low_bits = alternatingmask(pstr)
    if symbol_code == 0x01
        # X anticommutes with Y and Z, which have the high bit set
        return _shiftdown(pstr, 1) & low_bits
    elseif symbol_code == 0x02
        # Y anticommutes with X and Z, which have the low bit set
        return pstr & low_bits
    else
        # Z anticommutes with X and Y, whose bits differ
        return (pstr ⊻ _shiftdown(pstr, 1)) & low_bits
    end
end

# the low bit of every qubit on which `pstr` anticommutes with the rotation acting there
@inline function _candidates(reader::_ClassReader, pstr)
    candidates = _locallyanticommuting(reader.symbol_codes[1], pstr) & reader.role_masks[1]
    if reader.symbol_codes[2] != 0x00
        candidates |= _locallyanticommuting(reader.symbol_codes[2], pstr) & reader.role_masks[2]
    end
    return candidates
end

# the low bit of the lower qubit of every rotation of the group that anticommutes with the string of `candidates`
@inline function _groupanticommuting(group::_RotationGroup, candidates)
    if group.shift == 0
        return candidates & group.lower_mask
    else
        # a rotation on two qubits anticommutes if the Pauli string anticommutes with it on exactly one of them
        return (candidates ⊻ _shiftdown(candidates, group.shift)) & group.lower_mask
    end
end

# Shifts of a whole Pauli string. Below 64 bits, a shift only moves bits between neighbouring limbs.
@inline _shiftdown(pstr, shift::Int) = pstr >> shift
@inline _shiftup(pstr, shift::Int) = pstr << shift

@inline function _shiftdown(pstr::NTupleInteger{N}, shift::Int) where {N}
    if !(0 <= shift < 64)
        return pstr >> shift
    end
    limbs = pstr.limbs
    shifted_limb(i) = (limbs[i] >> shift) | ifelse(i < N, limbs[min(i + 1, N)] << (64 - shift), zero(UInt64))
    return NTupleInteger{N}(ntuple(shifted_limb, Val(N)))
end

@inline function _shiftup(pstr::NTupleInteger{N}, shift::Int) where {N}
    if !(0 <= shift < 64)
        return pstr << shift
    end
    limbs = pstr.limbs
    shifted_limb(i) = (limbs[i] << shift) | ifelse(i > 1, limbs[max(i - 1, 1)] >> (64 - shift), zero(UInt64))
    return NTupleInteger{N}(ntuple(shifted_limb, Val(N)))
end


### Keys

# The Pauli string with every qubit that the rotations it anticommutes with act on reduced to what they leave unchanged:
# X leaves the high bit, Y the low bit and Z the parity of the two, and a qubit acted on with two different Paulis is cleared.
# All Pauli strings of a class have the same key.
@inline function _classkey(pstr::TT, touched::NTuple{3,TT}) where {TT}
    touched_x, touched_y, touched_z = touched
    mixed = (touched_x & touched_y) | (touched_x & touched_z) | (touched_y & touched_z)
    key = pstr & ~(mixed | _shiftup(mixed, 1))
    key &= ~(touched_x & ~mixed)
    key &= ~_shiftup(touched_y & ~mixed, 1)
    low_z = key & touched_z & ~mixed
    return key ⊻ (low_z | _shiftup(low_z, 1))
end

# The bits that tell the strings of a class apart, on the qubits that its rotations touch: the low bit where they touch
# with X or Z, the high bit where they touch with Y, and both where with two Paulis.
@inline function _keybits(touched::NTuple{3,TT}) where {TT}
    touched_x, touched_y, touched_z = touched
    mixed = (touched_x & touched_y) | (touched_x & touched_z) | (touched_y & touched_z)
    return touched_x | touched_z | mixed | _shiftup(touched_y | mixed, 1)
end

# The label of the record of a Pauli string: a hash above `_HASH_SHIFT`, which picks the partition, and in the lowest bit
# whether the string anticommutes with any rotation. The hash is that of the key of its class, or else that of the string,
# so that the strings no rotation touches are spread over the partitions as well.
@inline function _classlabel(plan, pstr)
    touched = _touchedqubits(plan.reader, pstr)
    if iszero(touched[1] | touched[2] | touched[3])
        return (_hashbits(pstr) << _HASH_SHIFT) % Int
    end
    return ((_hashbits(_classkey(pstr, touched)) << _HASH_SHIFT) | one(UInt64)) % Int
end

### Rotating the classes of a partition

"""
    _rotatepartition!(output, task, plan, truncfunc, record_terms, record_coeffs, record_labels, lo, hi)

Groups the Pauli strings of the records `lo` to `hi` by the hash of their class, rotates every class, and writes what the
truncations keep to `output`. A Pauli string that anticommutes with no rotation goes to `output` as it is.
"""
function _rotatepartition!(output, task, plan, truncfunc::F, record_terms::Vector{TT}, record_coeffs::Vector{CT},
    record_labels::Vector{Int}, lo::Int, hi::Int) where {F,TT,CT}

    if lo > hi
        return output
    end
    if !(1 <= lo && hi <= min(length(record_terms), length(record_coeffs), length(record_labels)))
        throw(ArgumentError("the records $lo to $hi are not among the records"))
    end
    n_records = hi - lo + 1
    group_of = _ensurelength!(task.group_of, n_records)
    group_hashes = _ensurelength!(task.group_hashes, n_records)
    group_starts = _ensurelength!(task.group_starts, n_records + 1)

    # the groups found so far, through their hash
    table_length = max(16, nextpow(2, 2 * n_records))
    slot_mask = table_length - 1
    slots = _ensurelength!(task.slots, table_length)
    fill!(view(slots, 1:table_length), zero(Int32))
    n_groups = 0

    for i in lo:hi
        label = record_labels[i]
        if !_hasclass(label)
            group_of[i-lo+1] = 0
            pstr = record_terms[i]
            coeff = record_coeffs[i]
            if !@inline(truncfunc(pstr, coeff))
                _emit!(output, pstr, coeff)
            end
            continue
        end

        hashbits = _labelhash(label)
        slot = Int(hashbits & (slot_mask % UInt64)) + 1
        group = Int(slots[slot])
        while group != 0 && group_hashes[group] != hashbits
            slot = (slot & slot_mask) + 1
            group = Int(slots[slot])
        end
        if group == 0
            n_groups += 1
            group = n_groups
            group_hashes[group] = hashbits
            group_starts[group] = 0
            slots[slot] = group
        end
        group_starts[group] += 1
        group_of[i-lo+1] = group
    end

    # the records one group after the other
    next_start = 1
    for group in 1:n_groups
        n_here = group_starts[group]
        group_starts[group] = next_start
        next_start += n_here
    end
    group_starts[n_groups+1] = next_start
    class_records = _ensurelength!(task.class_records, n_records)
    for i in 1:n_records
        group = group_of[i]
        if group != 0
            class_records[group_starts[group]] = i
            group_starts[group] += 1
        end
    end
    for group in n_groups:-1:1
        group_starts[group+1] = group_starts[group]
    end
    group_starts[1] = 1

    for group in 1:n_groups
        _rotategroup!(output, task, plan, truncfunc, record_terms, record_coeffs, lo, group_starts[group], group_starts[group+1] - 1)
    end
    return output
end

# The records `class_records[first:last]`, counted from `lo`, share the hash of their class's key: they are one class,
# or, where two keys share a hash, several. Every class is moved to the front in turn and rotated. Within a pass no qubit
# is acted on with two different Paulis, so reducing a string on the qubits that another string's rotations touch only
# multiplies it by Paulis that commute with every rotation: the two strings have the same key that way only if they
# anticommute with the same rotations, and so belong to the same class.
function _rotategroup!(output, task, plan, truncfunc::F, record_terms::Vector{TT}, record_coeffs::Vector{CT}, lo::Int, first::Int,
    last::Int) where {F,TT,CT}

    class_records = task.class_records
    while first <= last
        first_pstr = record_terms[lo-1+class_records[first]]
        positions, touched = _anticommuting(plan.reader, first_pstr)
        key = _classkey(first_pstr, touched)
        class_end = first
        for index in first:last
            record = class_records[index]
            if _classkey(record_terms[lo-1+record], touched) == key
                class_records[index] = class_records[class_end]
                class_records[class_end] = record
                class_end += 1
            end
        end
        _rotateclass!(output, task, plan, truncfunc, record_terms, record_coeffs, lo, first, class_end - 1, positions, touched)
        first = class_end
    end
    return output
end

# the most key bits of a class that is rotated as a dense block
const _MAX_BLOCK_KEY_BITS = 8

# whether a class of `n_key_bits` key bits is rotated as a dense block, rather than in a table
_rotatesasblock(n_key_bits::Int) = n_key_bits <= _MAX_BLOCK_KEY_BITS

# the most bits that a key gathered into an integer can have
const _MAX_KEY_BITS = 64

"""
    _rotateclass!(output, task, plan, truncfunc, record_terms, record_coeffs, lo, first, last, positions, touched)

Rotates the class of the records `class_records[first:last]`, counted from `lo`, one rotation after the other, and writes
what the truncations keep to `output`: as a dense block if the class has few key bits, and in a table otherwise.
`positions` and `touched` are what `_anticommuting` reads from any string of the class.
"""
function _rotateclass!(output, task, plan, truncfunc::F, record_terms::Vector{TT}, record_coeffs::Vector{CT}, lo::Int,
    first::Int, last::Int, positions, touched) where {F,TT,CT}

    n_rotations = _rotationsinorder!(task.rotations, plan, positions)
    key_bits = _keybits(touched)
    n_key_bits = sum(count_ones, _limbs(key_bits))

    # the kernels are compiled for keys of each type, behind this barrier
    if _rotatesasblock(n_key_bits)
        _rotateblock!(output, task, plan, truncfunc, record_terms, record_coeffs, lo, first, last, n_rotations, key_bits, n_key_bits)
    elseif n_key_bits <= _MAX_KEY_BITS
        _rotatekeyedclass!(output, task, plan, truncfunc, record_terms, record_coeffs, lo, first, last, n_rotations,
            task.entry_keys, key_bits, n_key_bits)
    else
        _rotatekeyedclass!(output, task, plan, truncfunc, record_terms, record_coeffs, lo, first, last, n_rotations,
            task.entry_terms, nothing, n_key_bits)
    end
    return output
end


### Rotating a class in a table

# The Pauli strings of a class differ only in the bits that its rotations flip: the low bit of a qubit that they touch
# with X, the high bit of one touched with Y, and both bits of one touched with Z, which flip together. A qubit touched
# with two different Paulis has both of its bits flipped independently. The strings of a class are therefore told apart
# by the low bit of every qubit touched with X or Z, the high bit of every qubit touched with Y, and both bits of every
# qubit touched with two Paulis. These bits, gathered next to each other, are the key of a string in its class. A
# rotation flips a fixed set of bits of the key, so the key of the partner it makes is the key of the string with those
# bits flipped. The table of a class finds an entry by its key: through a slot of its own where the keys are few enough,
# and through a hash of the key otherwise. A class with more such bits than an integer holds is keyed by its Pauli
# strings themselves.

# the most bits that a key of its own slot can have
const _MAX_DIRECT_KEY_BITS = 20

# A coefficient of at most this fraction below the smallest kept one is checked against the truncation when the partner
# it would make is decided, so that rounding cannot hide a partner that the truncation keeps.
const _MAKE_MARGIN = 1 - 1e-12

function _rotatekeyedclass!(output, task, plan, truncfunc::F, record_terms::Vector{TT}, record_coeffs::Vector{CT}, lo::Int,
    first::Int, last::Int, n_rotations::Int, entry_keys::Vector{K}, key_bits, n_key_bits::Int) where {F,TT,CT,K}

    _openclass!(task, last - first + 1, K === UInt64 && n_key_bits <= _MAX_DIRECT_KEY_BITS, n_key_bits)
    _addmembers!(task, entry_keys, key_bits, record_terms, record_coeffs, lo, first, last)

    for step in 1:n_rotations
        rotation = Int(task.rotations[step])
        _applytoclass!(task, plan, truncfunc, entry_keys, _keyof(plan.masks[rotation], key_bits), rotation, Int32(step), CT)
    end

    _emitclass!(output, task)
    _closeclass!(task, entry_keys)
    return output
end

# The key of `pstr` in its class: its key bits gathered into an integer, every key bit that is set moved to the number of
# key bits below it. Without key bits, the Pauli string itself.
@inline function _keyof(pstr::TT, key_bits::TT) where {TT<:Integer}
    set_limbs = _limbs(pstr & key_bits)
    key_limbs = _limbs(key_bits)
    key = zero(UInt64)
    n_below = 0
    for limb_index in eachindex(key_limbs)
        set_bits = set_limbs[limb_index]
        while set_bits != 0
            lowest = set_bits & (~set_bits + one(UInt64))
            set_bits &= set_bits - one(UInt64)
            key |= one(UInt64) << ((n_below + count_ones(key_limbs[limb_index] & (lowest - one(UInt64)))) & 63)
        end
        n_below += count_ones(key_limbs[limb_index])
    end
    return key
end

@inline _keyof(pstr, ::Nothing) = pstr

# The rotation mixes every two entries that its generator turns into each other, and every entry without a present
# partner keeps cos θ of its coefficient and makes the partner, if the truncation keeps that. Most entries do no more
# than keep cos θ, so the entries that do more are listed first, and those that only keep cos θ are scaled after them.
# A pair is listed at the entry whose key has the lowest bit of `key_mask` clear. Entries made by the rotation come after
# the others and are not visited.
function _applytoclass!(task, plan, truncfunc::F, entry_keys::Vector{K}, key_mask::K, rotation::Int, step::Int32, ::Type{CT}) where {F,K,CT}

    mask = plan.masks[rotation]
    cos_val = plan.cosines[rotation]
    sin_val = plan.sines[rotation]
    signs = plan.signs
    local_mask = Int(plan.local_mask)
    lower_bit = _lowestbit(key_mask)

    min_coeff_to_make = plan.min_coeffs_to_make[rotation]

    # the Paulis on the qubits of the rotation are read from the words that hold them
    first_qind, second_qind = plan.qinds[rotation]
    first_bit = 2 * (first_qind - 1)
    second_bit = 2 * (max(second_qind, 1) - 1)
    reads_second = second_qind != 0

    # nothing grows while the rotation visits the entries
    n_before = task.n_entries
    _prepareentries!(task, entry_keys, n_before)
    entry_terms = task.entry_terms
    entry_coeffs = task.entry_coeffs
    entry_present = task.entry_present
    entry_steps = task.entry_steps
    events = task.events
    event_partners = task.event_partners
    slots = task.class_slots
    slot_mask = task.class_table_length - 1
    hash_shift = task.class_hash_shift

    # the entries that the rotation does more to than keeping cos θ, without branching on them
    n_events = 0
    @inbounds for entry in 1:n_before
        key = entry_keys[entry]
        partner, _ = _findentry(slots, entry_keys, slot_mask, hash_shift, key ⊻ key_mask)
        present = entry_present[entry]
        partner_present = (partner != 0) & entry_present[max(partner, 1)]
        is_lower_of_pair = present & partner_present & iszero(key & lower_bit)
        makes_partner = present & !partner_present & (abs(entry_coeffs[entry]) >= min_coeff_to_make)
        events[n_events+1] = entry % Int32
        event_partners[n_events+1] = partner % Int32
        n_events += is_lower_of_pair | makes_partner
    end

    n_entries = n_before
    @inbounds for event in 1:n_events
        entry = Int(events[event])
        partner = Int(event_partners[event])
        pstr = entry_terms[entry]
        partner_pstr = pstr ⊻ mask

        paulis = ((PropagationBase._wordat(pstr, first_bit) >> (first_bit & 63)) & 0x03) % Int
        if reads_second
            paulis |= (((PropagationBase._wordat(pstr, second_bit) >> (second_bit & 63)) & 0x03) % Int) << 2
        end
        sign_to_partner = signs[(paulis&15)+1]
        sign_from_partner = signs[((paulis⊻local_mask)&15)+1]

        # an entry that was truncated holds zero
        coeff = entry_coeffs[entry]
        partner_coeff = ifelse(partner == 0, zero(CT), entry_coeffs[max(partner, 1)])
        new_coeff = mergefunc(coeff * cos_val, partner_coeff * sin_val * sign_from_partner)
        new_partner_coeff = mergefunc(partner_coeff * cos_val, coeff * sin_val * sign_to_partner)

        keep = !@inline(truncfunc(pstr, new_coeff))
        keep_partner = !@inline(truncfunc(partner_pstr, new_partner_coeff))
        entry_coeffs[entry] = ifelse(keep, new_coeff, zero(CT))
        entry_present[entry] = keep
        entry_steps[entry] = step

        if partner != 0
            entry_coeffs[partner] = ifelse(keep_partner, new_partner_coeff, zero(CT))
            entry_present[partner] = keep_partner
            entry_steps[partner] = step
        elseif keep_partner
            partner_key = entry_keys[entry] ⊻ key_mask
            _, slot = _findentry(slots, entry_keys, slot_mask, hash_shift, partner_key)
            n_entries += 1
            entry_terms[n_entries] = partner_pstr
            entry_keys[n_entries] = partner_key
            entry_coeffs[n_entries] = new_partner_coeff
            entry_present[n_entries] = true
            entry_steps[n_entries] = step
            slots[slot+1] = n_entries % Int32
        end
    end

    # every other present entry keeps cos θ of its coefficient
    @inbounds for entry in 1:n_before
        coeff = entry_coeffs[entry]
        scales = entry_present[entry] & (entry_steps[entry] != step)
        new_coeff = coeff * cos_val
        keep = !@inline(truncfunc(entry_terms[entry], new_coeff))
        entry_coeffs[entry] = ifelse(scales, ifelse(keep, new_coeff, zero(CT)), coeff)
        entry_present[entry] = ifelse(scales, keep, entry_present[entry])
    end

    task.n_entries = n_entries
    return task
end

# The smallest coefficient that can make a partner the truncation keeps, of a rotation with this sine. Without a smallest
# kept coefficient, any can.
function _mincoefftomake(min_abs_coeff::Real, sin_val)
    if iszero(min_abs_coeff)
        return zero(float(min_abs_coeff))
    end
    return _MAKE_MARGIN * min_abs_coeff / abs(sin_val)
end

# the lowest set bit of a key
@inline _lowestbit(key::UInt64) = key & (~key + one(UInt64))

@inline function _lowestbit(pstr::TT) where {TT}
    limbs = _limbs(pstr)
    for limb_index in eachindex(limbs)
        if limbs[limb_index] != 0
            return _shiftup(one(TT), 64 * (limb_index - 1) + trailing_zeros(limbs[limb_index]))
        end
    end
    return zero(TT)
end


### The table of a class

# A hash for the table of a class: the sum of the limbs, each times an odd factor, whose highest bits pick the slot.
@inline function _classhash(pstr)
    limbs = _limbs(pstr)
    folded = zero(UInt64)
    for i in eachindex(limbs)
        folded += limbs[i] * _foldfactor(i)
    end
    return folded
end

@inline _classhash(key::UInt64) = key * 0x9e3779b97f4a7c15

# An empty table for a class of `n_members` Pauli strings. Where every key has a slot of its own, the slot of a key is the
# key; otherwise the table is filled to at most an eighth through a hash of the keys. The slots are all zero outside a
# class: a class clears the slots it used when it closes, and a class that did not close leaves them to be cleared here.
function _openclass!(task, n_members::Int, has_direct_slots::Bool, n_key_bits::Int)
    if has_direct_slots
        table_length = 1 << n_key_bits
        task.class_hash_shift = 0
    else
        table_length = max(16, nextpow(2, 8 * n_members))
        task.class_hash_shift = 64 - trailing_zeros(table_length)
    end
    slots = task.class_slots
    n_zeroed = length(slots)
    if task.class_is_open
        fill!(slots, zero(Int32))
    end
    if table_length > n_zeroed
        resize!(slots, max(table_length, 2 * n_zeroed))
        fill!(view(slots, n_zeroed+1:length(slots)), zero(Int32))
    end
    task.class_is_open = true
    task.class_table_length = table_length
    task.n_entries = 0
    return task
end

# clears the slots that the class used
function _closeclass!(task, entry_keys::Vector)
    slots = task.class_slots
    if task.class_hash_shift == 0
        for entry in 1:task.n_entries
            slots[(entry_keys[entry]%Int)+1] = zero(Int32)
        end
    else
        fill!(view(slots, 1:task.class_table_length), zero(Int32))
    end
    task.class_is_open = false
    return task
end

# The entry with the key `key`, or 0, and the slot where it is or would be added.
@inline function _findentry(slots::Vector{Int32}, entry_keys::Vector{UInt64}, slot_mask::Int, hash_shift::Int, key::UInt64)
    if hash_shift == 0
        slot = key % Int
        return (@inbounds slots[slot+1]) % Int, slot
    end
    return _probeslots(slots, entry_keys, slot_mask, hash_shift, key)
end

@inline _findentry(slots::Vector{Int32}, entry_keys::Vector, slot_mask::Int, hash_shift::Int, key) =
    _probeslots(slots, entry_keys, slot_mask, hash_shift, key)

# The callers give slots at least the length of the table and keys for every entry in it.
@inline function _probeslots(slots::Vector{Int32}, entry_keys::Vector, slot_mask::Int, hash_shift::Int, key)
    slot = (_classhash(key) >> (hash_shift & 63)) % Int
    while true
        entry = (@inbounds slots[slot+1]) % Int
        if entry == 0 || @inbounds(entry_keys[entry]) == key
            return entry, slot
        end
        slot = (slot + 1) & slot_mask
    end
end

# The records `class_records[first:last]`, counted from `lo`, as the entries of the table of their class, where a Pauli
# string that comes twice is added up.
function _addmembers!(task, entry_keys::Vector, key_bits, record_terms::Vector{TT}, record_coeffs::Vector{CT}, lo::Int,
    first::Int, last::Int) where {TT,CT}

    class_records = task.class_records
    _prepareentries!(task, entry_keys, last - first + 1)
    entry_terms = task.entry_terms
    entry_coeffs = task.entry_coeffs
    entry_present = task.entry_present
    entry_steps = task.entry_steps
    slots = task.class_slots
    slot_mask = task.class_table_length - 1
    hash_shift = task.class_hash_shift

    n_entries = task.n_entries
    for index in first:last
        i = lo - 1 + class_records[index]
        pstr = record_terms[i]
        key = _keyof(pstr, key_bits)
        entry, slot = _findentry(slots, entry_keys, slot_mask, hash_shift, key)
        if entry == 0
            n_entries += 1
            entry_terms[n_entries] = pstr
            entry_keys[n_entries] = key
            entry_coeffs[n_entries] = record_coeffs[i]
            entry_present[n_entries] = true
            entry_steps[n_entries] = zero(Int32)
            slots[slot+1] = n_entries % Int32
        else
            entry_coeffs[entry] = mergefunc(entry_coeffs[entry], record_coeffs[i])
        end
    end
    task.n_entries = n_entries
    return task
end

# Room for twice `n_entries` entries and their events, and a hashed table that they fill to at most a quarter, so that
# a rotation can add an entry for every entry it visits without anything growing.
function _prepareentries!(task, entry_keys::Vector, n_entries::Int)
    n_room = 2 * n_entries + 1
    if n_room > min(length(task.entry_terms), length(entry_keys), length(task.entry_coeffs), length(task.entry_present),
        length(task.entry_steps), length(task.events), length(task.event_partners))
        _ensurelength!(task.entry_terms, n_room)
        _ensurelength!(entry_keys, n_room)
        _ensurelength!(task.entry_coeffs, n_room)
        _ensurelength!(task.entry_present, n_room)
        _ensurelength!(task.entry_steps, n_room)
        _ensurelength!(task.events, n_room)
        _ensurelength!(task.event_partners, n_room)
    end
    if task.class_hash_shift != 0 && task.class_table_length < 8 * n_entries
        _growclass!(task, entry_keys, 8 * n_entries)
    end
    _checkentries(task, entry_keys, n_room)
    return task
end

# a hashed table of at least `n_slots` slots with the entries that are there
function _growclass!(task, entry_keys::Vector, n_slots::Int)
    table_length = nextpow(2, n_slots)
    slots = task.class_slots
    n_zeroed = length(slots)
    if table_length > n_zeroed
        resize!(slots, max(table_length, 2 * n_zeroed))
    end
    fill!(view(slots, 1:length(slots)), zero(Int32))
    slot_mask = table_length - 1
    hash_shift = 64 - trailing_zeros(table_length)

    for entry in 1:task.n_entries
        slot = Int(_classhash(entry_keys[entry]) >> hash_shift)
        while slots[slot+1] != 0
            slot = (slot + 1) & slot_mask
        end
        slots[slot+1] = entry
    end
    task.class_table_length = table_length
    task.class_hash_shift = hash_shift
    return task
end

# the loops over the entries of a class index its arrays without bounds checks
function _checkentries(task, entry_keys::Vector, n_entries::Int)
    if n_entries > min(length(task.entry_terms), length(entry_keys), length(task.entry_coeffs), length(task.entry_present),
           length(task.entry_steps), length(task.events), length(task.event_partners)) ||
       task.class_table_length > length(task.class_slots) || !ispow2(task.class_table_length) ||
       (task.class_hash_shift != 0 && task.class_hash_shift != 64 - trailing_zeros(task.class_table_length))
        throw(ArgumentError("the $n_entries entries of a class do not fit the workspace"))
    end
    return
end

# the entries of the class that are present
function _emitclass!(output, task)
    n_entries = task.n_entries
    room = _reserve!(output, n_entries)
    for entry in 1:n_entries
        if task.entry_present[entry]
            _put!(output, room, task.entry_terms[entry], task.entry_coeffs[entry])
        end
    end
    return
end
