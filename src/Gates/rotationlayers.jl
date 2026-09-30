### rotationlayers.jl
##
# A file for layers of Pauli rotations that commute with each other.
# For example RotationLayer([:Z, :Z], staircasetopology(4)) holds the rotations RZZ_12, RZZ_23 and RZZ_34.
# A layer is propagated as a whole, which is faster than propagating its rotations one after the other.
##
###

"""
A type for a layer of Pauli rotations that commute with each other, carrying the gate generator and the qubit indices of every rotation.
"""
struct RotationLayer <: ParametrizedGate
    symbols::Vector{Symbol}
    qinds::Vector{Vector{Int}}
    sublayers::Vector{Vector{Int}}

    # a layer whose sublayers are known already, each in the order or the reverse order of its lowest qubits
    function RotationLayer(symbols::Vector{Symbol}, qinds::Vector{Vector{Int}}, sublayers::Vector{Vector{Int}})
        for sublayer in sublayers
            lowest_qubits = [minimum(qinds[index]) for index in sublayer]
            if !allunique(lowest_qubits) || !(issorted(lowest_qubits) || issorted(lowest_qubits; rev=true))
                throw(ArgumentError("The rotations of a sublayer must be in the order or the reverse order of their lowest qubits. Got $lowest_qubits."))
            end
        end
        return new(symbols, qinds, sublayers)
    end

    @doc """
        RotationLayer(symbols, qinds)
        RotationLayer(PauliRotation, symbols, qinds)

    A parametrized layer of Pauli rotations, one `PauliRotation(symbols, qinds[i])` for every entry of `qinds`.
    For example RotationLayer(:X, 1:4) or RotationLayer([:Z, :Z], staircasetopology(4)).
    The rotations act on one or two qubits each, on any qubits and in any order, and need to commute with each other.
    The parameter of the layer is one angle for all rotations, or a vector with one angle per entry of `qinds`.

    In the Schrödinger picture, the rotations are applied sublayer by sublayer, and within a sublayer in the order of their lowest qubits.
    No two rotations of a sublayer have the same lowest qubit, so an open chain is one sublayer, and a ring or a square lattice two.
    With truncation, the result can therefore differ from that of the rotations applied in the order of `qinds`.
    `torotations` returns the rotations in the order in which they are applied.
    """
    function RotationLayer(symbols, qinds)

        # turn symbols into vectors
        if isa(symbols, Symbol)
            symbols = [symbols]
        else
            symbols = Symbol[symbol for symbol in symbols]
        end

        _rotationlayersymbolcheck(symbols)

        # the qubit indices of every rotation as a vector
        qinds = Vector{Int}[_rotationqinds(rotation_qinds) for rotation_qinds in qinds]

        for rotation_qinds in qinds
            _qinds_check(rotation_qinds)
            _qindslengthcheck(symbols, rotation_qinds)
        end

        _commutationcheck(symbols, qinds)

        return new(symbols, qinds, _sublayers(qinds))
    end
end

RotationLayer(::Type{PauliRotation}, symbols, qinds) = RotationLayer(symbols, qinds)

_rotationqinds(qind::Integer) = [Int(qind)]
_rotationqinds(qinds) = Int[qind for qind in qinds]

function _rotationlayersymbolcheck(symbols)
    if any(s -> s ∉ (:X, :Y, :Z), symbols)
        throw(ArgumentError("Symbols must be `:X`, `:Y`, or `:Z`. Got $symbols."))
    end

    if !(1 <= length(symbols) <= 2)
        throw(ArgumentError("`RotationLayer` is defined for rotations on one or two qubits. Got $(length(symbols)) symbols."))
    end
end

function _commutationcheck(symbols::Vector{Symbol}, qinds::Vector{Vector{Int}})
    rotations_on_qubit = Dict{Int,Vector{Int}}()
    for (index, rotation_qinds) in enumerate(qinds)
        for qind in rotation_qinds
            push!(get!(rotations_on_qubit, qind, Int[]), index)
        end
    end

    for rotations in values(rotations_on_qubit)
        for index1 in rotations, index2 in rotations
            if index1 < index2 && !_generatorscommute(symbols, qinds[index1], qinds[index2])
                throw(ArgumentError(
                    "The rotations of a `RotationLayer` must commute with each other. " *
                    "The rotations on the qubits $(qinds[index1]) and $(qinds[index2]) do not."
                ))
            end
        end
    end
end

# Two rotations commute if their generators differ on an even number of the qubits they share.
function _generatorscommute(symbols::Vector{Symbol}, qinds1::Vector{Int}, qinds2::Vector{Int})
    n_differing = 0
    for (symbol1, qind1) in zip(symbols, qinds1), (symbol2, qind2) in zip(symbols, qinds2)
        if qind1 == qind2 && symbol1 != symbol2
            n_differing += 1
        end
    end
    return iseven(n_differing)
end

# Every rotation goes into the first sublayer in which no other rotation has the same lowest qubit, and every sublayer is then sorted by the lowest qubits.
function _sublayers(qinds::Vector{Vector{Int}})
    sublayers = Vector{Int}[]
    lowest_qubits = Set{Int}[]

    for (index, rotation_qinds) in enumerate(qinds)
        lowest_qubit = minimum(rotation_qinds)
        sublayer_id = findfirst(taken -> lowest_qubit ∉ taken, lowest_qubits)

        if isnothing(sublayer_id)
            push!(sublayers, Int[])
            push!(lowest_qubits, Set{Int}())
            sublayer_id = length(sublayers)
        end

        push!(sublayers[sublayer_id], index)
        push!(lowest_qubits[sublayer_id], lowest_qubit)
    end

    lowest_qubit_of(index) = minimum(qinds[index])
    for sublayer in sublayers
        sort!(sublayer; by=lowest_qubit_of)
    end
    return sublayers
end

function Base.show(io::IO, layer::RotationLayer)
    print(io, "RotationLayer($(layer.symbols), $(length(layer.qinds)) rotations)")
end

"""
    torotations(layer::RotationLayer)
    torotations(layer::RotationLayer, theta)

Returns the rotations of the layer as a vector of `PauliRotation`s, in the order in which the layer applies them in the Schrödinger picture.
With `theta`, the parameter of the layer, also returns the angles of the rotations in that order.
"""
function torotations(layer::RotationLayer)
    return [PauliRotation(layer.symbols, layer.qinds[index]) for sublayer in layer.sublayers for index in sublayer]
end

function torotations(layer::RotationLayer, theta)
    _rotationanglecheck(layer, theta)
    thetas = [_rotationangle(theta, index) for sublayer in layer.sublayers for index in sublayer]
    return torotations(layer), thetas
end

# the angle of the rotation on `layer.qinds[index]`
_rotationangle(theta::Number, index::Int) = theta
_rotationangle(thetas, index::Int) = thetas[index]

_rotationanglecheck(layer::RotationLayer, theta::Number) = nothing

function _rotationanglecheck(layer::RotationLayer, thetas)
    if length(thetas) != length(layer.qinds)
        throw(ArgumentError(
            "The parameter of a `RotationLayer` is one angle or one angle per rotation. " *
            "Got $(length(thetas)) angles for $(length(layer.qinds)) rotations."
        ))
    end
end
