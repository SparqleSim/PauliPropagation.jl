###
##
# Layers of gates that commute with each other, applied together. By default a layer applies its gates one after the
# other; a basis can apply the layers of its gates in one pass over the sum instead.
##
###

"""
A type for a layer of gates that commute with each other.
"""
struct GateLayer{G} <: ParametrizedGate
    gates::Vector{G}
    guaranteed_commutes::Bool

    @doc """
        GateLayer(gates; guaranteed_commutes=false)

    A layer of `gates` that commute with each other.
    By default, the gates must act on different qubits.
    With `guaranteed_commutes=true`, gates may share qubits, and it is up to the caller that they commute.
    In a circuit, a layer counts as one parametrized gate.
    Its parameter is a vector with one entry per parametrized gate of the layer, in the order of `gates`,
    so a circuit of two layers takes parameters like [thetas1, thetas2].
    In the Schrödinger picture, the gates are applied in the order of `gates`, with truncation after each.
    """
    function GateLayer(gates; guaranteed_commutes::Bool=false)
        gates = [gate for gate in gates]
        if !guaranteed_commutes
            _qindsoverlapcheck(gates)
        end
        return new{eltype(gates)}(gates, guaranteed_commutes)
    end
end

function _qindsoverlapcheck(gates)
    gate_on_qubit = Dict{Int,Int}()
    for (index, gate) in enumerate(gates)
        for qind in qinds(gate)
            other_index = get(gate_on_qubit, qind, index)
            if other_index != index
                throw(ArgumentError(
                    "The gates $other_index and $index of the layer both act on the qubit $qind. " *
                    "Pass `guaranteed_commutes=true` if the gates commute nonetheless."
                ))
            end
            gate_on_qubit[qind] = index
        end
    end
end

# A layer prints how many of its gates are of each type, by the name of the type without its parameters, in the order in
# which the types first appear, as GateLayer(36 PauliRotation, 2 FrozenGate). After the first few types, the remaining
# gates are only counted.
function Base.show(io::IO, layer::GateLayer)
    names = String[]
    counts = Int[]
    for gate in layer.gates
        name = string(nameof(typeof(gate)))
        index = findfirst(==(name), names)
        if isnothing(index)
            push!(names, name)
            push!(counts, 1)
        else
            counts[index] += 1
        end
    end

    n_shown = min(length(names), _MAX_SHOWN_NAMES)
    parts = ["$(counts[index]) $(names[index])" for index in 1:n_shown]
    n_more = sum(counts[n_shown+1:end]; init=0)
    if n_more > 0
        push!(parts, "$n_more more")
    end
    print(io, "GateLayer(", join(parts, ", "), ")")
end

# the number of types that a layer prints at most
const _MAX_SHOWN_NAMES = 4

"""
    countparameters(layer::GateLayer)

Returns the number of parametrized gates in the layer, which is the length of its parameter.
"""
countparameters(layer::GateLayer) = countparameters(layer.gates)

# the qubits that the gates of the layer act on
function qinds(layer::GateLayer)
    layer_qinds = Int[]
    for gate in layer.gates
        append!(layer_qinds, qinds(gate))
    end
    return unique(layer_qinds)
end

"""
    applymergetruncate!(layer::GateLayer, prop_cache::AbstractPropagationCache, params; kwargs...)

Applies the gates of the layer one after the other, each with `applymergetruncate!`.
"""
function applymergetruncate!(layer::GateLayer, prop_cache::AbstractPropagationCache, params; kwargs...)
    _applygatesonebyone!(layer, prop_cache, params; kwargs...)
    return
end

function _applygatesonebyone!(layer::GateLayer, prop_cache::AbstractPropagationCache, params; kwargs...)
    function applygate!(gate, args...)
        applymergetruncate!(gate, prop_cache, args...; kwargs...)
        return
    end
    _foreachgate(applygate!, layer, params)
    return prop_cache
end

# Calls `f(gate, param)` for every parametrized gate of the layer, with its own parameter, and `f(gate)` for every
# other gate, in the order of the layer.
function _foreachgate(f::F, layer::GateLayer, params) where {F}
    _checklayerparameters(layer, params)
    param_index = 0
    for gate in layer.gates
        if gate isa ParametrizedGate
            param_index += 1
            f(gate, _gateparameter(params, param_index))
        else
            f(gate)
        end
    end
    return
end

# the parameter of the parametrized gate number `param_index` of a layer
_gateparameter(param::Number, param_index::Int) = param
_gateparameter(params, param_index::Int) = params[param_index]

function _checklayerparameters(layer::GateLayer, params)
    n_params = countparameters(layer)
    is_valid = if params isa Number
        n_params == 1
    else
        length(params) == n_params
    end
    if !is_valid
        # a lazy message, so that printing the parameters is not compiled before it is needed
        throw(ArgumentError(LazyString(
            "The parameter of a `GateLayer` is a vector with one entry per parametrized gate. ",
            "Got ", n_params, " gates but parameter ", params, "."
        )))
    end
end
