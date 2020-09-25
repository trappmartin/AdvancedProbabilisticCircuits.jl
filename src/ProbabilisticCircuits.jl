module ProbabilisticCircuits

using Reexport

using MacroTools
using StatsFuns

@reexport using Distributions


export @circuit

macro circuit(expr)
    esc(circuit(expr))
end

function circuit(expr)
    return expr
end

# includes
include("types.jl")

end
