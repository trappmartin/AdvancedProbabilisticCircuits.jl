module ProbabilisticCircuits

using Reexport
using MacroTools
using StatsFuns
using Crayons
using Crayons.Box

import Base.show
import Distributions.support

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
