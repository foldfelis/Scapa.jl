module Scapa

using Random
using ChainRulesCore
using CUDA

include("utils.jl")
include("activations.jl")
include("dense.jl")
include("model.jl")

end
