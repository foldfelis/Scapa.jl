module Scapa

using Random
using ChainRulesCore
using CUDA

include("utils.jl")
include("dense.jl")
include("model.jl")

end
