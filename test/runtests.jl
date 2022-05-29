using Scapa
using Test

using NNlib
using Enzyme
using ChainRulesCore

@testset "Scapa.jl" begin
    include("utils.jl")
    include("dense.jl")
end
