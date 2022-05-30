using Scapa
using Test

using NNlib
using Zygote
using ChainRulesCore
using Statistics

@testset "Scapa.jl" begin
    include("utils.jl")
    include("dense.jl")
end
