using Scapa
using Test

using NNlib
using Zygote
using Statistics
using CUDA

@testset "Scapa.jl" begin
    include("utils.jl")
    include("dense.jl")
    include("model.jl")
end
