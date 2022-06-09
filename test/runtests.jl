using Scapa
using Test

using Zygote
using Statistics
using CUDA

@testset "Scapa.jl" begin
    include("utils.jl")
    include("activations.jl")
    include("dense.jl")
    include("model.jl")
end
