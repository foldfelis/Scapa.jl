@testset "Dense" begin
    dense = Dense(CPU, Float64, 2, 3, σ=NNlib.gelu)

    @test size(dense(rand(2))) == (3, )
    @test size(dense(rand(2, 10))) == (3, 10)
    @test size(dense(rand(2, 4, 5, 6, 10))) == (3, 4, 5, 6, 10)

    if CUDA.functional()
        dense = Dense(GPU, Float32, 2, 3, σ=NNlib.gelu)

        @test size(dense(cu(rand(Float32, 2)))) == (3, )
        @test size(dense(cu(rand(Float32, 2, 10)))) == (3, 10)
        @test size(dense(cu(rand(Float32, 2, 4, 5, 6, 10)))) == (3, 4, 5, 6, 10)
    end
end

@testset "Dense grad" begin
    dense = Dense(CPU, Float64, 2, 3, σ=NNlib.gelu)

    x = rand(2)

    gradient(Params([dense.weights, dense.bias])) do
        (sum ∘ dense)(x)
    end

    if CUDA.functional()
        dense = Dense(GPU, Float32, 2, 3, σ=NNlib.gelu)

        x = cu(rand(Float32, 2))

        gradient(Params([dense.weights, dense.bias])) do
            (sum ∘ dense)(x)
        end
    end
end
