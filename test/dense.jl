@testset "Dense" begin
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        dense = Dense(CPU, T, 2, 3, σ=NNlib.gelu)

        @test size(dense(rand(T, 2))) == (3, )
        @test size(dense(rand(T, 2, 10))) == (3, 10)
        @test size(dense(rand(T, 2, 4, 5, 6, 10))) == (3, 4, 5, 6, 10)

        if CUDA.functional()
            dense = Dense(GPU, T, 2, 3, σ=NNlib.gelu)

            @test size(dense(cu(rand(T, 2)))) == (3, )
            @test size(dense(cu(rand(T, 2, 10)))) == (3, 10)
            @test size(dense(cu(rand(T, 2, 4, 5, 6, 10)))) == (3, 4, 5, 6, 10)
        end
    end
end

@testset "Dense grad" begin
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        dense = Dense(CPU, T, 2, 3, σ=NNlib.gelu)

        x = rand(T, 2)

        gradient(Params([dense.weights, dense.bias])) do
            (real ∘ sum ∘ dense)(x)
        end

        if CUDA.functional()
            dense = Dense(GPU, T, 2, 3, σ=NNlib.gelu)

            x = cu(rand(T, 2))

            gradient(Params([dense.weights, dense.bias])) do
                (real ∘ sum ∘ dense)(x)
            end
        end
    end
end
