@testset "Dense" begin
    dense = Dense(2, 3, σ=NNlib.gelu)

    @test size(dense(rand(2))) == (3, )
    @test size(dense(rand(2, 10))) == (3, 10)
    @test size(dense(rand(2, 4, 5, 6, 10))) == (3, 4, 5, 6, 10)
end

@testset "Dense grad" begin
    dense = Dense(2, 3, σ=NNlib.gelu)

    x = rand(Float32, 2)

    gradient(Params([dense.weights, dense.bias])) do
        (sum ∘ dense)(x)
    end
end
