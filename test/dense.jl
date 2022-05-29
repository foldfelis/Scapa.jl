@testset "Dense" begin
    dense = Dense(2, 3, σ=NNlib.gelu)

    @test size(dense(rand(2))) == (3, )
    @test size(dense(rand(2, 10))) == (3, 10)
    @test size(dense(rand(2, 4, 5, 6, 10))) == (3, 4, 5, 6, 10)
end

@testset "Dense grad" begin
    dense = Dense(2, 3, σ=NNlib.gelu)

    x = rand(Float32, 2)

    ȳ, ∇dense = rrule(dense, x)
    @test_broken ∇dense(ones(size(ȳ)))
end

@testset "grad" begin
    in_dim = 2
    out_dim = 3

    w = Scapa.glorot_uniform(out_dim, in_dim)
    ∂l_∂w = fill!(similar(w, size(w)), 0)
    b = fill!(similar(w, out_dim), 0)
    ∂l_∂b = fill!(similar(b, size(b)), 0)
    σ = NNlib.gelu

    function layer(x::AbstractVecOrMat, w, b, σ)
        return σ.(w * x .+ b)
    end

    x = rand(Float32, 2)

    @test_broken autodiff(layer, Const, x, Duplicated(w, ∂l_∂w), Duplicated(b, ∂l_∂b), σ) === ()
end
