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

@testset "multilayer perceptron" begin
    # #########
    # # utils #
    # #########

    function get_model_and_params(layers)
        model = identity
        ps = []
        for layer in layers
            model = layer ∘ model
            for param in Scapa.params(layer)
                push!(ps, param)
            end
        end

        return model, Params(ps)
    end

    # ########
    # # data #
    # ########

    g(x, y) = @. exp(-0.5 * (x^2 + y^2))

    xs = randn(Float32, 2, 1000)
    ys = g(xs[1, :], xs[2, :])

    # #######################
    # # model and loss func #
    # #######################

    layers = [
        Dense(2, 3, σ=NNlib.relu),
        Dense(3, 1, σ=NNlib.relu),
    ]

    model, ps = get_model_and_params(layers)

    loss(xs, ys) = mean(abs2, reshape(model(xs), :) - ys)

    # ####################
    # # training process #
    # ####################

    η₀ = 1f-2
    for _ in 1:500
        # ### get gradient ###
        gs = gradient(ps) do
            loss(xs, ys)
        end

        # ### update params ###
        for p in ps
            isnothing(gs[p]) && continue
            p .-= η₀ .* gs[p]
        end
    end

    @test loss(xs, ys) < 1e-1
end
