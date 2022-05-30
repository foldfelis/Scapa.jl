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

@testset "Dense grad" begin
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

        return model, ps
    end

    # ########
    # # data #
    # ########

    g(x, y) = exp(-0.5 * (x^2 + y^2))

    xs = randn(Float32, 2, 1000)
    ys = g.(xs[1, :], xs[2, :])

    # #######################
    # # model and loss func #
    # #######################

    layers = [
        Dense(2, 3, σ=NNlib.gelu),
        Dense(3, 1, σ=NNlib.gelu),
    ]

    model, ps = get_model_and_params(layers)

    loss(xs, ys) = mean(abs2, reshape(model(xs), :) - ys)

    # ####################
    # # training process #
    # ####################

    η₀ = 1f-2
    for _ in 1:500
        # ### get gradient ###

        gs = gradient(Params(ps)) do
            loss(xs, ys)
        end

        ps .-= gs .* η₀

        # ### update params ###

        i = 1
        for layer in layers
            Scapa.update!(layer, ps[i:(i+Scapa.nparams(layer)-1)])
            i += Scapa.nparams(layer)
        end

        model, ps = get_model_and_params(layers)
    end

    @test loss(xs, ys) < 1e-1
end
