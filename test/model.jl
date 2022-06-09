@testset "params for arb. functions" begin
    @test params(sum) == ()
end

@testset "multilayer perceptron" begin
    # ########
    # # data #
    # ########

    g(x, y) = @. exp(-0.5 * (x^2 + y^2))

    # ####################
    # # training process #
    # ####################

    function train(model, ps, xs, ys)
        loss(xs, ys) = mean(abs2, reshape(model(xs), :) - ys)

        η₀ = 1f-2
        for _ in 1:1000
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

    # #######
    # # CPU #
    # #######

    function train_on_cpu()
        xs = randn(2, 2000)
        ys = g(xs[1, :], xs[2, :])

        model = Model([
            Dense(CPU, Float64, 2, 3, σ=NNlib.relu),
            Dense(CPU, Float64, 3, 1, σ=NNlib.relu),
        ])
        ps = Params(params(model))

        train(model, ps, xs, ys)
    end

    train_on_cpu()

    # #######
    # # GPU #
    # #######

    function train_on_gpu()
        CUDA.functional() || return

        xs = cu(randn(Float32, 2, 2000))
        ys = g(xs[1, :], xs[2, :])

        model = Model([
            Dense(GPU, Float32, 2, 3, σ=NNlib.relu),
            Dense(GPU, Float32, 3, 1, σ=NNlib.relu),
        ])
        ps = Params(params(model))

        train(model, ps, xs, ys)
    end

    train_on_gpu()
end

#=
julia> @benchmark train_on_cpu()
BenchmarkTools.Trial: 2 samples with 1 evaluation.
 Range (min … max):  4.751 s …   4.769 s  ┊ GC (min … max): 17.93% … 17.91%
 Time  (median):     4.760 s              ┊ GC (median):    17.92%
 Time  (mean ± σ):   4.760 s ± 12.338 ms  ┊ GC (mean ± σ):  17.92% ±  0.02%

  █                                                       █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  4.75 s         Histogram: frequency by time        4.77 s <

 Memory estimate: 22.67 GiB, allocs estimate: 315035.

julia> @benchmark train_on_gpu()
BenchmarkTools.Trial: 24 samples with 1 evaluation.
 Range (min … max):  185.635 ms … 225.979 ms  ┊ GC (min … max): 0.00% … 5.99%
 Time  (median):     223.793 ms               ┊ GC (median):    5.81%
 Time  (mean ± σ):   216.277 ms ±  15.425 ms  ┊ GC (mean ± σ):  4.75% ± 2.40%

                                                          ▅ ██
  ██▁▁▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅▅████▅ ▁
  186 ms           Histogram: frequency by time          226 ms <

 Memory estimate: 54.13 MiB, allocs estimate: 1069215.
=#
