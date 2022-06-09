@testset "constructers" begin
    @test Scapa.Array(CPU) == Base.Array
    @test Scapa.rng(CPU) == Scapa.rng()

    if CUDA.functional()
        @test Scapa.Array(GPU) == CuArray
        @test Scapa.rng(GPU) == CUDA.default_rng()
    end
end

@testset "glorot uniform" begin
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        @test glorot_uniform(CPU, T, 3, 3) isa Matrix{T}
        @test glorot_uniform(T, 3, 3) isa Matrix{T}
        @test glorot_uniform(3, 3) isa Matrix{Float32}

        if CUDA.functional()
            @test glorot_uniform(GPU, T, 3, 3) isa CuArray{T, 2}
        end
    end
end

@testset "init_on" begin
    if CUDA.functional()
        @test glorot_uniform(Float32, 3, 3) isa Matrix{Float32}

        Scapa.init_on() = GPU
        @test glorot_uniform(Float32, 3, 3) isa CuArray{Float32, 2}
        @test glorot_uniform(CPU, Float32, 3, 3) isa Matrix{Float32}

        Scapa.init_on() = CPU
        @test glorot_uniform(Float32, 3, 3) isa Matrix{Float32}
    end
end

#=
julia> @benchmark glorot_uniform(CPU, Float32, 300, 500)
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  48.725 μs … 621.587 μs  ┊ GC (min … max):  0.00% … 57.64%
 Time  (median):     60.307 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   72.021 μs ±  60.287 μs  ┊ GC (mean ± σ):  12.30% ± 12.48%

  ▂█
  ██▇█▃▃▂▂▂▂▁▂▂▁▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▂▂▂▂▂▂▂ ▂
  48.7 μs         Histogram: frequency by time          450 μs <

 Memory estimate: 1.14 MiB, allocs estimate: 11.

julia> @benchmark glorot_uniform(GPU, Float32, 300, 500)
BenchmarkTools.Trial: 10000 samples with 6 evaluations.
 Range (min … max):  5.204 μs …  3.180 ms  ┊ GC (min … max): 0.00% … 32.23%
 Time  (median):     5.448 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   7.086 μs ± 70.371 μs  ┊ GC (mean ± σ):  7.02% ±  0.71%

          ▄▆▇██▇▆▅▃▂▁
  ▁▁▁▂▂▄▅████████████▇▇▆▅▆▅▅▅▄▄▄▃▄▄▃▃▂▃▃▂▂▂▂▂▂▁▂▁▂▁▁▁▁▁▁▁▁▁▁ ▃
  5.2 μs         Histogram: frequency by time        6.13 μs <

 Memory estimate: 3.84 KiB, allocs estimate: 72.
=#
