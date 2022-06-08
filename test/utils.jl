@testset "glorot uniform" begin
    @test typeof(glorot_uniform(CPU, Float64, 3, 3)) == Matrix{Float64}

    if CUDA.functional()
        @test typeof(glorot_uniform(GPU, Float32, 3, 3)) ==
            CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
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
