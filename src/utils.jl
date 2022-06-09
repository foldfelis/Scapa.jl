export
    AbstractDevice, CPU, GPU,
    init_on,
    nfan,
    glorot_uniform

abstract type AbstractDevice end

struct CPU<:AbstractDevice end

struct GPU<:AbstractDevice end

Array(::Type{CPU}) = Base.Array

Array(::Type{GPU}) = CuArray

"""
    init_on()

Can be overwritten to switch default initialization on which device.

# Examples

```julia
julia> typeof(glorot_uniform(3, 3))
Matrix{Float32} (alias for Array{Float32, 2})

julia> Scapa.init_on() = GPU

julia> typeof(glorot_uniform(3, 3))
CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}

julia> typeof(glorot_uniform(CPU, Float64, 3, 3))
Matrix{Float64} (alias for Array{Float64, 2})
```
"""
init_on() = CPU

if VERSION >= v"1.7"
    rng() = Random.default_rng()
else
    rng() = Random.GLOBAL_RNG
end

rng(::Type{CPU}) = rng()

rng(::Type{GPU}) = CUDA.default_rng()

nfan() = 1, 1 # fan_in, fan_out

nfan(n) = 1, n # a vector is treated as a n×1 matrix

nfan(out_dim, in_dim) = out_dim, in_dim # dense kernels

# nfan(dims...) = # convolution kernels

extend_imag(T::Type{<:Complex}, x::Real) = T(x + im*x)
extend_imag(T::Type{<:Real}, x::Real) = T(x)

function glorot_uniform(
    device::Type{<:AbstractDevice}, T::Type{<:Number},
    dims::Integer...;
    gain::Real=1
)
    out = Array(device){T}(undef, dims...)

    scale = T(gain) * √(T(24) / sum(nfan(dims...)))
    shift = extend_imag(T, 0.5)

    return (rand!(rng(device), out) .- shift) .* scale
end

function glorot_uniform(
    T::Type{<:Number},
    dims::Integer...;
    gain::Real=1
)
    return glorot_uniform(init_on(), T, dims...; gain=gain)
end

function glorot_uniform(dims::Integer...; gain::Real=1)
    return glorot_uniform(init_on(), Float32, dims...; gain=gain)
end

ChainRulesCore.@non_differentiable glorot_uniform(::Any...)

params(::Any) = ()
