export
    AbstractDevice, CPU, GPU,
    nfan,
    glorot_uniform

abstract type AbstractDevice end

struct CPU<:AbstractDevice end

struct GPU<:AbstractDevice end

Array(::Type{CPU}) = Base.Array

Array(::Type{GPU}) = CuArray

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

function glorot_uniform(
    device::Type{<:AbstractDevice}, T::Type{<:Real},
    dims::Integer...;
    gain::Real=1
)
    out = Array(device){T}(undef, dims...)

    scale = T(gain) * √(T(24) / sum(nfan(dims...)))

    return (rand!(rng(device), out) .- T(0.5)) .* scale
end

ChainRulesCore.@non_differentiable glorot_uniform(::Any...)

params(::Any) = ()
