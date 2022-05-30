if VERSION >= v"1.7"
    rng() = Random.default_rng()
else
    rng() = Random.GLOBAL_RNG
end

nfan() = 1, 1 # fan_in, fan_out

nfan(n) = 1, n # a vector is treated as a n×1 matrix

nfan(out_dim, in_dim) = out_dim, in_dim # dense kernels

# nfan(dims...) = # convolution kernels

function glorot_uniform(rng::AbstractRNG, T::Type{<:Real}, dims::Integer...; gain::Real=1)
  scale = T(gain) * √(T(24) / sum(nfan(dims...)))

  return (rand(rng, T, dims...) .- T(0.5)) .* scale
end

function glorot_uniform(dims::Integer...; kw...)
    return glorot_uniform(rng(), Float32, dims...; kw...)
end

ChainRulesCore.@non_differentiable glorot_uniform(::Any...)

params(::Any) = ()
update!(::Any) = ()
