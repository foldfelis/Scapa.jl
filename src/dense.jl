export Dense

struct Dense{F, W, B}
    weights::W
    bias::B
    σ::F
end

function Dense(
    in_dim::Integer, out_dim::Integer;
    bias::Bool=true, σ=identity, init=glorot_uniform
)
    w = init(out_dim, in_dim)
    b = bias ? fill!(similar(w, out_dim), 0) : false

    return Dense(w, b, σ)
end

params(l::Dense) = l.weights, l.bias

function Base.show(io::IO, l::Dense)
    print(io, "Dense(", size(l.weights, 2), " => ", size(l.weights, 1))
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == false && print(io, "; bias=false")
    print(io, ")")
end

function (l::Dense)(x::AbstractVecOrMat)
    return l.σ.(l.weights * x .+ l.bias)
end

function (l::Dense)(x::AbstractArray)
    return reshape(l(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)
end
