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

function Base.show(io::IO, l::Dense)
    print(io, "Dense(", size(l.weights, 2), " => ", size(l.weights, 1))
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == false && print(io, "; bias=false")
    print(io, ")")
end

function dense_forward(x::AbstractVecOrMat, weights, bias, σ)
    return σ.(weights * x .+ bias)
end

function (l::Dense)(x::AbstractVecOrMat)
    return dense_forward(x, l.weights, l.bias, l.σ)
end

function ChainRulesCore.rrule(l::Dense, x::AbstractVecOrMat)
    ∂l_∂w = fill!(similar(l.weights, size(l.weights)), 0)
    ∂l_∂b = fill!(similar(l.bias, size(l.bias)), 0)

    function dense_pullback(ȳ)
        autodiff(
            dense_forward, Const,
            x, Duplicated(l.weights, ∂l_∂w), Duplicated(l.bias, ∂l_∂b), l.σ
        )

        return ∂l_∂w, ∂l_∂b
    end

    return l(x), dense_pullback
end

function (l::Dense)(x::AbstractArray)
    return reshape(l(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)
end
