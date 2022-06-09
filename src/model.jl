export
    Model, params

params(::Any) = ()

struct Model{L}
    layers::L
end

function (m::Model)(x)
    for layer in m.layers
        x = layer(x)
    end

    return x
end

function params(m::Model)
    ps = []
    for layer in m.layers
        for param in params(layer)
            push!(ps, param)
        end
    end

    return ps
end
