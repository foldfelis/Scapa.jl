oftf(x, y) = oftype(float(x), y)

relu(x) = ifelse(x<0, zero(x), x)  # faster than max(zero(x), x), still preserves NaN

relu(z::Complex) = Complex(relu(real(z)), relu(imag(z)))

leakyrelu(x, a=oftf(x, leakyrelu_a)) = ifelse(x>0, float(x), oftf(x, a*x))  # max(a*x, x) is 3x slower

const leakyrelu_a = 0.01  # also used in gradient below

function leakyrelu(z::Complex, a=oftf(real(z), leakyrelu_a))
    re = leakyrelu(real(z), a)
    im = leakyrelu(imag(z), a)
    return Complex(re, im)
end

function gelu(x)
    α = oftf(x, 0.044715)
    λλ = oftf(x, gelu_2λ)
    return x * sigmoid(λλ * x * muladd(x^2, α, one(x)))
end

gelu(z::Complex) = Complex(gelu(real(z)), gelu(imag(z)))

const gelu_λ = √(2 / π)
const gelu_2λ = √(8 / π)

function σ(x)
    t = exp(-abs(x))
    return ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end

σ(z::Complex) = Complex(σ(real(z)), σ(imag(z)))

const sigmoid = σ
