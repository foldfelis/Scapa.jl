oftf(x, y) = oftype(float(x), y)

relu(x) = ifelse(x<0, zero(x), x)  # faster than max(zero(x), x), still preserves NaN

function relu(z::Complex)
    re = relu(real(z))
    im = relu(imag(z))
    return Complex(re, im)
end

leakyrelu(x, a=oftf(x, leakyrelu_a)) = ifelse(x>0, float(x), oftf(x, a*x))  # max(a*x, x) is 3x slower

const leakyrelu_a = 0.01  # also used in gradient below

function leakyrelu(z::Complex, a=oftf(real(z), leakyrelu_a))
    re = leakyrelu(real(z), a)
    im = leakyrelu(imag(z), a)
    return Complex(re, im)
end
