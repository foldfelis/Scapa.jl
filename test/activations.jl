@testset "activations" begin
    z = 2.0 - im*5.
    @test Scapa.relu(z) == Scapa.relu(2.0) + im*Scapa.relu(-5.)
    @test Scapa.leakyrelu(z) == Scapa.leakyrelu(2.0) + im*Scapa.leakyrelu(-5.)
    @test Scapa.gelu(z) == Scapa.gelu(2.0) + im*Scapa.gelu(-5.)
    @test Scapa.σ(z) == Scapa.σ(2.0) + im*Scapa.σ(-5.)

    zs = rand(ComplexF32, 100)
    @test Scapa.relu.(zs) isa Vector{ComplexF32}
    @test Scapa.leakyrelu.(zs) isa Vector{ComplexF32}
    @test Scapa.gelu.(zs) isa Vector{ComplexF32}
    @test Scapa.σ.(zs) isa Vector{ComplexF32}
end
