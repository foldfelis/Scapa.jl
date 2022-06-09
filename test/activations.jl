@testset "activations" begin
    z = 2.0 - im*5.
    @test Scapa.relu(z) == 2.0 + im*0.
    @test Scapa.leakyrelu(z) == 2.0 - im*0.05

    zs = rand(ComplexF32, 100)
    @test Scapa.relu.(zs) isa Vector{ComplexF32}
    @test Scapa.leakyrelu.(zs) isa Vector{ComplexF32}
end
