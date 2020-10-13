using Test
using AdvancedProbabilisticCircuits
using StatsFuns

const log2πhalf = log2π/2

@testset "nodes" begin

    @testset "constructors" begin
        @test Normal(1) isa AdvancedProbabilisticCircuits.AbstractLeaf
        @test Indicator(1) isa AdvancedProbabilisticCircuits.AbstractLeaf
        @test TruncatedNormal(1) isa AdvancedProbabilisticCircuits.AbstractLeaf
        @test Sum(Normal, 1, 2) isa AdvancedProbabilisticCircuits.Node
        @test Sum(Normal(1), Normal(1)) isa AdvancedProbabilisticCircuits.Node
        @test Prod(Normal, [1,2]) isa AdvancedProbabilisticCircuits.Node
        @test Prod(Normal(1), Normal(2)) isa AdvancedProbabilisticCircuits.Node
    end

    @testset "scope functions" begin
        @test scope(Normal(1)) == 1
        @test scope(Sum(Normal(1), Normal(1))) == 1
        @test scope(Prod(Normal(1), Normal(2))) == [1,2]
    end

    @testset "support functions" begin
        @test support(Normal(1)) == (1 => RealInterval(-Inf, Inf))
        @test support(Indicator(1)) == (1 => [1])
        @test support(Sum(Indicator(1, v = 1), Indicator(1, v = 2))) == (1 => [1,2])
    end

    @testset "logpdf functions" begin
        x = 0
        @test logpdf(Normal(1), x) ≈ -log2πhalf
        @test logpdf(Indicator(1), 1) ≈ 0
        @test logpdf(Indicator(1), 0) ≈ -Inf
        @test logpdf(TruncatedNormal(1), x) ≈ -log2πhalf
        @test logpdf(TruncatedNormal(1, min=1), x) ≈ -Inf
        @test logpdf(TruncatedNormal(1, max=-1), x) ≈ -Inf
        s = Sum(Normal(1), Normal(1))
        @test logpdf(s, x) ≈ logsumexp(-log2π/2 .+ params(s))
        @test logpdf(Prod(Normal(1), Normal(2)), [x,x]) ≈ -log2π
    end

end

@testset "properties" begin
    
end

@testset "util" begin
    
end

@testset "optimize" begin
    
end