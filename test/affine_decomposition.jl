using Test, Random
using ReducedBasis: AffineDecomposition, n_terms, compress

@testset "Basic AffineDecomposition funtionality" begin
    basis = fast_assemble()
    Q = rand(1:5)
    N = length(basis.snapshots[1])
    μ = rand(Q)

    @testset "Linear indices" begin
        terms = [rand(N, N) for _ in 1:Q]
        coefficient_map = μ -> rand(Q) .* μ
        ad = AffineDecomposition(terms, coefficient_map)
        ad_comp = compress(ad, basis)

        @test n_terms(ad) == length(terms)
        @test size(ad(μ)) == size(ad)
        @test n_terms(ad_comp) == n_terms(ad_comp)
        @test size(ad_comp(μ)) == size(ad_comp)
    end

    @testset "Multi indices" begin
        terms = map(r -> rand(N, N), zeros(Q, Q))
        coefficient_map = μ -> (μ * μ') * rand(Q, Q)
        ad = AffineDecomposition(terms, coefficient_map)
        ad_comp = compress(ad, basis)

        @test n_terms(ad) == length(terms)
        @test size(ad(μ)) == size(ad)
        @test n_terms(ad_comp) == n_terms(ad_comp)
        @test size(ad_comp(μ)) == size(ad_comp)
    end
    
end