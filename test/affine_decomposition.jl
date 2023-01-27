using Test, Random
using ReducedBasis: AffineDecomposition, n_terms, compress

@testset "Basic AffineDecomposition funtionality" begin
    basis = fast_assemble()
    Q = rand(1:5)
    N = length(basis.snapshots[1])
    μ = rand(Q)

    function ad_test(terms, ad, adcomp, adraw)    
        @test n_terms(ad) == length(terms)
        @test size(ad(μ)) == size(ad)
        @test n_terms(ad) == n_terms(adcomp)
        @test n_terms(ad) == n_terms(adraw)
        @test size(adcomp(μ)) == size(adcomp)
        @test size(adraw(μ)) == size(adraw)
    end

    @testset "Linear indices" begin
        terms = [rand(N, N) for _ in 1:Q]
        coefficient_map = μ -> rand(Q) .* μ
        ad = AffineDecomposition(terms, coefficient_map)
        adcomp, adraw = compress(ad, basis)
        ad_test(terms, ad, adcomp, adraw)
    end

    @testset "Multi indices" begin
        terms = map(r -> rand(N, N), zeros(Q, Q))
        coefficient_map = μ -> (μ * μ') * rand(Q, Q)
        ad = AffineDecomposition(terms, coefficient_map)
        adcomp, adraw = compress(ad, basis)
        ad_test(terms, ad, adcomp, adraw)
    end
end