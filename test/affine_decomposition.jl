using Test, Random
using ReducedBasis: AffineDecomposition, n_terms, compress

@testset "Basic AffineDecomposition funtionality" begin
    basis = fast_assemble()
    Q = rand(1:5)
    N = length(basis.snapshots[1])
    μ = rand(Q)

    function ad_test(terms, ad, adcomp, adraw)    
        @test length(ad) == length(terms[1])
        @test length(adcomp) != length(terms[1])
        @test length(adraw) != length(terms[1])
        @test size(ad(μ)) == size(ad)
        @test size(adcomp(μ)) == size(adcomp)
        @test size(adraw(μ)) == size(adraw)
        @test n_terms(ad) == length(terms)
        @test n_terms(ad) == n_terms(adcomp)
        @test n_terms(ad) == n_terms(adraw)
    end

    @testset "Linear indices" begin
        terms = [rand(N, N) for _ in 1:Q]
        coefficients = μ -> rand(Q) .* μ
        ad = AffineDecomposition(terms, coefficients)
        adcomp, adraw = compress(ad, basis)
        ad_test(terms, ad, adcomp, adraw)
    end

    @testset "Multi indices" begin
        terms = map(r -> rand(N, N), zeros(Q, Q))
        coefficients = μ -> (μ * μ') * rand(Q, Q)
        ad = AffineDecomposition(terms, coefficients)
        adcomp, adraw = compress(ad, basis)
        ad_test(terms, ad, adcomp, adraw)
    end

    @testset "Symmetric compression" begin
        d₁, d₂ = 3, 5
        @testset "d₁ < d₂" begin
            terms = map(r -> rand(N, N), zeros(d₁, d₂))
            coefficients = μ -> rand(d₁, d₂)
            ad = AffineDecomposition(terms, coefficients)
            adcomp, adraw = compress(ad, basis; symmetric_terms=true)
            ad_test(terms, ad, adcomp, adraw)
            for idx in findall(x -> x.I[1] > x.I[2], CartesianIndices(terms))
                @test adcomp.terms[idx] == adcomp.terms[last(idx.I), first(idx.I)]
                @test adraw.terms[idx]  == adraw.terms[last(idx.I), first(idx.I)]
            end  # Test equivalence of transposed compressed elements
        end
        @testset "d₁ > d₂" begin
            terms = map(r -> rand(N, N), zeros(d₂, d₁))
            coefficients = μ -> rand(d₂, d₁)
            ad = AffineDecomposition(terms, coefficients)
            adcomp, adraw = compress(ad, basis; symmetric_terms=true)
            ad_test(terms, ad, adcomp, adraw)
            for idx in findall(x -> x.I[1] < x.I[2], CartesianIndices(terms))
                @test adcomp.terms[idx] == adcomp.terms[last(idx.I), first(idx.I)]
                @test adraw.terms[idx]  == adraw.terms[last(idx.I), first(idx.I)]
            end  # Test equivalence of transposed compressed elements
        end
    end
end