using Test, LinearAlgebra, Random
using ReducedBasis

@testset "Basic RBasis functionality" begin
    n_truth = 16
    basis   = fast_assemble(n_truth)
    d_basis = dimension(basis)
    nt      = n_truthsolve(basis)
    mult    = multiplicity(basis)

    @testset "Property functions" begin
        @test length(mult) == nt
        @test all(mult .≥ 0)
    end

    @testset "Correct overlap computation" begin
        @test issymmetric(basis.snapshot_overlaps)
        @test size(basis.snapshot_overlaps) == (d_basis, d_basis)
    end

    @testset "Basis extension" begin
        m = rand(1:5)
        Ψ = [rand(length(basis.snapshots[1])) for _ in 1:m]
        μ = rand(length(basis.parameters[1]))
        @testset "NoCompress" begin
            ext = extend(basis, Ψ, μ, NoCompress())
            @test dimension(ext.basis) == d_basis + m
            @test n_truthsolve(ext.basis) == nt + 1
            @test multiplicity(ext.basis)[end] == m
            @test norm(ext.basis.metric - I) ≥ 0.1
        end

        @testset "QRCompress" begin
            ext = extend(basis, Ψ, μ, QRCompress())
            if !isnothing(ext.keep)
                @test dimension(ext.basis) ≤ d_basis + m
                @test n_truthsolve(ext.basis) == nt + 1
                @test multiplicity(ext.basis)[end] ≤ m
                @test norm(ext.basis.metric - I) ≤ 1e-9
            end
        end
    end
end