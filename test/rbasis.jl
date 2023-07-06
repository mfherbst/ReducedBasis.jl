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

@testset "Different overlap_matrix cases" begin 
    d1 = 5
    d2 = 3
    v1 = [rand(5) for _ in 1:d1]
    v2 = [rand(5) for _ in 1:d2]

    m_same = overlap_matrix(v1, v1)
    @test all(size(m_same) .== (d1, d1))

    m_diff = overlap_matrix(v1, v2)
    @test all(size(m_diff) .== (d1, d2))

    m_func = overlap_matrix(x -> x.^2, v1, v2)
    @test all(m_func .≈ overlap_matrix(v1, map(x -> x.^2, v2)))
end