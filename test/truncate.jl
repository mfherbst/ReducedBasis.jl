using Test
using ReducedBasis

@testset "Truncation" begin
    # Basis settings and an observable
    L        = 6
    H        = xxz_chain(L)
    M        = AffineDecomposition([H.terms[3]], μ -> [2 / L])
    grid_off = RegularGrid(range(-1.0, 2.5; length=40), range(0.0, 3.5; length=40))
    greedy   = Greedy(; estimator=Residual(), tol=1e-3, n_truth_max=32,
                    init_from_rb=true, verbose=false)
    lobpcg   = LOBPCG(; n_target=1, tol_degeneracy=1e-4, tol=1e-9)
    n_trunc = 4
    
    function truncate_all(n_truth, info, mraw)
        basis_trunc = truncate(info.basis, n_truth)
        h_cache_trunc = truncate(info.h_cache, basis_trunc)
        m_trunc = truncate(mraw, basis_trunc)
        (; basis_trunc, h_cache_trunc, m_trunc)
    end

    function test_truncated(basis, basis_tr, h_cache_tr, m_tr)
        d_trunc = sum(multiplicity(basis)[1:n_trunc])
        @test dimension(basis_tr) == d_trunc
        @test length(basis_tr.snapshots) == length(basis_tr.parameters)
        @test all(length.(h_cache_tr.HΨ) .== d_trunc)
        @test all(size(term) == (d_trunc, d_trunc) for term in h_cache_tr.ΨHΨ)
        @test all(size(term) == (d_trunc, d_trunc) for term in h_cache_tr.ΨHHΨ)
        @test size(h_cache_tr.h) == (d_trunc, d_trunc)
        @test size(h_cache_tr.h²) == (d_trunc, d_trunc)
        @test size(m_tr) == (d_trunc, d_trunc)
    end

    @testset "trivial vectors=I using QRCompress" begin
        qrcomp = QRCompress(; tol=1e-10)
        info = assemble(H, grid_off, greedy, lobpcg, qrcomp; callback=x -> x)
        m, _ = compress(M, info.basis)
        basis_tr, h_cache_tr, m_tr = truncate_all(n_trunc, info, m)
        test_truncated(info.basis, basis_tr, h_cache_tr, m_tr)
    end

    @testset "vectors=V using EigenDecomposition" begin
        edcomp = EigenDecomposition(; cutoff=1e-10)
        info = assemble(H, grid_off, greedy, lobpcg, edcomp; callback=x -> x)
        m, mraw = compress(M, info.basis)
        basis_tr, h_cache_tr, m_tr = truncate_all(n_trunc, info, mraw)
        test_truncated(info.basis, basis_tr, h_cache_tr, m_tr)
    end
end