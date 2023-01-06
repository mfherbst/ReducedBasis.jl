using Test, LinearAlgebra, SparseArrays
using ReducedBasis

@testset "Offline & online phase: XXZ chain" begin
    # Offline/online parameters
    L        = 6
    H        = xxz_chain(L)
    M        = AffineDecomposition([H.terms[3]], μ -> [2 / L])
    Δ_off    = range(-1.0, 2.5, 40)
    hJ_off   = range(0.0, 3.5, 40)
    grid_off = RegularGrid(Δ_off, hJ_off);
    Δ_on     = range(first(Δ_off), last(Δ_off), 100)
    hJ_on    = range(first(hJ_off), last(hJ_off), 100)
    grid_on  = RegularGrid(Δ_on, hJ_on)

    greedy = Greedy(;
        estimator=Residual(), tol=1e-3, n_truth_max=64, init_from_rb=true, verbose=false
    )
    qrcomp = QRCompress(; tol=1e-10)
    pod    = POD(; n_truth=32, verbose=false)

    # Check values of L/2+1 magnetization plateaus for L=6
    function test_L6_magn_plateaus(basis::RBasis, h::AffineDecomposition, solver_truth)
        fd = FullDiagonalization(;
            tol_degeneracy=solver_truth.tol_degeneracy, n_target=solver_truth.n_target
        )
        m = compress(M, basis)
        m_reduced = m([1])
        magnetization = Matrix{Float64}(undef, size(grid_on))
        for (idx, μ) in pairs(grid_on)
            _, φ_rb = solve(h, basis.metric, μ, fd)
            magnetization[idx] = sum(eachcol(φ_rb)) do φ
                abs(dot(φ, m_reduced, φ)) / size(φ_rb, 2)
            end
        end
        
        @test magnetization[end, 1] ≈ 0.0  atol=1e-6
        @test magnetization[1, end] ≈ 1.0  atol=1e-6
        @test magnetization[75, 35] ≈ 1//3 atol=1e-6
        @test magnetization[75, 60] ≈ 2//3 atol=1e-6
    end

    @testset "Greedy assembly: LOBPCG" begin
        @testset "denegerate" begin
            lobpcg = LOBPCG(; n_target=L+1, tol_degeneracy=1e-4, tol=1e-9)
            basis, h, info = assemble(H, grid_off, greedy, lobpcg, qrcomp; callback=x -> x)
            @test multiplicity(basis)[1] > 1
            test_L6_magn_plateaus(basis, h, lobpcg)
        end

        @testset "non-denegerate" begin
            lobpcg = LOBPCG(; n_target=1, tol_degeneracy=0.0, tol=1e-9)
            basis, h, info = assemble(H, grid_off, greedy, lobpcg, qrcomp; callback=x -> x)
            test_L6_magn_plateaus(basis, h, lobpcg)
        end
    end

    @testset "Greedy assembly: FullDiagonalization" begin
        @testset "degenerate" begin
            fulldiag = FullDiagonalization(; n_target=L+1, tol_degeneracy=1e-4)
            basis, h, info = assemble(H, grid_off, greedy, fulldiag, qrcomp; callback=x -> x)
            @test multiplicity(basis)[1] > 1
            test_L6_magn_plateaus(basis, h, fulldiag)
        end

        @testset "non-degenerate" begin
            fulldiag = FullDiagonalization(; n_target=1, tol_degeneracy=0.0)
            basis, h, info = assemble(H, grid_off, greedy, fulldiag, qrcomp; callback=x -> x)
            test_L6_magn_plateaus(basis, h, fulldiag)
        end
    end

    @testset "Proper orthogonal decomposition: LOBPCG" begin
        @testset "degenerate" begin
            lobpcg = LOBPCG(; n_target=L+1, tol_degeneracy=1e-4, tol=1e-9)
            basis, info = assemble(H, grid_off, pod, lobpcg)
            h_cache = HamiltonianCache(H, basis)
            @test multiplicity(basis)[1] > 1
            test_L6_magn_plateaus(basis, h_cache.h, lobpcg)
        end

        @testset "non-degenerate" begin
            lobpcg = LOBPCG(; n_target=1, tol_degeneracy=0.0, tol=1e-9)
            basis, info = assemble(H, grid_off, pod, lobpcg)
            h_cache = HamiltonianCache(H, basis)
            test_L6_magn_plateaus(basis, h_cache.h, lobpcg)
        end
    end
end