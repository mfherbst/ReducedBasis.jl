using Test, LinearAlgebra, SparseArrays
using ReducedBasis

@testset "Offline and online phase for XXZ chain" begin
    # Convert local-site to many-body operator
    function to_global(L::Int, op::M, i::Int) where {M<:AbstractMatrix}
        d = size(op, 1)
        if i == 1
            return kron(op, M(I, d^(L - 1), d^(L - 1)))
        elseif i == L
            return kron(M(I, d^(L - 1), d^(L - 1)), op)
        else
            return kron(kron(M(I, d^(i - 1), d^(i - 1)), op), M(I, d^(L - i), d^(L - i)))
        end
    end

    # Construct XXZ Hamiltonian matrix
    function xxz_chain(N)
        σx = sparse([0.0 1.0; 1.0 0.0])
        σy = sparse([0.0 -im; im 0.0])
        σz = sparse([1.0 0.0; 0.0 -1.0])

        H1 = 0.25 * sum([to_global(N, σx, i) * to_global(N, σx, i + 1) +
                         to_global(N, σy, i) * to_global(N, σy, i + 1) for i in 1:(N-1)])
        H2 = 0.25 * sum([to_global(N, σz, i) * to_global(N, σz, i + 1) for i in 1:(N-1)])
        H3 = 0.5  * sum([to_global(N, σz, i) for i in 1:N])
        coefficient_map = μ -> [1.0, μ[1], -μ[2]]

        AffineDecomposition([H1, H2, H3], coefficient_map)
    end

    # Offline/online parameters
    L        = 6
    H        = xxz_chain(L)
    M        = AffineDecomposition([H.terms[3]], μ -> [2 / L])
    Δ_off    = range(-1.0, 2.5, 40)
    hJ_off   = range(0.0, 3.5, 40)
    grid_off = RegularGrid(Δ_off, hJ_off);
    Δ_on     = range(first(Δ_off), last(Δ_off), 150)
    hJ_on    = range(first(hJ_off), last(hJ_off), 150)
    grid_on  = RegularGrid(Δ_on, hJ_on)

    greedy = Greedy(; estimator=Residual(), tol=1e-3, n_truth_max=64, init_from_rb=true)
    qrcomp = QRCompress(; tol=1e-10)
    pod    = POD(; n_truth=32, verbose=false)

    # Check values of L/2+1 magnetization plateaus for L=6
    function test_L6_magn_plateaus(basis::RBasis, h::AffineDecomposition)
        m = compress(M, basis)
        m_reduced = m([1])
        magnetization = Matrix{Float64}(undef, size(grid_on))
        for (idx, μ) in pairs(grid_on)
            _, φ_rb = solve(h, basis.metric, μ, fulldiag)
            magnetization[idx] = sum(eachcol(φ_rb)) do φ
                abs(dot(φ, m_reduced, φ)) / size(φ_rb, 2)
            end
        end
        
        @test magnetization[end, 1]  ≈ 0.0 atol=1e-6
        @test magnetization[1, end]  ≈ 1.0 atol=1e-6
        @test magnetization[110, 50] ≈ 1//3 atol=1e-6
        @test magnetization[110, 90] ≈ 2//3 atol=1e-6
    end

    @testset "Greedy assembly using LOBPCG" begin
        @testset "denegerate" begin
            lobpcg = LOBPCG(; n_target=L+1, tol_degeneracy=1e-4, tol=1e-9)
            basis, h, info = assemble(H, grid_off, greedy, lobpcg, qrcomp; callback=x -> x)
            test_L6_magn_plateaus(basis, h)
        end

        @testset "non-denegerate" begin
            lobpcg = LOBPCG(; n_target=1, tol_degeneracy=0.0, tol=1e-9)
            basis, h, info = assemble(H, grid_off, greedy, lobpcg, qrcomp; callback=x -> x)
            test_L6_magn_plateaus(basis, h)
        end
    end

    @testset "Greedy assembly using FullDiagonalization" begin
        @testset "degenerate" begin
            fulldiag = FullDiagonalization(; n_target=L+1, tol_degeneracy=1e-4)
            basis, h, info = assemble(H, grid_off, greedy, fulldiag, qrcomp; callback=x -> x)
            test_L6_magn_plateaus(basis, h)
        end

        @testset "non-degenerate" begin
            fulldiag = FullDiagonalization(; n_target=1, tol_degeneracy=0.0)
            basis, h, info = assemble(H, grid_off, greedy, fulldiag, qrcomp; callback=x -> x)
            test_L6_magn_plateaus(basis, h)
        end
    end

    @testset "Proper orthogonal decomposition using LOBPCG" begin
        @testset "degenerate" begin
            lobpcg = LOBPCG(; n_target=L+1, tol_degeneracy=1e-4, tol=1e-9)
            basis, h_cache, info = assemble(H, grid_off, pod, lobpcg)
            h = h_cache.h
            test_L6_magn_plateaus(basis, h)
        end

        @testset "non-degenerate" begin
            lobpcg = LOBPCG(; n_target=1, tol_degeneracy=0.0, tol=1e-9)
            basis, h_cache, info = assemble(H, grid_off, pod, lobpcg)
            h = h_cache.h
            test_L6_magn_plateaus(basis, h)
        end
    end
end