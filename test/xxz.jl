using Test, LinearAlgebra
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
        estimator=Residual(), tol=1e-3, n_truth_max=32, init_from_rb=true, verbose=false
    )
    qrcomp = QRCompress(; tol=1e-10)
    pod    = POD(; n_truth=32, verbose=false)

    # Check if RB energy differences for subsequent assembly iterations are positive on grid
    function test_variational(ic::InfoCollector)
        E_grids = [map(maximum, λ_grid) for λ_grid in ic.data[:λ_grid]]
        all_greater_zero = [all(round.(E .- E_grids[end]; digits=12) .≥ 0.0) for E in E_grids]
        @test all(all_greater_zero)
    end

    # Check values of L/2+1 magnetization plateaus for L=6
    function test_L6_magn_plateaus(basis::RBasis, h::AffineDecomposition, solver_truth)
        @testset "Correct magnetization values" begin
            fd = FullDiagonalization(solver_truth)
            m = compress(M, basis)
            m_reduced = m([1])
            magnetization = map(grid_on) do μ
                _, φ_rb = solve(h, basis.metric, μ, fd)
                sum(eachcol(φ_rb)) do φ
                    abs(dot(φ, m_reduced, φ)) / size(φ_rb, 2)
                end
            end
            
            @test magnetization[end, 1] ≈ 0.0  atol=1e-6
            @test magnetization[1, end] ≈ 1.0  atol=1e-6
            @test magnetization[75, 35] ≈ 1//3 atol=1e-6
            @test magnetization[75, 60] ≈ 2//3 atol=1e-6
        end
    end

    # Check if errors are low at solved parameter points
    function test_low_errors(basis::RBasis, info, solver_truth)
        @testset "Low errors at solved parameter points" begin
            fd = FullDiagonalization(solver_truth)
            errors     = Float64[]
            values     = Float64[]
            vectors    = Matrix[]
            values_fd  = Float64[]
            vectors_fd = Matrix[]
            for μ in unique(basis.parameters)
                sol    = solve(info.h_cache.h, basis.metric, μ, fd)
                sol_fd = solve(info.h_cache.H, μ, nothing, fd)
                push!(errors, estimate_error(greedy.estimator, μ, info.h_cache, basis, sol))
                append!(values, sol.values)
                push!(vectors, sol.vectors)
                append!(values_fd, sol_fd.values)
                push!(vectors_fd, hcat(sol_fd.vectors...))
            end

            # Low error estimates
            @test maximum(abs, errors) < info.err_max
            # Low eigenvalue errors
            @test maximum((values .- values_fd) ./ values_fd) < sqrt(info.err_max)
            # If degenerate: low eigenvector errors (via projectors onto degenerate subspace)
            if solver_truth.n_target > 1 && solver_truth.tol_degeneracy > 0.0
                B = hcat(basis.snapshots...) * basis.vectors
                hilbert_vectors = map(φ -> B * φ, vectors)
                proj_fd = [v * v' for v in vectors_fd]
                vector_errors = [norm(Φ * Φ' - p) / norm(p) for (Φ, p) in zip(hilbert_vectors, proj_fd)]
                @test maximum(vector_errors) < sqrt(info.err_max)
            end
        end
    end

    @testset "Greedy assembly: LOBPCG" begin
        @testset "degenerate" begin
            collector = InfoCollector(:λ_grid)
            lobpcg = LOBPCG(; n_target=1, tol_degeneracy=1e-4, tol=1e-9)
            basis, h, info = assemble(H, grid_off, greedy, lobpcg, qrcomp; callback=collector)
            @test multiplicity(basis)[1] > 1
            test_variational(collector)
            test_L6_magn_plateaus(basis, h, lobpcg)
            test_low_errors(basis, info, lobpcg)
        end

        @testset "non-degenerate" begin
            collector = InfoCollector(:λ_grid)
            lobpcg = LOBPCG(; n_target=1, tol_degeneracy=0.0, tol=1e-9)
            basis, h, info = assemble(H, grid_off, greedy, lobpcg, qrcomp; callback=collector)
            test_variational(collector)
            test_L6_magn_plateaus(basis, h, lobpcg)
            test_low_errors(basis, info, lobpcg)
        end
    end

    @testset "Greedy assembly: FullDiagonalization" begin
        @testset "degenerate" begin
            collector = InfoCollector(:λ_grid)
            fulldiag = FullDiagonalization(; n_target=1, tol_degeneracy=1e-4)
            basis, h, info = assemble(H, grid_off, greedy, fulldiag, qrcomp; callback=collector)
            @test multiplicity(basis)[1] > 1
            test_variational(collector)
            test_L6_magn_plateaus(basis, h, fulldiag)
            test_low_errors(basis, info, fulldiag)
        end

        # TODO: find replacement for weird behavior of lowest eigenvector when using eigen without a 1:n range
        # @testset "non-degenerate" begin
        #     collector = InfoCollector(:λ_grid)
        #     fulldiag = FullDiagonalization(; n_target=1, tol_degeneracy=0.0)
        #     basis, h, info = assemble(H, grid_off, greedy, fulldiag, qrcomp; callback=collector)
        #     test_variational(collector)
        #     test_L6_magn_plateaus(basis, h, fulldiag)
        #     test_low_errors(basis, info, fulldiag)
        # end
    end

    @testset "Proper orthogonal decomposition: LOBPCG" begin
        @testset "degenerate" begin
            lobpcg = LOBPCG(; n_target=1, tol_degeneracy=1e-4, tol=1e-9)
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