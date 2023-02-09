using Test, LinearAlgebra, ITensors, SparseArrays
using ReducedBasis
using ReducedBasis: reconstruct

@testset "MPS offline & online phase: XXZ chain" begin
    function xxz_chain(sts::IndexSet; kwargs...)
        xy_term, zz_term, magn_term = OpSum(), OpSum(), OpSum()
        for i in 1:(length(sts) - 1)
            xy_term   += 0.5, "S+", i, "S-", i + 1
            xy_term   += 0.5, "S-", i, "S+", i + 1
            zz_term   +=      "Sz", i, "Sz", i + 1
            magn_term +=      "Sz", i
        end
        magn_term += "Sz", length(sts)  # Add last magnetization term
        coefficients = μ -> [1.0, μ[1], -μ[2]]
        AffineDecomposition(
            [ApproxMPO(MPO(xy_term, sts), xy_term; kwargs...),
            ApproxMPO(MPO(zz_term, sts), zz_term; kwargs...),
            ApproxMPO(MPO(magn_term, sts), magn_term; kwargs...)],
            coefficients)
    end
    function xxz_chain(N)
        σx = sparse([0.0 1.0; 1.0 0.0])
        σy = sparse([0.0 -im; im 0.0])
        σz = sparse([1.0 0.0; 0.0 -1.0])
        H1 = 0.25 * sum([to_global(N, σx, i) * to_global(N, σx, i + 1) +
                            to_global(N, σy, i) * to_global(N, σy, i + 1) for i in 1:(N-1)])
        H2 = 0.25 * sum([to_global(N, σz, i) * to_global(N, σz, i + 1) for i in 1:(N-1)])
        H3 = 0.5  * sum([to_global(N, σz, i) for i in 1:N])
        coefficients = μ -> [1.0, μ[1], -μ[2]]
        AffineDecomposition([H1, H2, H3], coefficients)
    end

    # Offline/online parameters
    L        = 6
    sites    = siteinds("S=1/2", L)
    H        = xxz_chain(sites; cutoff=1e-14)  # TODO: if too large -> low residual errors at solved μ do not hold!
    H_matrix = xxz_chain(L)
    M        = AffineDecomposition([H.terms[3]], μ -> [2 / L])
    Δ_off    = range(-1.0, 2.5; length=40)
    hJ_off   = range(0.0, 3.5; length=40)
    grid_off = RegularGrid(Δ_off, hJ_off);
    Δ_on     = range(first(Δ_off), last(Δ_off); length=100)
    hJ_on    = range(first(hJ_off), last(hJ_off); length=100)
    grid_on  = RegularGrid(Δ_on, hJ_on)

    edcomp = EigenDecomposition(; cutoff=1e-7)
    dm_deg = DMRG(; n_states=L+1, tol_degeneracy=1e-4,
                  sweeps=default_sweeps(),
                  observer=() -> DMRGObserver(; energy_tol=1e-9))
    dm_nondeg = DMRG(; n_states=1, tol_degeneracy=0.0,
                     sweeps=default_sweeps(),
                     observer=() -> DMRGObserver(; energy_tol=1e-9))

    # Check if RB energy differences for subsequent iterations are positive on grid
    function test_variational(ic::InfoCollector)
        E_grids = [map(maximum, λ_grid) for λ_grid in ic.data[:λ_grid]]
        all_greater_zero = [all(round.(E .- E_grids[end]; digits=12) .≥ 0.0)
                            for E in E_grids]
        @test all(all_greater_zero)
    end

    # Check values of L/2+1 magnetization plateaus for L=6
    function test_L6_magn_plateaus(info, solver_truth)
        @testset "Correct magnetization values" begin
            fd = FullDiagonalization(solver_truth)
            m, _ = compress(M, info.basis)
            m_reduced = m()
            magnetization = map(grid_on) do μ
                _, φ_rb = solve(info.h_cache.h, info.basis.metric, μ, fd)
                sum(eachcol(φ_rb)) do φ
                    abs(dot(φ, m_reduced, φ))
                end / size(φ_rb, 2)
            end
            
            @test magnetization[end, 1] ≈ 0.0  atol=1e-6
            @test magnetization[1, end] ≈ 1.0  atol=1e-6
            @test magnetization[75, 35] ≈ 1//3 atol=1e-6
            @test magnetization[75, 60] ≈ 2//3 atol=1e-6
        end
    end

    # Check if errors are low at solved parameter points
    function test_low_errors(info, greedy, solver_truth)
        @testset "Low errors at solved parameter points" begin
            fd = FullDiagonalization(solver_truth)
            errors     = Float64[]
            values     = Float64[]
            vectors    = Matrix[]
            values_fd  = Float64[]
            vectors_fd = Matrix[]
            for μ in unique(info.basis.parameters)
                sol    = solve(info.h_cache.h, info.basis.metric, μ, fd)
                sol_fd = solve(H_matrix, μ, nothing, fd)
                push!(errors, estimate_error(greedy.estimator, μ, info.h_cache,
                                             info.basis, sol))
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
            if solver_truth.n_states > 1 && solver_truth.tol_degeneracy > 0.0
                vec_snapshots = reconstruct.(info.basis.snapshots)
                B = hcat(vec_snapshots...) * info.basis.vectors
                hilbert_vectors = map(φ -> B * φ, vectors)
                proj_fd = [v * v' for v in vectors_fd]
                vector_errors = [norm(Φ * Φ' - p) / norm(p)
                                 for (Φ, p) in zip(hilbert_vectors, proj_fd)]
                @test maximum(vector_errors) < sqrt(info.err_max)
            end
        end
    end
    
    @testset "Initial guess from RB eigenvector" begin
        greedy = Greedy(; estimator=Residual(), tol=1e-3, n_truth_max=64, 
                        init_from_rb=true, verbose=false)
        @testset "Greedy assembly: degenerate" begin
            collector = InfoCollector(:λ_grid)
            info = assemble(H, grid_off, greedy, dm_deg, edcomp; callback=collector)
            @test multiplicity(info.basis)[1] > 1
            test_variational(collector)
            test_L6_magn_plateaus(info, dm_deg)
            test_low_errors(info, greedy, dm_deg)
        end
        @testset "Greedy assembly: non-degenerate" begin
            collector = InfoCollector(:λ_grid)
            info = assemble(H, grid_off, greedy, dm_nondeg, edcomp; callback=collector)
            test_variational(collector)
            test_L6_magn_plateaus(info, dm_nondeg)
            test_low_errors(info, greedy, dm_nondeg)
        end
    end

    @testset "Random initial guess" begin
        greedy = Greedy(; estimator=Residual(), tol=1e-3, n_truth_max=64, 
                        init_from_rb=false, verbose=false)
        @testset "Greedy assembly: degenerate" begin
            collector = InfoCollector(:λ_grid)
            info = assemble(H, grid_off, greedy, dm_deg, edcomp; callback=collector)
            @test multiplicity(info.basis)[1] > 1
            test_variational(collector)
            test_L6_magn_plateaus(info, dm_deg)
            test_low_errors(info, greedy, dm_deg)
        end
        @testset "Greedy assembly: non-degenerate" begin
            collector = InfoCollector(:λ_grid)
            info = assemble(H, grid_off, greedy, dm_nondeg, edcomp; callback=collector)
            test_variational(collector)
            test_L6_magn_plateaus(info, dm_nondeg)
            test_low_errors(info, greedy, dm_nondeg)
        end
    end
end
