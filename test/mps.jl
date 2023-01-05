using Test, ITensors
using ReducedBasis

@testset "Offline and online phase for XXZ chain using MPS" begin
    function xxz_chain(sts::IndexSet; kwargs...)
        xy_term   = OpSum()
        zz_term   = OpSum()
        magn_term = OpSum()
        for i in 1:(length(sts) - 1)
            xy_term   += 0.5, "S+", i, "S-", i + 1
            xy_term   += 0.5, "S-", i, "S+", i + 1
            zz_term   +=      "Sz", i, "Sz", i + 1
            magn_term +=      "Sz", i
        end
        magn_term += "Sz", length(sts)  # Add last magnetization term
        coefficient_map = μ -> [1.0, μ[1], -μ[2]]
        AffineDecomposition(
            [ApproxMPO(MPO(xy_term, sts), xy_term; kwargs...),
            ApproxMPO(MPO(zz_term, sts), zz_term; kwargs...),
            ApproxMPO(MPO(magn_term, sts), magn_term; kwargs...)],
            coefficient_map,
        )
    end

    # Offline/online parameters
    L        = 6
    sites    = siteinds("S=1/2", L)
    H        = xxz_chain(sites; cutoff=1e-9)
    M        = AffineDecomposition([H.terms[3]], μ -> [2 / L])
    Δ_off    = range(-1.0, 2.5, 40)
    hJ_off   = range(0.0, 3.5, 40)
    grid_off = RegularGrid(Δ_off, hJ_off);
    Δ_on     = range(first(Δ_off), last(Δ_off), 150)
    hJ_on    = range(first(hJ_off), last(hJ_off), 150)
    grid_on  = RegularGrid(Δ_on, hJ_on)

    edcomp = EigenDecomposition(; cutoff=1e-7)
    dm_deg = DMRG(;
        n_target=L+1,
        tol_degeneracy=1e-4,
        sweeps=default_sweeps(),
        observer=() -> DMRGObserver(; energy_tol=1e-9),
    )
    dm_nondeg = DMRG(;
        n_target=1,
        tol_degeneracy=0.0,
        sweeps=default_sweeps(),
        observer=() -> DMRGObserver(; energy_tol=1e-9),
    )

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
    
    @testset "Initial guess from RB eigenvector" begin
        greedy = Greedy(; estimator=Residual(), tol=1e-3, n_truth_max=64, init_from_rb=true)
        @testset "Greedy assembly: degenerate" begin
            basis, h, info = assemble(H, grid_off, greedy, dm_deg, edcomp; callback=x -> x)
            test_L6_magn_plateaus(basis, h)
        end
        @testset "Greedy assembly: non-degenerate" begin
            basis, h, info = assemble(H, grid_off, greedy, dm_nondeg, edcomp; callback=x -> x)
            test_L6_magn_plateaus(basis, h)
        end
    end

    @testset "Random initial guess" begin
        greedy = Greedy(; estimator=Residual(), tol=1e-3, n_truth_max=64, init_from_rb=false)
        @testset "Greedy assembly: degenerate" begin
            basis, h, info = assemble(H, grid_off, greedy, dm_deg, edcomp; callback=x -> x)
            test_L6_magn_plateaus(basis, h)
        end
        @testset "Greedy assembly: non-degenerate" begin
            basis, h, info = assemble(H, grid_off, greedy, dm_nondeg, edcomp; callback=x -> x)
            test_L6_magn_plateaus(basis, h)
        end
    end
end
