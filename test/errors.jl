using Test, LinearAlgebra, Random
using ReducedBasis

@testset "Low errors at snapshot parameter points" begin
    # Offline/online parameters
    L    = 6
    H    = xxz_chain(L)
    M    = AffineDecomposition([H.terms[3]], μ -> [2 / L])
    Δ    = range(-1.0, 2.5, 30)
    hJ   = range(0.0, 3.5, 30)
    grid = RegularGrid(Δ, hJ)

    # Offline basis assembly
    greedy = Greedy(;
        estimator=Residual(), tol=1e-3, n_truth_max=64, init_from_rb=true, verbose=false
    )
    lobpcg = LOBPCG(; n_target=L+1, tol_degeneracy=1e-4, tol=1e-9)
    fulldiag = FullDiagonalization(; n_target=L+1, tol_degeneracy=1e-4)
    qrcomp = QRCompress(; tol=1e-10)
    basis, h, info = assemble(H, grid, greedy, lobpcg, qrcomp; callback=x -> x)
    h_cache = info.h_cache

    # Online EVP and full diagonalization on solved μ
    errors     = Float64[]
    values     = Float64[]
    vectors    = Matrix[]
    values_fd  = Float64[]
    vectors_fd = Matrix[]
    for μ in unique(basis.parameters)
        sol    = solve(h_cache.h, basis.metric, μ, fulldiag)
        sol_fd = solve(h_cache.H, μ, nothing, fulldiag)
        push!(errors, estimate_error(greedy.estimator, μ, h_cache, basis, sol))
        append!(values, sol.values)
        push!(vectors, sol.vectors)
        append!(values_fd, sol_fd.values)
        push!(vectors_fd, hcat(sol_fd.vectors...))
    end

    # Low error estimates
    @test maximum(abs, errors) < info.err_max
    # Low eigenvalue errors
    @test maximum((values .- values_fd) ./ values_fd) < sqrt(info.err_max)
    # Low eigenvector errors (via projectors onto degenerate subspace)
    B = hcat(basis.snapshots...) * basis.vectors
    hilbert_vectors = map(φ -> B * φ, vectors)
    proj_fd = [v * v' for v in vectors_fd]
    vector_errors = [norm(Φ * Φ' - p) / norm(p) for (Φ, p) in zip(hilbert_vectors, proj_fd)]
    @test maximum(vector_errors) < sqrt(info.err_max)
end