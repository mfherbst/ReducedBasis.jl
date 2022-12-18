# Proper orthogonal decomposition as reference method
struct POD
    n_truth::Int
end
POD(; n_truth=64) = POD(n_truth)

function assemble(H::AffineDecomposition, grid, pod::POD, solver_truth)
    @assert pod.n_truth ≤ length(grid) && pod.n_truth ≤ size(H, 1)

    # Compute truth solution on all points of the grid
    vectors = Vector{Matrix}(undef, length(grid))
    parameters = eltype(grid)[]
    @showprogress "Truth solving on $(size(grid)) grid..." for (i, μ) in enumerate(grid)
        Ψ₀ = Matrix(qr(randn(size(H, 1), solver_truth.n_target)).Q)
        sol = solve(H, μ, Ψ₀, solver_truth)
        vectors[i] = sol.vectors
        append!(parameters, fill(μ, size(sol.vectors, 2)))
    end

    # SVD to obtain orthogonal basis
    U, Σ, V = svd(hcat(vectors...))

    # Extract reduced basis
    snapshots = U[:, 1:pod.n_truth]
    # TODO: reorder parameters according to singular value ordering?
    basis = RBasis(snapshots, parameters, I, snapshots' * snapshots)

    # Hamiltonian compressions
    h_cache = HamiltonianCache(H, basis)

    info = (; basis, h_cache, U, Σ, V, Σ_norm=(Σ/Σ[1])[1:pod.n_truth])
    basis, h_cache.h, info
end