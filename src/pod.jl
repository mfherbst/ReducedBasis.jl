# Proper orthogonal decomposition as reference
struct POD
    n_truth_max::Int
end

function assemble(H::AffineDecomposition, grid, pod::POD, solver_truth)
    @assert pod.n_truth ≤ length(grid) && pod.n_truth_max ≤ size(H, 1)

    # Compute truth solution on all points of the grid
    vectors = Vector{Matrix}(undef, length(grid))
    parameters = SVector[]
    for (i, μ) in enumerate(grid)
        Ψ₀ = error()
        sol = solve(H, μ, Ψ₀, solver_truth)
        vectors[:, i] = sol.vectors
        push!(parameters, fill(μ, size(sol.vectors, 2)))
    end

    # SVD to obtain orthogonal basis
    U, Σ, V = svd(hcat(vectors...))

    # Extract reduced basis
    snapshots = U[:, 1:pod.n_truth]
    basis = RBasis(snapshots, parameters, I, snapshots' * snapshots)

    # Compress relevant parameter-independent matrices for fast online computation
    h_cache = compress(H, basis)

    info = (; basis, h_cache, U, Σ, V, Σ_norm=(Σ/Σ[1])[1:n_truth])
    basis, h_cache.h, info
end