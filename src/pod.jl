# Proper orthogonal decomposition as reference method
Base.@kwdef struct POD
    n_truth::Int = 64
    verbose::Bool = true
end

function assemble(H::AffineDecomposition, grid, pod::POD, solver_truth)
    @assert pod.n_truth ≤ length(grid) && pod.n_truth ≤ size(H, 1)

    # Compute truth solution on all points of the grid
    vectors = Vector{Matrix}(undef, length(grid))
    parameters = eltype(grid)[]
    progbar = Progress(
        length(grid); desc="Truth solving on $(size(grid)) grid...", enabled=pod.verbose
    )
    for (i, μ) in enumerate(grid)
        Ψ₀         = Matrix(qr(randn(size(H, 1), solver_truth.n_target)).Q)
        sol        = solve(H, μ, Ψ₀, solver_truth)
        vectors[i] = hcat(sol.vectors...)
        append!(parameters, fill(μ, length(sol.vectors)))
        next!(progbar)
    end

    # SVD to obtain orthogonal basis
    U, Σ, V = svd(hcat(vectors...))

    # Extract reduced basis with desired number of truth solves
    mult      = [count(isequal(μ), parameters) for μ in unique(parameters)]
    idx_trunc = sum(@view(mult[1:(pod.n_truth)])) 
    # TODO: reorder parameters according to singular value ordering?
    snapshots = [U[:, i] for i in 1:idx_trunc]
    U_trunc   = @view U[:, 1:idx_trunc]
    BᵀB       = U_trunc' * U_trunc
    basis     = RBasis(snapshots, parameters[1:idx_trunc], I, BᵀB, BᵀB)

    basis, (; U, Σ, V)
end