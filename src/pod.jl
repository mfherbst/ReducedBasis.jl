"""
Proper orthogonal decomposition assembly strategy.

# Fields

- `n_vectors::Int`: number of retained singular vectors of the snapshot matrix in the
  returned basis.
- `verbose::Bool=true`: shows the truth solve progress if `true`.
"""
@kwdef struct POD
    n_vectors::Int
    verbose::Bool = true
end

"""
    assemble(H::AffineDecomposition, grid, pod::POD, solver_truth)

Assemble basis using [`POD`](@ref).

Only ED solvers such as [`FullDiagonalization`](@ref) and [`LOBPCG`](@ref) are supported.
The generated [`RBasis`](@ref) will contain `pod.n_vectors` singular vectors in `snapshots`
and all grid points in `parameters`. This means that `parameters` and `snapshots` generally
have different lengths.
"""
function assemble(H::AffineDecomposition, grid, pod::POD, solver_truth)
    # TODO: This should be solved by a type annotation and an appropriate abstract type
    #       and not an assertion
    (solver_truth isa DMRG) && ArgumentError("Only ED solvers are supported.")
    @assert pod.n_vectors ≤ length(grid) && pod.n_vectors ≤ size(H, 1)

    # Compute truth solution on all points of the grid
    vectors = Vector{Matrix}(undef, length(grid))
    parameters = eltype(grid)[]
    progbar = Progress(length(grid);
                       desc="Truth solving on $(size(grid)) grid...", enabled=pod.verbose)
    for (i, μ) in enumerate(grid)
        Ψ₀  = Matrix(qr(randn(size(H, 1), solver_truth.n_target)).Q)
        sol = solve(H, μ, Ψ₀, solver_truth)
        vectors[i] = hcat(sol.vectors...)
        append!(parameters, fill(μ, length(sol.vectors)))
        next!(progbar)
    end

    # SVD to obtain orthogonal basis
    U, Σ, V = svd(hcat(vectors...))

    # Extract reduced basis with desired number of truth solves
    mult      = [count(isequal(μ), parameters) for μ in unique(parameters)]
    idx_trunc = sum(@view(mult[1:(pod.n_vectors)]))
    snapshots = [U[:, i] for i in 1:idx_trunc]
    U_trunc   = @view U[:, 1:idx_trunc]
    BᵀB       = U_trunc' * U_trunc
    basis     = RBasis(snapshots, parameters, I, BᵀB, BᵀB)

    (; basis, U, Σ, V)
end
