"""
Proper orthogonal decomposition assembly strategy.

# Fields

- `n_truth::Int=64`: number of included truth solves in the returned basis.
- `verbose::Bool=true`: shows the truth solve progress if `true`.
"""
@kwdef struct POD
    n_truth::Int = 64
    verbose::Bool = true
end

"""
    assemble(H::AffineDecomposition, grid, pod::POD, solver_truth)

Assemble basis using [`POD`](@ref). Only ED solvers such as
[`FullDiagonalization`](@ref) and [`LOBPCG`](@ref) are supported.
"""
function assemble(H::AffineDecomposition, grid, pod::POD, solver_truth)
    # TODO: This should be solved by a type annotation and an appropriate abstract type
    #       and not an assertion
    (solver_truth isa DMRG) && ArgumentError("Only ED solvers are supported.")
    @assert pod.n_truth ≤ length(grid) && pod.n_truth ≤ size(H, 1)

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
    idx_trunc = sum(@view(mult[1:(pod.n_truth)]))
    snapshots = [U[:, i] for i in 1:idx_trunc]
    U_trunc   = @view U[:, 1:idx_trunc]
    BᵀB       = U_trunc' * U_trunc
    basis     = RBasis(snapshots, parameters, I, BᵀB, BᵀB)

    basis, (; U, Σ, V)
end
