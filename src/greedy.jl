abstract type ErrorEstimate end
struct Residual <: ErrorEstimate end

function estimate_error(::Residual, μ, h²::AffineDecomposition, b, λ_rb, φ_rb)
    h²_sum = h²(μ)
    sum_of_squares = sum(zip(λ_rb, eachcol(φ_rb))) do (λ, φ)
        abs(φ' * h²_sum * φ - λ^2 * φ' * b * φ)
    end
    sqrt(sum_of_squares)
end

struct Greedy
    estimator::ErrorEstimate
    tol::Float64
    n_truth_max::Int
end
function Greedy(; estimator=Residual(), tol=1e-3, n_truth_max=64)
    Greedy(estimator, tol, n_truth_max)
end

# Reconstruct ground state from RB eigenvector
# TODO: Change name?
function estimate_gs(basis::RBasis, h::AffineDecomposition, μ, solver_online)
    _, φ_rb = solve(h, basis.metric, μ, solver_online)
    basis.snapshots * basis.vectors * φ_rb
end

# Overlap matrix of column vectors of two matrices
# TODO: Is this really necessary?
overlap_matrix(m1::Matrix, m2::Matrix) = m1' * m2

function assemble(
    H::AffineDecomposition,
    grid,
    greedy::Greedy,
    solver_truth,
    compressalg;
    solver_online=FullDiagonalization(
        tol_degeneracy=solver_truth.tol_degeneracy,
        n_target=solver_truth.n_target
    ),
    init_from_rb=true,
    callback=print_callback
)
    # TODO: First iteration separate? (to avoid annoying if-statements in for-loop)
    t_init = time_ns()
    μ₁ = grid[1]
    truth = solve(H, μ₁, nothing, solver_truth)
    basis = RBasis(truth.vectors, fill(μ₁, size(truth.vectors, 2)), I, Matrix(overlap_matrix(truth.vectors, truth.vectors)))
    h_cache = HamiltonianCache(H, basis)
    info = (; iteration=1, err_max=NaN, μ=μ₁, basis, h_cache, t=t_init, state=:run)
    callback(info)

    for n = 2:greedy.n_truth_max
        t_iterstart = time_ns()
        # Compute residual on training grid and find maximum for greedy condition
        err_grid, λ_grid = similar(grid, Float64), similar(grid, Vector{Float64})
        for (idx, μ) in pairs(grid)
            λ_grid[idx], φ_rb = solve(h_cache.h, basis.metric, μ, solver_online)
            err_grid[idx] = estimate_error(
                greedy.estimator, μ, h_cache.h², basis.metric, λ_grid[idx], φ_rb
            )
        end
        err_max, idx_max = findmax(err_grid)
        μ_next = grid[idx_max]

        # Construct initial guess at μ_next and run truth solve
        Ψ₀ = init_from_rb ? estimate_gs(basis, h_cache.h, μ_next, solver_online) : nothing
        truth = solve(H, μ_next, Ψ₀, solver_truth)

        # Append truth vector according to solver method
        d_basis_old = dimension(basis)
        basis_new, extend_info = extend!(basis, truth.vectors, μ_next, compressalg)

        # Exit: ill-conditioned BᵀB
        metric_condition = cond(basis_new.metric)
        if metric_condition > 1e2 # Global constant for max. condition?
            @warn "Stopped assembly due to ill-conditioned BᵀB." metric_condition
            break
        end
        # Exit: no vector was appended to basis
        if dimension(basis_new) == d_basis_old
            @warn "Stopped assembly since new snapshot was insignificant."
            break
        end

        # Update basis with new snapshot/vector/metric and compute reduced terms
        basis = basis_new
        h_cache = extend!(h_cache, basis)

        # Update iteration state info
        info = (; iteration=n, err_grid, λ_grid, err_max, μ=μ_next, extend_info, basis, h_cache, t=t_iterstart, state=:run)
        callback(info)

        # Exit iteration if residuals drops below tolerance
        if err_max < greedy.tol
            # TODO: Use println(...) or @info() here?
            println("Reached residual target.")
            break
        end
    end

    callback((; t=t_init, state=:finalize))
    basis, h_cache.h, info
end