abstract type ErrorEstimate end
struct Residual <: ErrorEstimate end

function estimate_error(::Residual, μ, h_cache::HamiltonianCache, basis::RBasis, sol_rb)
    h²_sum = h_cache.h²(μ)
    sum_of_squares = sum(zip(sol_rb.values, eachcol(sol_rb.vectors))) do (λ, φ)
        abs(φ' * h²_sum * φ - λ^2 * φ' * basis.metric * φ)
    end
    sqrt(sum_of_squares)
end

Base.@kwdef struct Greedy
    estimator::ErrorEstimate
    tol::Float64 = 1e-3
    n_truth_max::Int = 64
    init_from_rb::Bool = true
    verbose::Bool = true
end

# Reconstruct ground state from RB eigenvector
function estimate_gs(
    basis::RBasis, h::AffineDecomposition, μ, _, solver_online,
)  # TODO: How to deal with redundant argument in this case?
    _, φ_rb = solve(h, basis.metric, μ, solver_online)
    hcat(basis.snapshots...) * basis.vectors * φ_rb
    # φ_trans = basis.vectors * φ_rb
    # Φ_rb = Matrix{eltype(φ_trans)}(undef, length(basis.snapshots[1]), size(φ_rb, 2))
    # for j in 1:length(basis.snapshots), k in 1:size(φ_trans, 2)
    #     for i in 1:length(basis.snapshots)
    #         Φ_rb[j, k] += basis.snapshots[i][j] * φ_trans[i, k]  # Error with .+= ?
    #     end
    # end
    # Φ_rb
end

function assemble(
    H::AffineDecomposition,
    grid,
    greedy::Greedy,
    solver_truth,
    compressalg;
    solver_online=FullDiagonalization(;
        tol_degeneracy=solver_truth.tol_degeneracy, n_target=solver_truth.n_target,
    ),
    callback=print_callback,
)
    t_init  = time_ns()  # TODO: how to outsource this time measurement to callback?
    μ₁      = grid[1]
    truth   = solve(H, μ₁, nothing, solver_truth)
    BᵀB     = overlap_matrix(truth.vectors, truth.vectors)
    basis   = RBasis(truth.vectors, fill(μ₁, length(truth.vectors)), I, BᵀB, BᵀB)
    h_cache = HamiltonianCache(H, basis)
    info    = (; iteration=1, err_max=NaN, μ=μ₁, basis, h_cache, t=t_init, state=:iterate)
    callback(info)

    for n in 2:(greedy.n_truth_max)
        t_iterstart = time_ns()
        # Compute residual on training grid and find maximum for greedy condition
        err_grid = similar(grid, Float64)
        λ_grid   = similar(grid, Vector{Float64})
        for (idx, μ) in pairs(grid)
            sol = solve(h_cache.h, basis.metric, μ, solver_online)
            λ_grid[idx] = sol.values
            err_grid[idx] = estimate_error(greedy.estimator, μ, h_cache, basis, sol)
        end
        err_max, idx_max = findmax(err_grid)
        μ_next = grid[idx_max]
        # Exit: μ_next has already been solved
        if μ_next ∈ basis.parameters
            greedy.verbose && @warn "μ=$μ_next has already been solved"
            break
        end

        # Construct initial guess at μ_next and run truth solve
        if greedy.init_from_rb
            Ψ₀ = estimate_gs(basis, h_cache.h, μ_next, solver_truth, solver_online)
        else
            Ψ₀ = nothing
        end
        truth = solve(H, μ_next, Ψ₀, solver_truth)

        # Append truth vector according to solver method
        d_basis_old = dimension(basis)
        # basis_new, extend_info... = extend!(basis, truth.vectors, μ_next, compressalg)
        basis_new, extend_info... = extend(basis, truth.vectors, μ_next, compressalg)

        # Exit: ill-conditioned BᵀB
        metric_condition = cond(basis_new.metric)
        if metric_condition > 1e2  # TODO: Global constant for max. condition?
            greedy.verbose && @warn "stopped assembly due to ill-conditioned BᵀB" metric_condition
            break
        end
        # Exit: no vector was appended to basis
        if dimension(basis_new) == d_basis_old
            greedy.verbose && @warn "stopped assembly since new snapshot was insignificant"
            break
        end

        # Update basis with new snapshot/vector/metric and compute reduced terms
        basis   = basis_new
        h_cache = HamiltonianCache(h_cache, basis)

        # Update iteration state info
        info = (; iteration=n, err_grid, λ_grid, err_max, μ=μ_next,
                extend_info, basis, h_cache, t=t_iterstart, state=:iterate)
        callback(info)

        # Exit iteration if residuals drops below tolerance
        if err_max < greedy.tol
            greedy.verbose && println("reached residual target")
            break
        end
    end

    callback((; t=t_init, state=:finalize))
    basis, h_cache.h, info
end