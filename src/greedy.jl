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

# Convenience struct for efficient H, H² compression
# TODO: make parametric type
struct HamiltonianCache
    H::AffineDecomposition
    HΨ::Array
    h::AffineDecomposition
    h²::AffineDecomposition
end
function HamiltonianCache(H::AffineDecomposition, basis::RBasis)
    HΨ = [term * basis.snapshots * basis.vectors for term in H.terms]
    # h = AffineDecomposition([basis.vectors' * basis.snapshots' * v for v in HΨ], H.coefficient_map)
    h = compress(H, basis)
    h² = AffineDecomposition(
        reshape([v1' * v2 for v1 in HΨ for v2 in HΨ], (n_terms(H), n_terms(H))),
        μ -> (H.coefficient_map(μ) * H.coefficient_map(μ)')
    )
    HamiltonianCache(H, HΨ, h, h²)
end

# Full reduce: recompute all HΨ, compute all H matrix elements
# Replace by: extend!(hc, Ψ) that computes only new elements
function extend!(hc::HamiltonianCache, basis::RBasis)
    # TODO: only necessary overlaps
    HamiltonianCache(hc.H, basis)
end

# Reconstruct ground state from RB eigenvector
# TODO: Where to place this? -> not possible in rbasis.jl due to AffineDecomposition
function reconstruct(basis::RBasis, h::AffineDecomposition, μ, solver_online)
    _, φ_rb = solve(h, basis.metric, μ, solver_online)
    basis.snapshots * basis.vectors * φ_rb
end

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
    basis = RBasis(truth.vectors, [μ₁,], I, truth.vectors' * truth.vectors)
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
        Ψ₀ = init_from_rb ? reconstruct(basis, h_cache.h, μ_next, solver_online) : nothing
        @show Ψ₀
        truth = solve(H, μ_next, Ψ₀, solver_truth)

        # Append truth vector according to solver method
        # TODO: write extend or extend! -> possibly mutates snapshots (e.g. with MPS)
        basis_new, extend_info = extend!(basis, truth.vectors, μ_next, compressalg)

        # Exit: ill-conditioned BᵀB
        metric_condition = cond(basis_new.metric)
        if metric_condition > 1e2 # Global constant for max. condition?
            @warn "Stopped assembly due to ill-conditioned BᵀB." metric_condition
            break
        end
        # Exit: no vector was appended to basis
        if dim(basis_new) == dim(basis)
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