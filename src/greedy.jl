struct Greedy
    estimator::ErrorEstimate
    tol_residual::Float64
    n_truth_max::Int
end
function Greedy(; estimator=Residual(), tol_residual=1e-3, n_truth_max=64)
    Greedy(estimator, tol_residual, n_truth_max)
end

abstract type ErrorEstimate end
struct Residual <: ErrorEstimate end

function estimate_error(::Residual, μ, h²::AffineDecomposition, b, λ_rb, φ_rb)
    h²_sum = h²(μ)
    sum_of_squares = sum(zip(λ_rb, eachcol(φ_rb))) do (λ, φ)
        abs(φ' * h²_sum * φ - λ^2 * φ' * b * φ)
    end
    sqrt(sum_of_squares)
end

# Convenience struct for efficient H, H² compression
# D: dimension of Hamiltonian AD
# M: matrix-type of Hamiltonian AD
# V: vector-type of Hamiltonian application HΨ
mutable struct HamiltonianCache{D,M,V}
    H::AffineDecomposition{D,M,Function}
    HΨ::Array{D,V}
    h::AffineDecomposition{D,M,Function}
    h²::AffineDecomposition{D,M,Function}
end
# Non-mutable struct also possible?
function HamiltonianCache(H::AffineDecomposition, basis::ReducedBasis)
    HΨ = [term * basis.snapshots for term in H.terms]
    h = AffineDecomposition([basis.snapshots' * v for v in HΨ], H.coefficient_map) # B'HB
    h² = AffineDecomposition(
        [v1' * v2 for v1 in HΨ for v2 in HΨ],
        μ -> (H.coefficient_map(μ) * H.coefficient_map(μ)')
    ) # B'HHB
    HamiltonianCache(H, HΨ, h, h²)
end

# Full reduce: recompute all HΨ, compute all H matrix elements
# Replace by: extend!(hc, Ψ) that computes only new elements
function extend!(hc::HamiltonianCache, basis::ReducedBasis)
    hc.HΨ = [term * basis.snapshots for term in hc.H.terms]
    hc.h = AffineDecomposition([basis.snapshots' * v for v in hc.HΨ], hc.H.coefficient_map) # B'HB
    hc.h² = AffineDecomposition(
        [v1' * v2 for v1 in hc.HΨ for v2 in hc.HΨ],
        μ -> (hc.H.coefficient_map(μ) * hc.H.coefficient_map(μ)')
    ) # B'HHB
end

function assemble(H::AffineDecomposition, grid, greedy::Greedy, solver_truth;
                  solver_online=FullDiagonalization(), init_from_rb=true,
                  callback=default_callback)
    # First iteration separate? (to avoid annoying if-statements in for-loop)
    t_start = time_ns()
    μ₁ = grid[1]
    truth = truth_solve(model, μ₁, solver_truth)
    basis = ReducedBasis(truth.vectors, fill(μ₁, length(truth.vectors)),
                         I, truth.vectors' * truth.vectors)
    h_cache = HamiltonianCache(H, basis)
    iter_time = TimerOutput.prettytime(time_ns() - t_start)
    info = (; iteration=1, err_max=NaN, μ=μ₁, basis, h_cache, iter_time, state=:starting)
    callback(info)

    for n = 2:greedy.n_truth_max
        t_start = time_ns()
        # Compute residual on training grid and find maximum for greedy condition
        err = map(grid) do μ
            solution = online_solve(h_cache.h, metric, μ, solver_online)
            estimate_error(greedy.estimator, μ, h_cache.h², metric, solution...)
        end
        err_max, idx_max = findmax(err)
        μ_next = grid[idx_max]

        # Construct initial guess at μ_next and run truth solve
        Ψ₀ = init_from_rb ? reconstruct(basis, μ) : nothing
        truth = truth_solve(H, μ_next, Ψ₀, solver_truth)

        # Append truth vector according to solver method
        # Includes orthonormalization
        # How to deal with large truth solve objects? (cannot copy)
        # write extend or extend! -> possibly mutates snapshots (e.g. with MPS)
        newbasis = extend(basis, truth.vectors, solver_truth)

        # Early exit conditions
        # Ill-conditioned BᵀB
        metric_condition = cond(newbasis.metric)
        if metric_condition > 1e2 # Global constant for max. condition?
            @warn "Stopped assembly due to ill-conditioned BᵀB." metric_condition
            break
        end
        # No vector was appended to basis
        if dim(newbasis) == dim(basis)
            @warn "Stopped assmembly since new snapshot was insignificant."
            break
        end

        # Update basis with new snapshot/vector/metric and compute reduced terms
        basis = newbasis
        extend!(h_cache, basis)

        # Update iteration state info
        iter_time = TimerOutput.prettytime(time_ns() - t_start)
        info = (; iteration=n, err, err_max, μ=μ_next, basis, h_cache, iter_time, state=:running)
        callback(info)

        # Exit iteration if residuals drops below tolerance
        if err_max < greedy.tol_residual
            # Use println(...) or @info() here?
            println("Reached residual target.")
            break
        end
    end

    diagnostics = callback(info)

    basis, h_cache.h, diagnostics
end