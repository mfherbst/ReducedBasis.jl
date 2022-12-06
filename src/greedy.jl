struct Greedy
    tol_residual::Float64
    n_truth_max::Int
end

function residual(grid::Grid, h::AffineDecomposition, h²::AffineDecomposition, metric)
    map(grid) do μ
        solution = eigen(h, metric, μ)
        compute_residual(μ, h², metric, solution...)
    end
end
function residual(μ, h²::AffineDecomposition, b, λ_rb, φ_rb)
    h²_sum = h²(μ)
    sum_of_squares = sum(zip(λ_rb, eachcol(φ_rb))) do (λ, φ)
        abs(φ' * h²_sum * φ - λ^2 * φ' * b * φ)
    end
    sqrt(sum_of_squares)
end

function assemble(H::AffineDecomposition, grid::Grid, greedy::Greedy, solver; display_progress=true)
    # How to have diagnostics solver-dependent?
    diagnostics = DataFrame(
        "n" => Int64[],
        "residual" => Float64[],
        "time" => String[],
        "snapshot" => eltype(grid)[],
    )

    # First iteration separate? (to avoid annoying if-statements in for-loop)
    μ₁ = grid[1]
    truth = truth_solve(model, μ₁, solver)
    basis = ReducedBasis(
        truth.vectors,
        fill(μ₁, length(truth.vectors)),
        I,
        truth.vectors' * truth.vectors,
    )
    HΨ, h, h² = reduce_hamiltonian(H, basis)

    for n = 2:greedy.n_truth_max
        # Compute residual on training grid and find maximum for greedy condition
        res = residual(h, basis.metric, grid)
        res_max, idx_max = findmax(res)
        μ_next = grid[idx_max]

        # Construct initial guess at μ_next and run truth solve
        Ψ₀ = reconstruct(basis, μ)
        truth = truth_solve(H, μ_next, Ψ₀, solver)

        # Append truth vector according to solver method
        # Includes orthonormalization
        # How to deal with large truth solve objects? (cannot copy)
        newbasis = extend(basis, truth.vectors, solver)

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
        HΨ, h, h² = reduce_hamiltonian(H, basis) # How to do this efficiently? -> need HΨ terms

        # Print diagnostics and exit iteration if residuals drops below tolerance
        display_progress && print_row(diagnostics, n)
        if res_max < greedy.tol_residual
            # Use println(...) or @info() here?
            println("Reached residual target.")
            break
        end
    end

    (; basis, h, diagnostics)
end