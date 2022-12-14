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
    HΨ::Vector  # Hard-code to Vector for now (generalized structure even needed?)
    ΨHΨ::Vector  # Matrix elements are needed for efficient extension/truncation
    ΨHHΨ::Matrix
    h::AffineDecomposition
    h²::AffineDecomposition
end
function HamiltonianCache(H::AffineDecomposition, basis::RBasis)
    HΨ = [term * basis.snapshots for term in H.terms]
    ΨHΨ = [basis.snapshots' * v for v in HΨ]
    ΨHHΨ = reshape([v1' * v2 for v1 in HΨ for v2 in HΨ], (n_terms(H), n_terms(H)))
    h = AffineDecomposition([basis.vectors' * matel * basis.vectors for matel in ΨHΨ], H.coefficient_map)
    h² = AffineDecomposition(
        [basis.vectors' * matel * basis.vectors for matel in ΨHHΨ],
        μ -> (H.coefficient_map(μ) * H.coefficient_map(μ)')
    )
    HamiltonianCache(H, HΨ, ΨHΨ, ΨHHΨ, h, h²)
end

# Compute only new HΨ and necessary matrix elements
function extend!(hc::HamiltonianCache, basis::RBasis)
    d_basis = dim(basis)
    m = multiplicity(basis)[end]  # Multiplicity of last truth solve
    
    # Compute new Hamiltonian application HΨ
    for (q, term) in enumerate(hc.HΨ)
        term_new = zeros(size(term, 1), size(term, 2) + m)
        term_new[:, 1:d_basis-m] = term
        term_new[:, d_basis-m+1:end] = hc.H.terms[q] * @view(basis.snapshots[:, d_basis-m+1:end])
        hc.HΨ[q] = term_new  # TODO: how to generalize application to MPS case?
    end

    # Compute only new matrix elements
    for (q, term) in enumerate(hc.ΨHΨ)
        term_new = zeros(size(term) .+ m)
        term_new[1:d_basis-m, 1:d_basis-m] = term
        for j = d_basis-m+1:d_basis
            for i = 1:j
                # TODO: how to generalize this column slicing to MPS case?
                term_new[i, j] = dot(@view(basis.snapshots[:, i]), @view(hc.HΨ[q][:, j]))
                term_new[j, i] = term_new[i, j]'
            end
        end
        hc.ΨHΨ[q] = term_new
    end
    for (idx, term) in pairs(hc.ΨHHΨ)
        term_new = zeros(size(term) .+ m)
        term_new[1:d_basis-m, 1:d_basis-m] = term
        for j = d_basis-m+1:d_basis
            for i = 1:j
                term_new[i, j] = dot(@view(hc.HΨ[first(idx.I)][:, i]), @view(hc.HΨ[last(idx.I)][:, j]))
                term_new[j, i] = term_new[i, j]'
            end
        end
        hc.ΨHHΨ[idx] = term_new
    end

    # Transform using basis.vectors and creat AffineDecompositions
    h_new = AffineDecomposition(
        [basis.vectors' * term * basis.vectors for term in hc.ΨHΨ],
        hc.H.coefficient_map
    )
    h²_new = AffineDecomposition(
        [basis.vectors' * term * basis.vectors for term in hc.ΨHHΨ],
        μ -> (hc.H.coefficient_map(μ) * hc.H.coefficient_map(μ)')
    )

    HamiltonianCache(hc.H, hc.HΨ, hc.ΨHΨ, hc.ΨHHΨ, h_new, h²_new)
end

# Reconstruct ground state from RB eigenvector
# TODO: Change name; where to place this? -> not possible in rbasis.jl due to AffineDecomposition
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
    basis = RBasis(truth.vectors, fill(μ₁, size(truth.vectors, 2)), I, truth.vectors' * truth.vectors)
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