"""
Super type of all error estimators.
"""
abstract type ErrorEstimate end

"""
Estimator type for the residual
``\\mathrm{Res}(\\bm{\\mu}) = \\lVert H(\\bm{\\mu}) B \\varphi(\\bm{\\mu}) - \\lambda B \\varphi(\\bm{\\mu}) \\rVert``.
"""
struct Residual <: ErrorEstimate end

"""
    estimate_error(::Residual, μ, h_cache::HamiltonianCache, basis::RBasis, sol_rb)
    
Estimate error of reduced basis using the [`Residual`](@ref) estimator.
"""
function estimate_error(::Residual, μ, h_cache::HamiltonianCache, basis::RBasis, sol_rb)
    h²_sum = h_cache.h²(μ)
    sum_of_squares = sum(zip(sol_rb.values, eachcol(sol_rb.vectors))) do (λ, φ)
        abs(φ' * h²_sum * φ - λ^2 * φ' * basis.metric * φ)
    end
    sqrt(sum_of_squares)
end

"""
Greedy reduced basis assembling strategy.

# Fields

- `estimator::ErrorEstimate`: error estimate used in greedy condition.
  See also [`estimate_error`](@ref)
- `tol::Float64=1e-3`: tolerance for error estimate, below which the assembly is terminated.
- `n_truth_max::Int=64`: maximal number of truth solves to be taken up in the basis.
- `Ψ_init::Function=rb_guess`: returns initial guess for truth solver.
  See also [`interpolate`](@ref).
- `verbose::Bool=true`: print information during assembly if `true`.
- `exit_checks::Bool=true`: if `false`, no exit checks will be performed and the assembly
  will run until `tol` or `n_truth_max` are reached.
"""
@kwdef struct Greedy
    estimator::ErrorEstimate
    tol::Float64      = 1e-3
    n_truth_max::Int  = 64
    Ψ_init::Function  = rb_guess
    verbose::Bool     = true
    exit_checks::Bool = true
end

"""
    interpolate(basis::RBasis, h::AffineDecomposition, μ, solver_online)

Compute Hilbert-space-dimensional ground state vector at parameter point `μ`
from the reduced basis using
``\\bm{\\Phi}(\\bm{\\mu}) = B \\bm{\\varphi}(\\bm{\\mu})``.
"""
function interpolate(basis::RBasis, h::AffineDecomposition, μ, solver_online)
    # TODO: benchmark for realistic problems
    _, φ_rb = solve(h, basis.metric, μ, solver_online)
    hcat(basis.snapshots...) * basis.vectors * φ_rb
end

"""
    rb_guess(info, offline_args; kwargs...)

Provide the reduced basis interpolated ground state as an initial guess for
the truth solver.
"""
function rb_guess(info, offline_args; kwargs...)
    if info.state == :start
        nothing
    else
        interpolate(info.basis, info.h_cache.h, info.μ, offline_args.solver_online;
                    kwargs...)
    end
end

"""
    random_guess(info, offline_args)

Provides a random initial guess according to the truth solver method.
"""
random_guess(info, offline_args) = nothing
# TODO: properly implement random_guess interface

"""
    assemble(H, grid, greedy, solver_truth, compressalg; <keyword arguments>)
    assemble(info, H, grid, greedy, solver_truth, compressalg; <keyword arguments>)

Assemble an `RBasis` using the greedy strategy and any truth solving method.

If `info` is not provided, start a greedy assembly from scratch.

# Arguments

- `info::NamedTuple`: iteration state from a previous simulation that will be resumed.
- `H::AffineDecomposition`: Hamiltonian for which a reduced basis is assembled.
- `grid`: parameter grid that defines the parameter domain.
- `greedy::Greedy`: greedy strategy containing assembly parameters.
  See also [`Greedy`](@ref).
- `solver_truth`: solving method for obtaining ground state snapshots.
- `compressalg`: compression method for orthogonalization, etc. See also [`extend`](@ref).
- `solver_online=FullDiagonalization(solver_truth)`: solving method that is used for the RB
  generalized eigenvalue problem.
- `callback=print_callback`: callback function that operates on the iteration state during
  assembly. It is possible to chain multiple callback functions using `∘`.
- `μ_start=grid[1]`: parameter point of the starting iteration, if no `info` is provided.
"""
function assemble(info::NamedTuple, H::AffineDecomposition, grid, greedy::Greedy, 
                  solver_truth, compressalg; callback=print_callback,
                  solver_online=FullDiagonalization(solver_truth))
    info = callback(info)
    for n in (info.iteration+1):(greedy.n_truth_max)
        # Compute residual on training grid and find maximum for greedy condition
        err_grid = similar(grid, Float64)
        λ_grid   = similar(grid, Vector{Float64})
        for (idx, μ) in pairs(grid)
            sol = solve(info.h_cache.h, info.basis.metric, μ, solver_online)
            λ_grid[idx] = sol.values
            err_grid[idx] = estimate_error(greedy.estimator, μ, info.h_cache,
                                           info.basis, sol)
        end
        err_max, idx_max = findmax(err_grid)
        μ_next = grid[idx_max]

        # Exit: μ_next has already been solved
        if greedy.exit_checks && μ_next ∈ info.basis.parameters
            greedy.verbose && @warn "μ=$μ_next has already been solved"
            break
        end

        # Construct initial guess at μ_next and run truth solve
        Ψ_init = greedy.Ψ_init((; info..., err_grid, λ_grid, err_max, μ=μ_next),
                               (; H, grid, greedy, solver_truth, compressalg,
                                solver_online))
        truth = solve(H, μ_next, Ψ_init, solver_truth)

        # Append truth vector according to solver method
        ext = extend(info.basis, truth.vectors, μ_next, compressalg)

        # Exit: ill-conditioned BᵀB
        metric_condition = cond(ext.basis.metric)
        if greedy.exit_checks && metric_condition > 1e2  # TODO: Global constant for max. condition?
            greedy.verbose &&
                @warn "stopped assembly due to ill-conditioned BᵀB" metric_condition
            break
        end

        # Exit: no vector was appended to basis
        if greedy.exit_checks && dimension(ext.basis) == dimension(info.basis)
            greedy.verbose && @warn "stopped assembly since new snapshot was insignificant"
            break
        end

        # Update basis with new snapshot/vector/metric and compute reduced terms
        h_cache = HamiltonianCache(info.h_cache, ext.basis)

        # Update iteration state info
        info_new = (; iteration=n, err_grid, λ_grid, err_max, μ=μ_next,
                    solver_info=truth, basis=ext.basis, extend_info=ext,
                    condnum=metric_condition, h=h_cache.h, h_cache,
                    cache=info.cache, state=:iterate)
        info = callback(info_new)

        # Exit iteration if error estimate drops below tolerance
        if err_max < greedy.tol
            greedy.verbose && println("reached residual target")
            break
        end
    end

    merge(info, (; state=:end))
end

function assemble(H::AffineDecomposition, grid, greedy::Greedy, solver_truth, compressalg;
                  μ_start=grid[1], callback=print_callback, kwargs...)
    info    = callback((; iteration=0, cache=(;), state=:start))
    Ψ_init  = greedy.Ψ_init(info, (; H, grid, greedy, solver_truth, compressalg, kwargs...))
    truth   = solve(H, μ_start, Ψ_init, solver_truth)
    BᵀB     = overlap_matrix(truth.vectors, truth.vectors)
    basis   = RBasis(truth.vectors, fill(μ_start, length(truth.vectors)), I, BᵀB, BᵀB)
    h_cache = HamiltonianCache(H, basis)
    info    = (; iteration=1, err_max=NaN, μ=μ_start, basis, solver_info=truth,
                h=h_cache.h, h_cache, condnum=cond(BᵀB), cache=info.cache, state=:iterate)
    assemble(info, H, grid, greedy, solver_truth, compressalg; callback, kwargs...)
end
