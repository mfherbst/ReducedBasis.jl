import DFTK # Dirty hack to use the good LOBCG from DFTK (TODO: remove when out-sourced)
import SuiteSparse

struct LOBPCG
    n_states_max::Int
    tol_degeneracy::Float64
    n_ep_extra::Int # What is this actually doing? -> rename?
    tol::Float64,
    maxiter::Int,
    λ_shift::Float64,
    verbose::Bool,
    dense_fallback::Bool,
    maxdiagonal::Bool,
    full_orthogonalize::Bool
    tol_qr::Float64
end

## Eigensolver
struct DummyInplace
    M
end
LinearAlgebra.ldiv!(d::DummyInplace, x) = x .= (d.M \ x)
prepare_preconditioner(M::SuiteSparse.CHOLMOD.Factor) = DummyInplace(M)
prepare_preconditioner(M::AbstractMatrix) = M

function diagonalify(M::SparseMatrixCSC{Tv,Ti}, maxdiagonal) where {Tv,Ti}
    I, J, V = findnz(M)
    mask = (I .> (J .- maxdiagonal)) .& (I .< (J .+ maxdiagonal))
    sparse(I[mask], J[mask], V[mask], size(M)...)
end

function preconditioner(H::AffineDecomposition, μ; shift=-0.1, maxdiagonal=4000)
    hamiltonian = diagonalify(H(μ), maxdiagonal)
    @assert ishermitian(hamiltonian)
    factorize(Hermitian(hamiltonian + shift * I))
end

"""
Diagonalize `A` using LOBPCG starting from initial guess `X0`.

Supported kwargs
----------------
 - maxiter
 - prec
 - tol
 - largest
 - n_conv_check
 - miniter
 - ortho_tol
 - display_progress
"""
diag_lobpcg(A, X0; kwargs...) = DFTK.lobpcg_hyper(A, X0; kwargs...)

## Model-specific ED adjustment
default_shift(::AbstractModel{:ED}) = -100
default_n_ep_extra(::AbstractModel{:ED}) = 5
# Some distance metric to distinguish two parameter sets
# (used to estimate which available point is best to be used as a guess for iterative solvers)
distance_parameters(::AbstractModel{:ED}, μ, ν) = norm(μ - ν)

function truth_solve(H::AffineDecomposition, μ, ψ0=nothing, solver::LOBPCG)
    if isnothing(ψ0)  # Build initial guess if needed
        ψ0 = Matrix(qr(randn(hilbert_dimension(m), n_states_max + n_ep_extra)).Q)
    else
        @assert size(ψ0, 1) == hilbert_dimension(m)
        @assert size(ψ0, 2) ≥ n_states_max + n_ep_extra
    end

    # assemble Hamiltonian for this parameter value μ
    H = hamiltonian(m, μ) + shift * I
    P = prepare_preconditioner(preconditioner(m, μ; shift, maxdiagonal))

    n_conv_check = n_states_max
    iterations = 0
    chunks = 50
    while iterations < maxiter
        res = diag_lobpcg(
            H,
            ψ0;
            n_conv_check,
            prec=P,
            tol=tol,
            maxiter=chunks,
            display_progress=verbose,
            kwargs...
        )
        iterations += res.iterations

        n_target = n_states_max
        if tol_degeneracy > 0 # Adjust n_target (e.g. in case degeneracy has been found)
            n_target = findlast(abs.(res.λ .- res.λ[n_states_max]) .< tol_degeneracy)

            # Tolerance of eigenpairs to converge fully
            conv_tol = 10maximum(res.residual_norms[n_target:min(end, n_target + 1)])
            conv_tol = clamp(conv_tol, 1e-4, 10)
            n_conv_check = findlast(abs.(res.λ .- res.λ[n_target]) .< conv_tol)
        end
        n_conv_check = max(n_target, n_conv_check)

        if n_conv_check < size(res.X, 2) && all(res.residual_norms[1:n_conv_check] .< tol)
            # All relevant eigenpairs are converged
            verbose && println("Converged in $iterations iterations.")
            return (
                vectors=res.X[:, 1:n_target],
                values=res.λ[1:n_target] .- shift,
                converged=true,
                iterations=iterations,
                X=res.X,
                λ=res.λ .- shift,
                tol,
                tol_degeneracy,
            )
        end

        if n_conv_check + n_ep_extra > size(res.X, 2)
            ψ0_extra = randn(
                ComplexF64, size(ψ0, 1), n_conv_check + n_ep_extra - size(res.X, 2)
            )
            ψ0_extra .-= ψ0 * ψ0'ψ0_extra
            ψ0_extra = Matrix(qr(ψ0_extra).Q)
            ψ0 = hcat(res.X, ψ0_extra)
        else
            ψ0 = res.X # Update guess
        end
        n_states_max = n_target
    end

    if !dense_fallback
        @warn "Accepting non-converged state ..."
        return (
            vectors=res.X[:, 1:n_target],
            values=res.λ[1:n_target] .- shift,
            converged=false,
            iterations=iterations,
            X=res.X,
            λ=res.λ .- shift,
            tol,
            tol_degeneracy,
        )
    end
    verbose && @warn "Falling back to dense diagonalisation ..."
    res = diag_full(H, ψ0; tol, kwargs...)
    n_target = n_states_max
    if tol_degeneracy > 0 # Adjust n_target (e.g. in case degeneracy has been found)
        n_target = findlast(λi -> abs(λi - res.λ[n_states_max]) < tol_degeneracy, res.λ)
    end

    (
        vectors=res.X[:, 1:n_target],
        values=res.λ[1:n_target] .- shift,
        converged=true,
        iterations=0,
        X=res.X,
        λ=res.λ .- shift,
        tol,
        tol_degeneracy,
    )
end