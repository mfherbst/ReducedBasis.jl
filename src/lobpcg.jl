import DFTK # TODO: remove when out-sourced
import SuiteSparse

# Eigensolver utility
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

function preconditioner(M::AbstractMatrix; shift=-0.1, maxdiagonal=4000)
    M_diag = diagonalify(M, maxdiagonal)
    @assert ishermitian(M_diag)
    factorize(Hermitian(M_diag + shift * I))
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

# LOBPCG parameter struct
struct LOBPCG
    n_target::Int
    tol_degeneracy::Float64
    tol::Float64
    maxiter::Int
    n_ep_extra::Int # What is this actually doing? -> rename?
    shift::Float64
    verbose::Bool
    dense_fallback::Bool
    maxdiagonal::Int
end
function LOBPCG(; n_target=1, tol_degeneracy=0.0, tol=1e-9, maxiter=300, n_ep_extra=0, shift=-100, verbose=false, dense_fallback=true, maxdiagonal=400)
    LOBPCG(n_target, tol_degeneracy, tol, maxiter, n_ep_extra, shift, verbose, dense_fallback, maxdiagonal)
end

# LOBPCG truth solve
function solve(H::AffineDecomposition, μ, Ψ₀, lobpcg::LOBPCG)
    if isnothing(Ψ₀)  # Build initial guess if needed
        Ψ₀ = Matrix(qr(randn(size(H, 1), lobpcg.n_target + lobpcg.n_ep_extra)).Q)
    else
        @assert size(Ψ₀, 1) == size(H, 1) size(Ψ₀)
        @assert size(Ψ₀, 2) ≥ lobpcg.n_target + lobpcg.n_ep_extra
    end

    # Assemble Hamiltonian for parameter value μ
    H_matrix = H(μ)
    H_shifted = H_matrix + lobpcg.shift * I
    P = prepare_preconditioner(preconditioner(H_matrix; lobpcg.shift, lobpcg.maxdiagonal))

    n_conv_check = lobpcg.n_target
    iterations = 0
    chunks = 50  # TODO: set as kwarg?
    n_target = lobpcg.n_target
    while iterations < lobpcg.maxiter
        res = diag_lobpcg(
            H_shifted,
            Ψ₀;
            n_conv_check,
            prec=P,
            tol=lobpcg.tol,
            maxiter=chunks,
            display_progress=lobpcg.verbose,
        )
        iterations += res.iterations

        if lobpcg.tol_degeneracy > 0.0
            n_target = findlast(abs.(res.λ .- res.λ[1]) .< lobpcg.tol_degeneracy)

            # Tolerance of eigenpairs to converge fully
            tol_conv = 10maximum(res.residual_norms[n_target:min(end, n_target + 1)])
            tol_conv = clamp(tol_conv, 1e-4, 10)
            n_conv_check = findlast(abs.(res.λ .- res.λ[n_target]) .< tol_conv)
        end
        n_conv_check = max(n_target, n_conv_check)

        if n_conv_check < size(res.X, 2) && all(res.residual_norms[1:n_conv_check] .< lobpcg.tol)
            # All relevant eigenpairs are converged
            lobpcg.verbose && println("Converged in $iterations iterations.")
            return (
                vectors=res.X[:, 1:n_target],
                values=res.λ[1:n_target] .- lobpcg.shift,
                converged=true,
                iterations=iterations,
                X=res.X,
                λ=res.λ .- lobpcg.shift,
            )
        end

        if n_conv_check + lobpcg.n_ep_extra > size(res.X, 2)
            Ψ₀_extra = randn(
                ComplexF64, size(Ψ₀, 1), n_conv_check + lobpcg.n_ep_extra - size(res.X, 2)
            )
            Ψ₀_extra .-= Ψ₀ * (Ψ₀' * Ψ₀_extra)
            Ψ₀_extra = Matrix(qr(Ψ₀_extra).Q)
            Ψ₀ = hcat(res.X, Ψ₀_extra)
        else
            Ψ₀ = res.X  # Update guess
        end
    end

    if !lobpcg.dense_fallback
        @warn "Accepting non-converged state."
        return (
            vectors=res.X[:, 1:n_target],
            values=res.λ[1:n_target] .- lobpcg.shift,
            converged=false,
            iterations=iterations,
            X=res.X,
            λ=res.λ .- shift,
        )
    else
        lobpcg.verbose && @warn "Falling back to dense diagonalization."
        val, vec = eigen(Hermitian(Matrix(H_shifted)), 1:size(Ψ₀, 2))
        n_target = n_target
        if lobpcg.tol_degeneracy > 0.0
            n_target = findlast(λ -> abs(λ - val[1]) < lobpcg.tol_degeneracy, val)
        end

        return (
            vectors=vec[:, 1:n_target],
            values=val[1:n_target] .- lobpcg.shift,
            converged=true,
            # iterations=0,  # Still return original number of iterations?
            iterations=iterations,
            X=vec,
            λ=val .- lobpcg.shift,
        )
    end
end