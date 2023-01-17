using DFTK  # TODO: remove when out-sourced
using SuiteSparse: SuiteSparse

# Eigensolver utility
struct DummyInplace
    M::Any
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

# See: https://github.com/JuliaMolSim/DFTK.jl/blob/master/src/eigen/diag_lobpcg_hyper.jl
diag_lobpcg(A, X0; kwargs...) = lobpcg_hyper(A, X0; kwargs...)

"""
Solver type for the Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG).
Currently uses the DFTK [`lobpcg_hyper`](https://github.com/JuliaMolSim/DFTK.jl/blob/master/src/eigen/diag_lobpcg_hyper.jl) implementation.

# Fields

- `n_target::Int=1`: see [`FullDiagonalization`](@ref).
- `tol_degeneracy::Float64=0.0`: see [`FullDiagonalization`](@ref).
- `tol::Float64=1e-9`: tolerance for residual norms.
- `maxiter::Int=300`: maximal number of LOBPCG iterations.
- `n_ep_extra::Int=4`: number of extra eigenpairs that are kept to improve convergence.
- `shift::Float64=-100`: eigenvalue shift.
- `verbose::Bool=false`: if `true`, print convergence messages.
- `dense_fallback::Bool=true`: if `false`, also non-converged states will be accepted.
  Otherwise `LinearAlgebra.eigen` is used for non-converged iterations.
- `maxdiagonal::Int=400`
"""
@kwdef struct LOBPCG
    n_target::Int = 1
    tol_degeneracy::Float64 = 0.0
    tol::Float64 = 1e-9
    maxiter::Int = 300
    n_ep_extra::Int = 4
    shift::Float64 = -100
    verbose::Bool = false
    dense_fallback::Bool = true
    maxdiagonal::Int = 400
end

"""
    FullDiagonalization(lobpcg::LOBPCG) 

Construct the [`FullDiagonalization`](@ref) solver with degeneracy settings matching
`lobpcg`.
"""
function FullDiagonalization(lobpcg::LOBPCG)
    FullDiagonalization(; n_target=lobpcg.n_target, tol_degeneracy=lobpcg.tol_degeneracy)
end

"""
    solve(H::AffineDecomposition, μ, Ψ₀::Union{Matrix,Nothing}, lobpcg::LOBPCG)

Solve using [`LOBPCG`](@ref). If `nothing` is provided as an initial guess,
an orthogonal random matrix will be used with `lobpcg.n_target + lobpcg.n_ep_extra`
column vectors.
"""
function solve(H::AffineDecomposition, μ, Ψ₀::Union{Matrix,Nothing}, lobpcg::LOBPCG)
    if isnothing(Ψ₀)  # Build initial guess if needed
        n_states = lobpcg.n_target + lobpcg.n_ep_extra
        Ψ₀ = Matrix(qr(randn(ComplexF64, size(H, 1), n_states)).Q)
    else
        @assert size(Ψ₀, 1) == size(H, 1)
        # TODO: Why is this commented out?
        # @assert size(Ψ₀, 2) ≥ lobpcg.n_target + lobpcg.n_ep_extra
        # -> interpolate(...) might not deliver such size(Ψ₀, 2); add extra columns if needed?
    end

    # Assemble Hamiltonian for parameter value μ
    H_matrix = H(μ)
    H_shifted = H_matrix + lobpcg.shift * I
    P = prepare_preconditioner(preconditioner(H_matrix; lobpcg.shift, lobpcg.maxdiagonal))

    n_conv_check = size(Ψ₀, 2)
    iterations = 0
    chunks = 50  # TODO: set as kwarg?
    n_last = lobpcg.n_target
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
            n_last = findlast(abs.(res.λ .- res.λ[lobpcg.n_target]) .< lobpcg.tol_degeneracy)

            # Tolerance of eigenpairs to converge fully
            tol_conv = 10maximum(res.residual_norms[n_last:min(end, n_last + 1)])
            tol_conv = clamp(tol_conv, 1e-4, 10)
            n_conv_check = findlast(abs.(res.λ .- res.λ[lobpcg.n_target]) .< tol_conv)
        end
        n_conv_check = max(n_last, n_conv_check)

        if n_conv_check < size(res.X, 2) &&
           all(res.residual_norms[1:n_conv_check] .< lobpcg.tol)
            # All relevant eigenpairs are converged
            lobpcg.verbose && println("converged in $iterations iterations")
            return (
                values=res.λ[1:n_last] .- lobpcg.shift,
                vectors=[res.X[:, i] for i in 1:n_last],
                converged=true,
                iterations=iterations,
                X=res.X,
                λ=res.λ .- lobpcg.shift,
            )
        end

        if n_conv_check + lobpcg.n_ep_extra > size(res.X, 2)
            Ψ₀_extra = randn(
                ComplexF64, size(Ψ₀, 1),
                n_conv_check + lobpcg.n_ep_extra - size(res.X, 2),
            )
            Ψ₀_extra .-= Ψ₀ * (Ψ₀' * Ψ₀_extra)
            Ψ₀_extra = Matrix(qr(Ψ₀_extra).Q)
            Ψ₀ = hcat(res.X, Ψ₀_extra)
        else
            Ψ₀ = res.X  # Update guess
        end
    end

    if !lobpcg.dense_fallback
        @warn "accepting non-converged state"
        return (
            values=res.λ[1:n_last] .- lobpcg.shift,
            vectors=[res.X[:, i] for i in 1:n_last],
            converged=false,
            iterations=iterations,
            X=res.X,
            λ=res.λ .- shift,
        )
    else
        lobpcg.verbose && @warn "falling back to dense diagonalization"
        val, vec = eigen(Hermitian(Matrix(H_shifted)), 1:size(Ψ₀, 2))
        n_last = lobpcg.n_target
        if lobpcg.tol_degeneracy > 0.0
            n_last = findlast(abs.(val .- val[lobpcg.n_target]) .< lobpcg.tol_degeneracy)
        end

        return (
            values=val[1:n_last] .- lobpcg.shift,
            vectors=[vec[:, i] for i in 1:n_last],
            converged=true,
            iterations=iterations,
            X=vec,
            λ=val .- lobpcg.shift,
        )
    end
end
