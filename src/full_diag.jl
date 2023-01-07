"""
Solver type for full diagonalization using [`LinearAlgebra.eigen`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigen).

# Fields

- `n_target::Int=1`: the number of targeted vectors. If `tol_degeneracy=0`, it determines the number of returned vectors per solve.
- `tol_degeneracy::Float64=0.0`: tolerance for distinguishing two eigenvalues. If `abs(λ₁ - λ₂) < tol_degeneracy`, the eigenvalues belong to the same degenerate subspace.
"""
Base.@kwdef struct FullDiagonalization
    n_target::Int = 1
    tol_degeneracy::Float64 = 0.0
end

"""
    solve(H::AffineDecomposition, μ, _, fd::FullDiagonalization)

Solve Hamiltonian for lowest eigenvalues and eigenvectors at parameter point `μ`
using  [`FullDiagonalization`](@ref).
"""
function solve(H::AffineDecomposition, μ, _, fd::FullDiagonalization)
    H_matrix = Hermitian(H(μ))  # Hamiltonian are Hermitian by assumption
    if issparse(H_matrix)  # Convert to dense matrix if sparse
        H_matrix = Hermitian(Matrix(H_matrix))
    end

    Λ, Ψ = eigen(H_matrix, 1:(fd.n_target))
    n_target = fd.n_target
    if fd.n_target > 1 && fd.tol_degeneracy > 0.0
        n_target = findlast(abs.(Λ .- Λ[1]) .< fd.tol_degeneracy)
    end

    (values=Λ[1:n_target], vectors=[Ψ[:, i] for i in 1:n_target])
end
"""
    solve(h::AffineDecomposition, b::Matrix, μ, fd::FullDiagonalization)
    
Solve the generalized eigenvalue problem
``h(\\mathbf{\\mu})\\, \\varphi(\\mathbf{\\mu}) = \\lambda(\\mathbf{\\mu})\\, b\\, \\varphi(\\mathbf{\\mu})``
at parameter point `μ` using [`FullDiagonalization`](@ref).
"""
function solve(h::AffineDecomposition, b::Matrix, μ, fd::FullDiagonalization)
    Λ, φ = eigen(Hermitian(h(μ)), Hermitian(b))
    n_target = fd.n_target
    if fd.n_target > 1 && fd.tol_degeneracy > 0.0
        n_target = findlast(abs.(Λ .- Λ[1]) .< fd.tol_degeneracy)
    end

    (values=Λ[1:n_target], vectors=φ[:, 1:n_target])
end