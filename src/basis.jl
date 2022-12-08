# T -> floating-point type
# P -> type of the parameter vector (e.g. SVector{3, T})
struct ReducedBasis{T<:AbstractFloat,P<:AbstractVector,M<:AbstractMatrix{T},N<:AbstractMatrix{T}}
    # Column-wise truth solves yᵢ at certain parameter values μᵢ
    snapshots::M  # = y
    # Parameter values μᵢ associated with truth solve yᵢ
    # Contains μᵢ m times in case of m-fold degeneracy
    parameters::Vector{P}
    # Coefficients making up the reduced basis vectors as truthsolves * vectors
    vectors::N  # = V
    # Overlap between basis vectors, equivalent to (V'*Y'*Y*V)
    metric::Matrix{T}  # = V' * Y' * Y * V
end

dim(basis::ReducedBasis) = size(basis.snapshots, 2)
n_truthsolve(basis::ReducedBasis) = length(unique(basis.parameters))

# Reconstruct ground state from RB eigenvector
function reconstruct(basis::ReducedBasis, h::AffineDecomposition, μ)
    _, φ_rb = eigen(h, basis.metric, μ)
    basis.snapshots * basis.vectors * φ_rb
end

# Extend basis by vectors using QR compression/orthonormalization
function extend(basis::ReducedBasis, Ψ, solver)
    B = basis.snapshots * basis.vectors
    if solver.full_orthogonalize # QR factorization of the full basis
        fact = qr(hcat(basis, Ψ), Val(true))

        # Keep orthogonalized vectors of significant norm
        max_per_row = dropdims(maximum(abs, fact.R; dims=2); dims=2)
        keep = findlast(max_per_row .> solver.tol_qr)
        (keep ≤ size(basis, 2)) && (return basis)

        v_norm = abs(fact.R[keep, keep])
        newbasis = Matrix(fact.Q)[:, 1:keep]
    else # Orthogonalize snapshot vectors versus basis
        fact = qr(Ψ - B * (cholesky(basis.metric) \ (B' * Ψ)), Val(true))  # pivoted QR

        # Keep orthogonalized vectors of significant norm
        max_per_row = dropdims(maximum(abs, fact.R; dims=2); dims=2)
        keep = findlast(max_per_row .> solver.tol_qr)
        isnothing(keep) && (return basis)

        v = Matrix(fact.Q)[:, 1:keep]
        v_norm = abs(fact.R[keep, keep])
        newbasis = hcat(basis, v)
    end

    newbasis, keep, v_norm
end