# T -> floating-point type
# P -> type of the parameter vector (e.g. SVector{3, T})
struct RBasis{T<:Number,P<:AbstractVector,M<:AbstractMatrix{T},N<:Union{AbstractMatrix{T}, UniformScaling}}
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

dim(basis::RBasis) = size(basis.snapshots, 2)
n_truthsolve(basis::RBasis) = length(unique(basis.parameters))

# Extend basis by vectors using QR compression/orthonormalization
function extend!(basis::RBasis, Ψ, μ, solver)
    B = basis.snapshots * basis.vectors
    if solver.full_orthogonalize # QR factorization of the full basis
        fact = qr(hcat(B, Ψ), Val(true))

        # Keep orthogonalized vectors of significant norm
        max_per_row = dropdims(maximum(abs, fact.R; dims=2); dims=2)
        keep = findlast(max_per_row .> solver.tol_qr)
        (keep ≤ size(basis, 2)) && (return basis, keep)

        v_norm = abs(fact.R[keep, keep])
        B_new = Matrix(fact.Q)[:, 1:keep]
    else # Orthogonalize snapshot vectors versus basis
        fact = qr(Ψ - B * (cholesky(basis.metric) \ (B' * Ψ)), Val(true))  # pivoted QR

        # Keep orthogonalized vectors of significant norm
        max_per_row = dropdims(maximum(abs, fact.R; dims=2); dims=2)
        keep = findlast(max_per_row .> solver.tol_qr)
        isnothing(keep) && (return basis, keep)

        v = Matrix(fact.Q)[:, 1:keep]
        v_norm = abs(fact.R[keep, keep])
        B_new = hcat(B, v)
    end
    push!(basis.parameters, μ)

    RBasis(B_new, basis.parameters, I, B_new' * B_new), keep, v_norm
end