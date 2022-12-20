# T: floating-point type
# P: type of the parameter vector (e.g. SVector{3, T})
struct RBasis{T<:Number,P<:AbstractVector,V,N<:Union{AbstractMatrix{T},UniformScaling}}
    # Column-wise truth solves yᵢ at certain parameter values μᵢ
    snapshots::AbstractVector{V}
    # Parameter values μᵢ associated with truth solve yᵢ
    # Contains μᵢ m times in case of m-fold degeneracy
    parameters::Vector{P}
    # Coefficients making up the reduced basis vectors as truthsolves * vectors
    vectors::N
    # Overlaps between snapshot vectors
    snapshot_overlaps::Matrix{T}
    # Overlaps between basis vectors, equivalent to (V'*Y'*Y*V)
    metric::Matrix{T}
end

dimension(basis::RBasis) = length(basis.snapshots)
n_truthsolve(basis::RBasis) = length(unique(basis.parameters))
function multiplicity(basis::RBasis)
    return [count(x -> x == μ, basis.parameters) for μ in unique(basis.parameters)]
end

# Overlap matrix of two vectors of vecotrs: Sᵢⱼ = ⟨Ψᵢ|Ψⱼ⟩
function overlap_matrix(v1::Vector{V}, v2::Vector{V}) where {V}
    overlaps = Matrix{ComplexF64}(undef, length(v1), length(v2))
    for j in eachindex(v2), i in j:length(v1)
        overlaps[i, j] = dot(v1[i], v2[j])
        overlaps[j, i] = overlaps[i, j]' # Use symmetry of dot product
    end
    return overlaps
end

# Compute overlaps of m newest snapshots
function new_overlaps(basis::RBasis{T,P,V,N}, m::Int) where {T,P,V,N}
    d_basis = dimension(basis)
    overlaps = zeros(T, d_basis, d_basis)
    overlaps[1:(d_basis - m), 1:(d_basis - m)] = basis.snapshot_overlaps
    for j in (d_basis - m + 1):d_basis
        for i in 1:j
            overlaps[i, j] = dot(basis.snapshots[i], basis.snapshots[j])
            overlaps[j, i] = overlaps[i, j]'
        end
    end
    return overlaps
end

# Compression/orthonormalization algorithm using QR decomposition
struct QRCompress
    full_orthogonalize::Bool
    tol_qr::Float64
end
function QRCompress(; full_orthogonalize=false, tol_qr=1e-10)
    return QRCompress(full_orthogonalize, tol_qr)
end

# Compression/orthonormalization via eigenvalue decomposition (as in POD)
struct EigenDecomposition
    cutoff::Float64
end
EigenDecomposition(; cutoff=1e-6) = EigenDecomposition(cutoff)

# Extension without orthogonalization
function extend!(basis::RBasis, snapshots, μ, ::Nothing)
    append!(basis.snapshots, snapshots)
    m = length(snapshots)
    append!(basis.parameters, fill(μ, m))
    overlaps = new_overlaps(basis, m)

    return RBasis(basis.snapshots, basis.parameters, I, overlaps, overlaps), m
end
# Extension using QR decomposition for orthogonalization/compression
function extend!(basis::RBasis, snapshots, μ, qrcomp::QRCompress)
    B = hcat(basis.snapshots...)
    if qrcomp.full_orthogonalize  # QR factorization of the full basis
        fact = qr(hcat(B, snapshots...), Val(true))

        # Keep orthogonalized vectors of significant norm
        max_per_row = dropdims(maximum(abs, fact.R; dims=2); dims=2)
        d_target = findlast(max_per_row .> qrcomp.tol_qr)
        isnothing(d_target) && (return basis, d_target)
        keep = d_target - dimension(basis)

        v_norm = abs(fact.R[d_target, d_target])
        snapshots_new = [fact.Q[:, i] for i in 1:d_target]
        BᵀB = overlap_matrix(snapshots_new, snapshots_new)
        append!(basis.parameters, fill(μ, keep))
        newbasis = RBasis(snapshots_new, basis.parameters, I, BᵀB, BᵀB)
    else # Orthogonalize snapshot vectors versus basis
        Ψ = hcat(snapshots...)
        fact = qr(Ψ - B * (cholesky(basis.metric) \ (B' * Ψ)), Val(true))  # pivoted QR

        # Keep orthogonalized vectors of significant norm
        max_per_row = dropdims(maximum(abs, fact.R; dims=2); dims=2)
        keep = findlast(max_per_row .> qrcomp.tol_qr)
        isnothing(keep) && (return basis, keep)

        v = [fact.Q[:, i] for i in 1:keep]
        v_norm = abs(fact.R[keep, keep])
        newbasis, _ = extend!(basis, v, μ, nothing)
    end
    return newbasis, keep, v_norm
end
# Extension using eigenvalue decomposition for MPS RBasis
function extend!(
    basis::RBasis{T,P,MPS,N}, snapshots::Vector{MPS}, μ, ed::EigenDecomposition
) where {T,P,N}
    @assert all(length.(snapshots) .== length(basis.snapshots[1])) "MPS must have same length as column MPS"
    append!(basis.snapshots, snapshots)
    d_basis = dimension(basis)
    m = size(snapshots, 2)
    overlaps = new_overlaps(basis, m)

    # Orthonormalization via eigenvalue decomposition
    Λ, U = eigen(overlaps)
    λ_error_trunc = 0.0
    keep = 1
    if !iszero(ed.cutoff)
        U = U[:, sortperm(Λ; rev=true)]  # Sort by largest eigenvalue
        sort!(Λ; rev=true)
        λ²_psums = [sum(Λ[i:end] .^ 2) for i in eachindex(Λ)]
        λ_errors = @. sqrt(λ²_psums / λ²_psums[1])
        idx_trunc = findlast(λ_errors .> ed.cutoff)
        λ_error_trunc = λ_errors[idx_trunc]
        keep = idx_trunc - d_basis + m
        if iszero(keep)  # Return old basis, if no significant snapshots can be added
            # Remove new snapshots from MPSColumns (per-reference)
            splice!(basis.snapshots, (d_basis - m + 1):d_basis)
            return basis, keep, λ_error_trunc, minimum(Λ)
        end

        if idx_trunc != d_basis  # Truncate/compress
            Λ = Λ[1:idx_trunc]
            U = U[1:idx_trunc, 1:idx_trunc]
            splice!(basis.snapshots, (idx_trunc + 1):d_basis)
            overlaps = overlaps_new[1:idx_trunc, 1:idx_trunc]
            λ_error_trunc = λ_errors[idx_trunc + 1]
        end
    end
    append!(basis.parameters, fill(μ, keep))
    vectors_new = U * Diagonal(1 ./ sqrt.(abs.(Λ)))

    return RBasis(
        basis.snapshots,
        basis.parameters,
        vectors_new,
        overlaps,
        vectors_new' * overlaps * vectors_new,
    ),
    keep, λ_error_trunc,
    minimum(Λ)
end