# V: Snapshot vector-type of Hilbert space dimension
# T: floating-point type
# P: type of the parameter vector (e.g. SVector{3, T})
# N: matrix-type, UniformScaling, ...
struct RBasis{V,T<:Number,P,N}
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
    [count(isequal(μ), basis.parameters) for μ in basis.parameters]
end

# Overlap matrix of two vectors of vecotrs: Sᵢⱼ = ⟨Ψᵢ|Ψⱼ⟩
function overlap_matrix(v1::Vector, v2::Vector)
    @assert eltype(v1) == eltype(v2)
    # TODO: Which type for zeros to use here?
    overlaps = zeros(eltype(v1), length(v1), length(v2))
    for j in 1:length(v2), i in j:length(v1)
        overlaps[i, j] = dot(v1[i], v2[j])
        overlaps[j, i] = overlaps[i, j]' # Use symmetry of dot product
    end
    return overlaps
end

# Compute overlaps of m newest snapshots
function extend_overlaps(snapshots::Vector, overlaps_old::Matrix, m::Int)
    d_basis = length(snapshots)
    overlaps = zeros(eltype(overlaps_old), d_basis, d_basis)
    overlaps[1:(d_basis - m), 1:(d_basis - m)] = overlaps_old
    for j in (d_basis - m + 1):d_basis
        for i in 1:j
            overlaps[i, j] = dot(snapshots[i], snapshots[j])
            overlaps[j, i] = overlaps[i, j]'
        end
    end
    return overlaps
end

# Struct for using no orthogonalization
struct NoCompress end

# Compression/orthonormalization algorithm using QR decomposition
Base.@kwdef struct QRCompress
    tol::Float64 = 1e-10
end

# Compression/orthonormalization via eigenvalue decomposition (as in POD)
Base.@kwdef struct EigenDecomposition
    cutoff::Float64 = 1e-6
end

# Extension without orthogonalization
function extend!(basis::RBasis, snapshots, μ, ::NoCompress)
    append!(basis.snapshots, snapshots)
    m = length(snapshots)
    append!(basis.parameters, fill(μ, m))
    overlaps = extend_overlaps(basis.snapshots, basis.snapshot_overlaps, m)

    return RBasis(basis.snapshots, basis.parameters, I, overlaps, overlaps), m
end
# Extension using QR decomposition for orthogonalization/compression
function extend!(basis::RBasis, snapshots, μ, qrcomp::QRCompress)
    B    = hcat(basis.snapshots...)
    Ψ    = hcat(snapshots...)
    fact = qr(Ψ - B * (cholesky(basis.metric) \ (B' * Ψ)), Val(true))  # pivoted QR

    # Keep orthogonalized vectors of significant norm
    max_per_row = dropdims(maximum(abs, fact.R; dims=2); dims=2)
    keep        = findlast(max_per_row .> qrcomp.tol)
    isnothing(keep) && (return basis, keep)

    v           = [fact.Q[:, i] for i in 1:keep]
    v_norm      = abs(fact.R[keep, keep])
    newbasis, _ = extend!(basis, v, μ, nothing)
    return newbasis, keep, v_norm
end