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
    [count(isequal(μ), basis.parameters) for μ in unique(basis.parameters)]
end

# Overlap matrix of two vectors of vecotrs: Sᵢⱼ = ⟨Ψᵢ|Ψⱼ⟩
function overlap_matrix(v1::Vector, v2::Vector)
    @assert eltype(v1) == eltype(v2)
    # TODO: Which type for zeros to use here? (defaults to Float64)
    T = typeof(dot(v1[1], v2[1]))  # TODO: temporary -> replace by map?
    overlaps = zeros(T, length(v1), length(v2))
    for j in 1:length(v2), i in j:length(v1)
        overlaps[i, j] = dot(v1[i], v2[j])
        overlaps[j, i] = overlaps[i, j]'  # Use symmetry of dot product
    end
    overlaps
end

# Compute overlaps of m newest snapshots
function extend_overlaps(old_overlaps::Matrix, old_snapshots::Vector, new_snapshot::Vector)
    d_old = length(old_snapshots)
    m = length(new_snapshot)  # Multiplicity of new snapshot
    overlaps = zeros(eltype(old_overlaps), d_old + m, d_old + m)
    overlaps[1:d_old, 1:d_old] = old_overlaps
    for (j, Ψ_new) in enumerate(new_snapshot)
        for (i, Ψ_old) in enumerate(old_snapshots)
            overlaps[i, d_old+j] = dot(Ψ_old, Ψ_new)
            overlaps[d_old+j, i] = overlaps[i, d_old+j]'
        end
        for (i, Ψ_new′) in enumerate(new_snapshot)
            overlaps[d_old+i, d_old+j] = dot(Ψ_new′, Ψ_new)
            overlaps[d_old+j, d_old+i] = overlaps[d_old+i, d_old+j]'
        end
    end
    overlaps
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
function extend(basis::RBasis, new_snapshot, μ, ::NoCompress)
    overlaps = extend_overlaps(basis.snapshot_overlaps, basis.snapshots, new_snapshot)
    append!(basis.snapshots, new_snapshot)
    append!(basis.parameters, fill(μ, length(new_snapshot)))

    RBasis(basis.snapshots, basis.parameters, I, overlaps, overlaps), length(new_snapshot)
end
# Extension using QR decomposition for orthogonalization/compression
function extend(basis::RBasis, new_snapshot, μ, qrcomp::QRCompress)
    B    = hcat(basis.snapshots...)
    Ψ    = hcat(new_snapshot...)
    fact = qr(Ψ - B * (cholesky(basis.metric) \ (B' * Ψ)), Val(true))  # pivoted QR

    # Keep orthogonalized vectors of significant norm
    max_per_row = dropdims(maximum(abs, fact.R; dims=2); dims=2)
    keep        = findlast(max_per_row .> qrcomp.tol)
    isnothing(keep) && (return basis, keep)

    v          = [fact.Q[:, i] for i in 1:keep]
    v_norm     = abs(fact.R[keep, keep])
    new_basis, = extend(basis, v, μ, NoCompress())

    new_basis, keep, v_norm
end