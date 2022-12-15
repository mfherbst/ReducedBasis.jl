struct MPSColumns{T<:Number} <: AbstractMatrix{T}
    cols::Vector{MPS}  # Matrix columns represented as MPS
    overlaps::Hermitian{T,Matrix{T}}  # ⟨Ψᵢ|Ψⱼ⟩
end
function MPSColumns(cols::Vector{MPS})
    @assert all(length.(cols) .== length(cols[1])) "All column MPS have to be of the same length"
    MPSColumns(cols, Hermitian(outer(cols, cols)))
end

# AbstractMatrix interface
Base.length(mc::MPSColumns) = length(mc.cols[1])  # Length L of MPS (same for all columns)
Base.size(mc::MPSColumns) = (prod(dim.(siteinds(mc.cols[1]))), length(mc.cols))
Base.getindex(mc::MPSColumns, i::Int) = getindex(mc.cols, i)

# Outer product of vectors of MPS for computing overlap matrix
function ITensors.outer(Ψ₁::Vector{MPS}, Ψ₂::Vector{MPS})
    overlaps = Matrix{ComplexF64}(undef, length(Ψ₁), length(Ψ₂))
    for j in eachindex(Ψ₂), i = j:length(Ψ₁)
        overlaps[i, j] = inner(Ψ₁[i], Ψ₂[j])
        overlaps[j, i] = overlaps[i, j]' # Use symmetry of dot product
    end
    overlaps
end

# Reconstruct MPS in MPSColumns to Hilbert space dimensional vectors
function reconstruct(mc::MPSColumns)
    vecs = reconstruct.(mc.cols)
    hcat(vecs...)
end
# TODO: Is this type piracy?
function reconstruct(mps::MPS)
    sites = siteinds(mps)
    vec = zeros(eltype(mps[1]), Tuple(dim(s) for s in sites))
    combos = Iterators.product(Tuple(1:dim(s) for s in sites)...)

    for c in combos
        val = ITensor(1)
        for i in eachindex(mps)
            val *= mps[i] * state(sites[i], c[i])  # Extract state from MPS at index i
        end
        vec[c...] = scalar(val)
    end
    reshape(vec, length(vec))
end

# Compression/orthonormalization via eigenvalue decomposition (as in POD)
struct EigenDecomposition
    cutoff::Float64
end
EigenDecomposition(; cutoff=1e-6) = EigenDecomposition(cutoff)

# Efficient extension of MPSColumns by one column
function extend!(
    basis::RBasis{T,P,MPSColumns,N}, mps::Vector{MPS}, μ, ed::EigenDecomposition
) where {T,P,N}
    @assert all(length.(mps) .== length(basis.snapshots)) "MPS must have same length as column MPS"
    append!(basis.snapshots.cols, mps)
    d_basis = dimension(basis)
    m = length(mps)

    # New overlaps without recalculating elements
    overlaps_new = zeros(T, d_basis, d_basis)
    overlaps_new[1:d_basis-m, 1:d_basis-m] = basis.snapshots.overlaps
    for j = d_basis-m+1:d_basis
        for i = 1:j
            overlaps_new[i, j] = inner(basis.snapshots.cols[i], basis.snapshots.cols[j])
            overlaps_new[j, i] = overlaps_new[i, j]'
        end
    end
    overlaps_new = Hermitian(overlaps_new)

    # Orthonormalization via eigenvalue decomposition
    Λ, U = eigen(overlaps_new)
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
            splice!(basis.snapshots.cols, d_basis-m+1:d_basis)
            return basis, keep, λ_error_trunc, minimum(λ)
        end

        if idx_trunc != d_basis  # Truncate/compress
            Λ = Λ[1:idx_trunc]
            U = U[1:idx_trunc, 1:idx_trunc]
            splice!(basis.snapshots.cols, idx_trunc+1:d_basis)
            overlaps_new = Hermitian(overlaps_new[1:idx_trunc, 1:idx_trunc])
            λ_error_trunc = λ_errors[idx_trunc+1]
        end
    end
    append!(basis.parameters, fill(μ, keep))
    vectors_new = U * Diagonal(1 ./ sqrt.(abs.(Λ)))

    RBasis(
        MPSColumns(basis.snapshots.cols, overlaps_new),
        basis.parameters, vectors_new, vectors_new' * overlaps_new * vectors_new
    ),
    keep, λ_error_trunc, minimum(Λ)
end