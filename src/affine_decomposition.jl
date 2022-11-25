# Represents an affine decomposition O = ∑_i α_i(μ) M_i
struct AffineDecomposition{D,M,F<:Function}
    terms::Array{M,D}
    # Coefficient function mapping μᵢ → (αᵣ(μᵢ)) to array of size(terms)
    coefficient_map::F
end

n_terms(ad::AffineDecomposition) = length(ad.terms)
function Base.size(ad::AffineDecomposition, args...)
    @assert all(s -> s == size(ad.terms[1]), size.(ad.terms)) "Affine terms have different dimensions."
    return size(ad.terms[1], args...)
end

# Construction of explicit operator at parameter point
(ad::AffineDecomposition)(μ) = sum(ad.coefficient_map(μ) .* ad.terms)
function (ad::AffineDecomposition{D,M,F})(μ) where {D,M<:ApproxMPO,F<:Function}
    θ     = ad.coefficient_map(μ)
    opsum = +([θ[q] * term.opsum for (q, term) in enumerate(ad.terms)]...)
    return MPO(opsum, last.(siteinds(ad.terms[1].mpo)))
end

# Returns compressed/reduced observable
function compress(ad::AffineDecomposition, basis::RBasis)
    return AffineDecomposition(
        [compress(term, basis) for term in ad.terms], ad.coefficient_map
    )
end
# Vector-type specific compression methods
function compress(m::AbstractMatrix, basis::RBasis)
    B = hcat(basis.snapshots...) * basis.vectors
    return B' * m * B
end
function compress(mpo::ApproxMPO, basis::RBasis{T,P,MPS,N}) where {T,P,N}
    matel = overlap_matrix(basis.snapshots, map(Ψ -> mpo * Ψ, basis.snapshots))
    return basis.vectors' * matel * basis.vectors
end
