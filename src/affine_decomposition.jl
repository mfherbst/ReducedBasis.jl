# Represents an affine decomposition O = ∑_i α_i(μ) M_i
struct AffineDecomposition{D,M,F<:Function}
    terms::Array{M,D}
    # Coefficient function mapping μᵢ → (αᵣ(μᵢ)) to array of size(terms)
    coefficient_map::F
end

(ad::AffineDecomposition)(μ) = sum(ad.coefficient_map(μ) .* ad.terms)
n_terms(ad::AffineDecomposition) = length(ad.terms)
function Base.size(ad::AffineDecomposition, args...)
    @assert all(s -> s == size(ad.terms[1]), size.(ad.terms)) "Affine terms have different dimensions."
    size(ad.terms[1], args...)
end

# Returns compressed/reduced observable
function compress(ad::AffineDecomposition, basis::RBasis)
    AffineDecomposition([compress(term, basis) for term in ad.terms], ad.coefficient_map)
end
# Vector-type specific compression methods
function compress(m::AbstractMatrix, basis::RBasis)
    B = hcat(basis.snapshots...) * basis.vectors
    B' * m * B
end
function compress(mpo::ApproxMPO, basis::RBasis{T,P,MPS,N}) where {T,P,N}
    matel = overlap_matrix(basis.snapshots, map(Ψ -> mpo * Ψ, basis.snapshots))
    basis.vectors' * matel * basis.vectors
end
