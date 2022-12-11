# Represents an affine decomposition O = ∑_i α_i(μ) M_i
struct AffineDecomposition{D,M,F<:Function}
    terms::Array{M,D}
    # Coefficient function mapping μᵢ → (αᵣ(μᵢ)) to array of size(terms)
    coefficient_map::F
end

(ad::AffineDecomposition)(μ) = sum(ad.coefficient_map(μ) .* ad.terms)
n_terms(ad::AffineDecomposition) = length(ad.terms)

# Returns compressed/reduced observable
function compress(ad::AffineDecomposition, basis::RBasis)
    AffineDecomposition([compress(basis, term) for term in ad.terms], ad.coefficient_map)
end

# General compression method
# Different methods for specific m-types (e.g. typeof(m) = Vectors{MPS} etc.)
function compress(m::AbstractMatrix, basis::RBasis)
    basis.vectors' * (basis.snapshots' * (m * basis.snapshots)) * basis.vectors
end
