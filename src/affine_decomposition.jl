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

# General compression method
# Different methods for specific m-types (e.g. typeof(m) = Vectors{MPS} etc.)
function compress(m::AbstractMatrix, basis::RBasis)
    basis.vectors' * (basis.snapshots' * (m * basis.snapshots)) * basis.vectors
end
