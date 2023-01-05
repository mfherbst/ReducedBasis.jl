# Represents an affine decomposition O = ∑_i α_i(μ) M_i
struct AffineDecomposition{T<:AbstractArray,F<:Function}
    terms::T
    coefficient_map::F  # Coefficient mapping μᵢ → (αᵣ(μᵢ)) to array of size(terms)
    function AffineDecomposition(terms::AbstractArray, coefficient_map::Function)
        if !all(s -> s == size(terms[1]), size.(terms))
            error("affine terms have different dimensions")
        else
            new{typeof(terms),typeof(coefficient_map)}(terms, coefficient_map)
        end
    end
end

n_terms(ad::AffineDecomposition) = length(ad.terms)
Base.length(ad::AffineDecomposition) = length(ad.terms)
Base.size(ad::AffineDecomposition, args...) = size(ad.terms[1], args...)

# Construction of explicit operator at parameter point
(ad::AffineDecomposition)(μ) = sum(ad.coefficient_map(μ) .* ad.terms)

# Returns compressed/reduced observable
function compress(ad::AffineDecomposition, basis::RBasis)
    AffineDecomposition(compress.(ad.terms, Ref(basis)), ad.coefficient_map)
end

# Vector-type specific compression methods
function compress(M::AbstractMatrix, basis::RBasis)
    B = hcat(basis.snapshots...) * basis.vectors
    B' * M * B
    # TODO: which type to use here? -> problems with LOBPCG and Float64(...complex...)
    # m1 = Matrix{eltype(M)}(undef, size(M, 1), dimension(basis))
    # for i in 1:dimension(basis)
    #     m1[:, i] .= M * basis.snapshots[i]
    # end
    # m2 = Matrix{eltype(M)}(undef, dimension(basis), dimension(basis))
    # for j = 1:dimension(basis)
    #     m2[j, :] .= dropdims(basis.snapshots[j]' * m1; dims=1)
    # end
    # basis.vectors' * m2 * basis.vectors
end
