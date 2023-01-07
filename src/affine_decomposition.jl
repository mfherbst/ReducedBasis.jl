"""
Represents an affine decomposition

```math
O(\\mathbf{\\mu}) = \\sum_{r=1}^R \\alpha_r(\\mathbf{\\mu})\\, O_r
```

where `terms<:AbstractArray` carries the ``O_r`` matrices (or more generally linear maps)
and the `coefficient_map<:Function` implements the ``\\alpha_r(\\mathbf{\\mu})``
coefficient functions.

Note that ``r = (r_1, \\dots, r_d)`` generally is a multi-index and as such `terms`
can be a ``d``-dimensional array. Correspondingly, `coefficient_map` maps parameter points ``\\mathbf{\\mu}`` to an `size(terms)` array.
"""
struct AffineDecomposition{T<:AbstractArray,F<:Function}
    terms::T
    coefficient_map::F
    function AffineDecomposition(terms::AbstractArray, coefficient_map::Function)
        if !all(s -> s == size(terms[1]), size.(terms))
            error("affine terms have different dimensions")
        else
            new{typeof(terms),typeof(coefficient_map)}(terms, coefficient_map)
        end
    end
end

Base.length(ad::AffineDecomposition) = length(ad.terms[1])
Base.size(ad::AffineDecomposition, args...) = size(ad.terms[1], args...)
n_terms(ad::AffineDecomposition) = length(ad.terms)

"""
    (ad::AffineDecomposition)(μ)

Explicitly construct the affine decomposition sum.
"""
(ad::AffineDecomposition)(μ) = sum(ad.coefficient_map(μ) .* ad.terms)

"""
    compress(ad::AffineDecomposition, basis::RBasis)

Perform the compression of an [`AffineDecomposition`](@ref)
corresponding to ``o = B^\\dagger O B``.
"""
function compress(ad::AffineDecomposition, basis::RBasis)
    AffineDecomposition(compress.(ad.terms, Ref(basis)), ad.coefficient_map)
end
"""
    compress(M::AbstractMatrix, basis::RBasis)

Compress one term of an [`AffineDecomposition`](@ref) of `AbstractMatrix` type.
"""
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
