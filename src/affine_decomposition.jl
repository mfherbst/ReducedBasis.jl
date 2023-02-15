"""
Represents an affine decomposition

```math
O(\\bm{\\mu}) = \\sum_{r=1}^R \\alpha_r(\\bm{\\mu})\\, O_r
```

where `terms<:AbstractArray` carries the ``O_r`` matrices (or more generally linear maps)
and the `coefficients` implements the ``\\alpha_r(\\bm{\\mu})`` coefficient functions.

In the case where the ``\\alpha_r`` are constant coefficients, i.e. that don't depend on
the parameter vector, `coefficients` can also just be an `AbstractArray` of the same size
as `terms`.

Note that ``r = (r_1, \\dots, r_d)`` generally is a multi-index and as such `terms`
can be a ``d``-dimensional array. Correspondingly, `coefficients` maps parameter points
``\\bm{\\mu}`` to an `size(terms)` array.
"""
struct AffineDecomposition{T<:AbstractArray,C}
    terms::T
    coefficients::C
    function AffineDecomposition(terms::AbstractArray, coefficients)
        if !all(s -> s == size(terms[1]), size.(terms))
            error("affine terms have different dimensions")
        else
            new{typeof(terms),typeof(coefficients)}(terms, coefficients)
        end
    end
end

Base.length(ad::AffineDecomposition) = length(ad.terms[1])

Base.size(ad::AffineDecomposition) = size(ad.terms[1])

Base.size(ad::AffineDecomposition, i::Int) = size(ad.terms[1], i)

n_terms(ad::AffineDecomposition) = length(ad.terms)

# TODO: fix docs here -> use DocStringExtensions
"""
    (ad::AffineDecomposition)(μ)
    (ad::AffineDecomposition{<:AbstractArray,<:AbstractArray})()

Explicitly evaluate the affine decomposition sum at parameter point `μ` or if no parameter
dependency exists, construct the constant linear combination.
"""
(ad::AffineDecomposition)(μ) = sum(ad.coefficients(μ) .* ad.terms)

function (ad::AffineDecomposition{<:AbstractArray,<:AbstractArray})()
    sum(ad.coefficients .* ad.terms)
end

"""
    compress(ad::AffineDecomposition, basis::RBasis)

Perform the compression of an [`AffineDecomposition`](@ref)
corresponding to ``o = B^\\dagger O B``.
"""
function compress(ad::AffineDecomposition, basis::RBasis)
    matrixel = compress.(ad.terms, Ref(basis.snapshots))  # ⟨ψᵢ|O|ψⱼ⟩
    rbterms = map(m -> basis.vectors' * m * basis.vectors, matrixel)  # transform to RB
    (; rb=AffineDecomposition(rbterms, ad.coefficients),
     raw=AffineDecomposition(matrixel, ad.coefficients))
end

"""
    compress(ad::AffineDecomposition{<:Matrix,<:Function},
             basis::RBasis; symmetric_terms=false)

Perform compression for an [`AffineDecomposition`](@ref) with terms
with two indices (double-sum observables), including an option to
exploit the possible symmetry of terms ``O_{r,r'} = O_{r',r}``,
such that only the necessary compressions are computed.
"""
function compress(ad::AffineDecomposition{<:Matrix}, basis::RBasis; symmetric_terms=false)
    # TODO: replace symmetric_terms by some generalized Symmetric type in the long run
    is_nonredundant(a, b) = (size(ad.terms, 1) > size(ad.terms, 2)) ? ≥(a, b) : ≤(a, b)
    matrixel = map(CartesianIndices(ad.terms)) do idx
        if is_nonredundant(first(idx.I), last(idx.I))  # Upper/lower triangle
            return compress(ad.terms[idx], basis.snapshots)
        end
    end
    for idx in findall(isnothing, matrixel)
        if symmetric_terms
            matrixel[idx] = matrixel[last(idx.I), first(idx.I)]
        else
            matrixel[idx] = compress(ad.terms[idx], basis.snapshots)
        end
    end  # Use "symmetry" to set transposed elements
    matrixel = promote_type.(matrixel)  # Promote to common floating-point type
    rbterms = map(m -> basis.vectors' * m * basis.vectors, matrixel)
    (; rb=AffineDecomposition(rbterms, ad.coefficients),
     raw=AffineDecomposition(matrixel, ad.coefficients))
end

"""
    compress(op, snapshots::AbstractVector)

Compress one term of an [`AffineDecomposition`](@ref) `ApproxMPO` type.
"""
function compress(op, snapshots::AbstractVector)
    overlap_matrix(snapshots, map(Ψ -> op * Ψ, snapshots))
end

"""
    compress(M::AbstractMatrix, snapshots::AbstractVector{<:AbstractVector})

Compress term using matrix multiplications.
"""
function compress(op::AbstractMatrix, snapshots::AbstractVector{<:AbstractVector})
    # TODO: `hcat` memory bottleneck for large problems? -> benchmark for realistic problems
    Y = hcat(snapshots...)
    Y' * op * Y
end
