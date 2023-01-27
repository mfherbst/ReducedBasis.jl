"""
Represents an affine decomposition

```math
O(\\bm{\\mu}) = \\sum_{r=1}^R \\alpha_r(\\bm{\\mu})\\, O_r
```

where `terms<:AbstractArray` carries the ``O_r`` matrices (or more generally linear maps)
and the `coefficient_map<:Function` implements the ``\\alpha_r(\\bm{\\mu})`` coefficient
functions.

Note that ``r = (r_1, \\dots, r_d)`` generally is a multi-index and as such `terms`
can be a ``d``-dimensional array. Correspondingly, `coefficient_map` maps parameter points
``\\bm{\\mu}`` to an `size(terms)` array.
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

Base.size(ad::AffineDecomposition) = size(ad.terms[1])

Base.size(ad::AffineDecomposition, i::Int) = size(ad.terms[1], i)

n_terms(ad::AffineDecomposition) = length(ad.terms)

# TODO: fix docs here -> use DocStringExtensions
"""
    (ad::AffineDecomposition)(μ)

Explicitly evaluate the affine decomposition sum at parameter point `μ`.
"""
(ad::AffineDecomposition)(μ) = sum(ad.coefficient_map(μ) .* ad.terms)

"""
    compress(ad::AffineDecomposition, basis::RBasis)

Perform the compression of an [`AffineDecomposition`](@ref)
corresponding to ``o = B^\\dagger O B``.
"""
function compress(ad::AffineDecomposition, basis::RBasis)
    matrixel = compress.(ad.terms, Ref(basis.snapshots))  # ⟨ψᵢ|O|ψⱼ⟩
    rbterms = map(m -> basis.vectors' * m * basis.vectors, matrixel)  # transform to RB
    (; rb=AffineDecomposition(rbterms, ad.coefficient_map),
     raw=AffineDecomposition(matrixel, ad.coefficient_map))
end

"""
    compress(ad::AffineDecomposition{<:Matrix,<:Function},
             basis::RBasis; symmetric_terms=false)

Perform compression for an [`AffineDecomposition`](@ref) with terms
with two indices (double-sum observables), including an option to
exploit the possible symmetry of terms ``O_{r,r'} = O_{r',r}``,
such that only the necessary compressions are computed.
"""
function compress(ad::AffineDecomposition{<:Matrix,<:Function},
                  basis::RBasis; symmetric_terms=false)
    # TODO: replace symmetric_terms by some generalized Symmetric type in the long run
    if symmetric_terms
        ⪒(a, b) = (size(ad.terms, 1) > size(ad.terms, 2)) ? ≥(a, b) : ≤(a, b)
        matrixel = map(CartesianIndices(ad.terms)) do idx
            if ⪒(first(idx.I), last(idx.I))  # Upper/lower triangle
                return compress(ad.terms[idx], basis.snapshots)
            end
        end
        for idx in findall(x -> isnothing(x), matrixel)
            matrixel[idx] = matrixel[last(idx.I), first(idx.I)]
        end  # Use "symmetry" to set transposed elements
        matrixel = promote_type.(matrixel)  # Promote to common floating-point type
    else
        matrixel = compress.(ad.terms, Ref(basis.snapshots))
    end
    rbterms = map(m -> basis.vectors' * m * basis.vectors, matrixel)
    (; rb=AffineDecomposition(rbterms, ad.coefficient_map),
     raw=AffineDecomposition(matrixel, ad.coefficient_map))
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
