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

# TODO: fix docs here
"""
    evaluate(ad::AffineDecomposition, μ)
    (ad::AffineDecomposition)(μ)

Explicitly evaluate the affine decomposition sum at parameter point `μ`.
"""
evaluate(ad::AffineDecomposition, μ) = sum(ad.coefficient_map(μ) .* ad.terms)
(ad::AffineDecomposition)(μ) = evaluate(ad, μ)

"""
    compress(ad::AffineDecomposition, basis::RBasis)

Perform the compression of an [`AffineDecomposition`](@ref)
corresponding to ``o = B^\\dagger O B``.
"""
function compress(ad::AffineDecomposition, basis::RBasis)
    AffineDecomposition(compress.(ad.terms, Ref(basis)), ad.coefficient_map)
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
    if symmetric_terms
        terms = Matrix{Matrix}(undef, size(ad.terms))
        for i = 1:size(ad.terms, 1), j = 1:size(ad.terms, 2)
            if i > j  # Exploit symmetry
                terms[i, j] = terms[j, i]
                continue
            end
            terms[i, j] = compress(ad.terms[i, j], basis)
        end
    else
        terms = compress.(ad.terms, Ref(basis))
    end
    AffineDecomposition(terms, ad.coefficient_map)
end

"""
    compress(op, basis::RBasis)

Compress one term of an [`AffineDecomposition`](@ref) `ApproxMPO` type.
"""
function compress(op, basis::RBasis)
    term = overlap_matrix(basis.snapshots, map(Ψ -> op * Ψ, basis.snapshots))
    basis.vectors' * term * basis.vectors
end

"""
    compress(M::AbstractMatrix, basis::RBasis{<:AbstractVector})

Compress term using matrix multiplications.
"""
function compress(M::AbstractMatrix, basis::RBasis{<:AbstractVector})
    # `hcat` memory bottleneck for large problems?
    # TODO: benchmark for realistic problems
    B = hcat(basis.snapshots...) * basis.vectors
    B' * M * B
end
