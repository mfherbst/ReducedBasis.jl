"""
    Base.truncate(basis::RBasis, n_truth::Int)

Truncate the [`RBasis`](@ref) to `n_truth` snapshots.

Note that `n_truth` does not amount to the dimension of the truncated basis,
but the number of truth solves included in the basis, which can feature degenerate
snapshots.
"""
function Base.truncate(basis::RBasis, n_truth::Int)
    if n_truth > n_truthsolve(basis)
        throw(ArgumentError("n_truth is larger than the basis number of truth solves"))
    end

    # Remove last snapshots and parameter vectors (non-mutating)
    idx_trunc = sum(multiplicity(basis)[1:n_truth])
    snapshots_trunc = copy(basis.snapshots)
    splice!(snapshots_trunc, (idx_trunc+1):dimension(basis))
    parameters_trunc = copy(basis.parameters)
    splice!(parameters_trunc, (idx_trunc+1):dimension(basis))

    # Adjust overlaps and vector coefficients
    snapshot_overlaps_trunc = basis.snapshot_overlaps[1:idx_trunc, 1:idx_trunc]
    if basis.vectors isa UniformScaling
        vectors_trunc = I
        metric_trunc  = basis.metric[1:idx_trunc, 1:idx_trunc]
    else
        Λ, U = eigen(Hermitian(snapshot_overlaps_trunc))
        vectors_trunc = U * Diagonal(1 ./ sqrt.(abs.(Λ)))
        metric_trunc = vectors_trunc' * snapshot_overlaps_trunc * vectors_trunc
    end

    RBasis(snapshots_trunc, parameters_trunc, vectors_trunc,
           snapshot_overlaps_trunc, metric_trunc)
end

"""
    Base.truncate(hc::HamiltonianCache, basis_trunc::RBasis)

Truncate a [`HamiltonianCache`] according to an already truncated basis.
"""
function Base.truncate(hc::HamiltonianCache, basis_trunc::RBasis)
    # Remove last Hamiltonian applications and corresponding matrix elements
    idx_trunc = dimension(basis_trunc)
    HΨ_trunc = [copy(term) for term in hc.HΨ]  # Make non-mutating
    for q in n_terms(hc.H)
        splice!(HΨ_trunc[q], (idx_trunc+1):length(HΨ_trunc[q]))
    end
    ΨHΨ_trunc = [term[1:idx_trunc, 1:idx_trunc] for term in hc.ΨHΨ]
    ΨHHΨ_trunc = [term[1:idx_trunc, 1:idx_trunc] for term in hc.ΨHHΨ]

    # Adjust reduced Hamiltonian AffineDecompositions
    h_trunc = AffineDecomposition(
        [basis_trunc.vectors' * term * basis_trunc.vectors for term in ΨHΨ_trunc],
        hc.H.coefficient_map
    )
    h²_trunc = AffineDecomposition(
        [basis_trunc.vectors' * term * basis_trunc.vectors for term in ΨHHΨ_trunc],
        μ -> (hc.H.coefficient_map(μ) * hc.H.coefficient_map(μ)')
    )

    HamiltonianCache(hc.H, HΨ_trunc, ΨHΨ_trunc, ΨHHΨ_trunc, h_trunc, h²_trunc)
end

"""
    Base.truncate(ad_raw::AffineDecomposition, basis_trunc::RBasis)
    Base.truncate(ad::AffineDecomposition,
                  basis_trunc::RBasis{V,T,P,<:UniformScaling}) where {V,T<:Number,P}

Truncate an [`AffineDecomposition`](@ref).

For general [`RBasis`](@ref), the "raw" [`AffineDecomposition`](@ref) has to be provided,
where `ad_raw.terms` are the matrix elements ``\\Psi_i^\\dagger O_r \\Psi_j``.
The returned decomposition then contains the fully compressed transformed terms, i.e.
``B^\\dagger O_r B`` using the truncated basis.

In case a basis is provided that has trivial `vectors=I`, the truncation is performed
naively on the fully compressed terms.
"""
function Base.truncate(ad_raw::AffineDecomposition, basis_trunc::RBasis)
    idx_trunc = dimension(basis_trunc)
    V = basis_trunc.vectors
    terms_trunc = [V' * term[1:idx_trunc, 1:idx_trunc] * V for term in ad_raw.terms]
    AffineDecomposition(terms_trunc, ad_raw.coefficient_map)
end

function Base.truncate(ad::AffineDecomposition,
                       basis_trunc::RBasis{V,T,P,<:UniformScaling}) where {V,T<:Number,P}
    idx_trunc = dimension(basis_trunc)
    AffineDecomposition([term[1:idx_trunc, 1:idx_trunc] for term in ad.terms],
                        ad.coefficient_map)
end