# Convenience struct for efficient H, H² compression
# TODO: introduce type parameters?
struct HamiltonianCache{T<:Number,V}
    H::AffineDecomposition
    HΨ::Vector{V}
    ΨHΨ::Vector{Matrix{T}}
    ΨHHΨ::Matrix{Matrix{T}}
    h::AffineDecomposition
    h²::AffineDecomposition
end
function HamiltonianCache(H::AffineDecomposition, basis::RBasis)
    HΨ = [map(Ψ -> term * Ψ, basis.snapshots) for term in H.terms]
    ΨHΨ = [overlap_matrix(basis.snapshots, v) for v in HΨ]
    ΨHHΨ = reshape(
        [overlap_matrix(v1, v2) for v1 in HΨ for v2 in HΨ], (n_terms(H), n_terms(H))
    )
    h = AffineDecomposition(
        [basis.vectors' * matel * basis.vectors for matel in ΨHΨ], H.coefficient_map
    )
    h² = AffineDecomposition(
        [basis.vectors' * matel * basis.vectors for matel in ΨHHΨ],
        μ -> (H.coefficient_map(μ) * H.coefficient_map(μ)'),
    )
    HamiltonianCache(H, HΨ, ΨHΨ, ΨHHΨ, h, h²)
end
# Compute only new HΨ and necessary matrix elements
function HamiltonianCache(hc::HamiltonianCache, basis::RBasis{V,T}) where {V,T}
    d_basis = dimension(basis)
    m = multiplicity(basis)[end]  # Multiplicity of last truth solve

    # Compute new Hamiltonian application HΨ
    HΨ = copy(hc.HΨ)  # To make function non-mutating
    for q in 1:length(HΨ)
        for j in (d_basis - m + 1):d_basis
            push!(HΨ[q], hc.H.terms[q] * basis.snapshots[j])
        end
    end

    # Compute only new matrix elements
    ΨHΨ = copy(hc.ΨHΨ)
    for (q, term) in enumerate(ΨHΨ)
        term_new = zeros(T, size(term) .+ m)
        term_new[1:(d_basis - m), 1:(d_basis - m)] = term
        for j in (d_basis - m + 1):d_basis
            for i in 1:j
                term_new[i, j] = dot(basis.snapshots[i], HΨ[q][j])
                term_new[j, i] = term_new[i, j]'
            end
        end
        ΨHΨ[q] = term_new
    end
    ΨHHΨ = copy(hc.ΨHHΨ)
    for (idx, term) in pairs(ΨHHΨ)
        term_new = zeros(T, size(term) .+ m)
        term_new[1:(d_basis - m), 1:(d_basis - m)] = term
        for j in (d_basis - m + 1):d_basis
            for i in 1:j
                term_new[i, j] = dot(HΨ[first(idx.I)][i], HΨ[last(idx.I)][j])
                term_new[j, i] = term_new[i, j]'
            end
        end
        ΨHHΨ[idx] = term_new
    end

    # Transform using basis.vectors and creat AffineDecompositions
    h = AffineDecomposition(
        [basis.vectors' * term * basis.vectors for term in ΨHΨ], hc.H.coefficient_map
    )
    h² = AffineDecomposition(
        [basis.vectors' * term * basis.vectors for term in ΨHHΨ],
        μ -> (hc.H.coefficient_map(μ) * hc.H.coefficient_map(μ)'),
    )

    HamiltonianCache(hc.H, HΨ, ΨHΨ, ΨHHΨ, h, h²)
end