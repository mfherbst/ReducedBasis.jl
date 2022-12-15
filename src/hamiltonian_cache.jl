# Convenience struct for efficient H, H² compression
# TODO: introduce type parameters
struct HamiltonianCache
    H::AffineDecomposition
    HΨ::Vector  # Hard-code to Vector for now (generalized structure even needed?)
    ΨHΨ::Vector  # Matrix elements are needed for efficient extension/truncation
    ΨHHΨ::Matrix
    h::AffineDecomposition
    h²::AffineDecomposition
end
function HamiltonianCache(H::AffineDecomposition, basis::RBasis)
    HΨ = [term * basis.snapshots for term in H.terms]
    ΨHΨ = [basis.snapshots' * v for v in HΨ]
    ΨHHΨ = reshape([v1' * v2 for v1 in HΨ for v2 in HΨ], (n_terms(H), n_terms(H)))
    h = AffineDecomposition([basis.vectors' * matel * basis.vectors for matel in ΨHΨ], H.coefficient_map)
    h² = AffineDecomposition(
        [basis.vectors' * matel * basis.vectors for matel in ΨHHΨ],
        μ -> (H.coefficient_map(μ) * H.coefficient_map(μ)')
    )
    HamiltonianCache(H, HΨ, ΨHΨ, ΨHHΨ, h, h²)
end

# Compute only new HΨ and necessary matrix elements
function extend!(hc::HamiltonianCache, basis::RBasis)
    d_basis = dimension(basis)
    m = multiplicity(basis)[end]  # Multiplicity of last truth solve
    
    # Compute new Hamiltonian application HΨ
    for (q, term) in enumerate(hc.HΨ)
        term_new = zeros(size(term, 1), size(term, 2) + m)
        term_new[:, 1:d_basis-m] = term
        term_new[:, d_basis-m+1:end] = hc.H.terms[q] * @view(basis.snapshots[:, d_basis-m+1:end])
        hc.HΨ[q] = term_new  # TODO: how to generalize application to MPS case?
    end

    # Compute only new matrix elements
    for (q, term) in enumerate(hc.ΨHΨ)
        term_new = zeros(size(term) .+ m)
        term_new[1:d_basis-m, 1:d_basis-m] = term
        for j = d_basis-m+1:d_basis
            for i = 1:j
                # TODO: how to generalize this column slicing to MPS case?
                term_new[i, j] = dot(@view(basis.snapshots[:, i]), @view(hc.HΨ[q][:, j]))
                term_new[j, i] = term_new[i, j]'
            end
        end
        hc.ΨHΨ[q] = term_new
    end
    for (idx, term) in pairs(hc.ΨHHΨ)
        term_new = zeros(size(term) .+ m)
        term_new[1:d_basis-m, 1:d_basis-m] = term
        for j = d_basis-m+1:d_basis
            for i = 1:j
                term_new[i, j] = dot(@view(hc.HΨ[first(idx.I)][:, i]), @view(hc.HΨ[last(idx.I)][:, j]))
                term_new[j, i] = term_new[i, j]'
            end
        end
        hc.ΨHHΨ[idx] = term_new
    end

    # Transform using basis.vectors and creat AffineDecompositions
    h_new = AffineDecomposition(
        [basis.vectors' * term * basis.vectors for term in hc.ΨHΨ],
        hc.H.coefficient_map
    )
    h²_new = AffineDecomposition(
        [basis.vectors' * term * basis.vectors for term in hc.ΨHHΨ],
        μ -> (hc.H.coefficient_map(μ) * hc.H.coefficient_map(μ)')
    )

    HamiltonianCache(hc.H, hc.HΨ, hc.ΨHΨ, hc.ΨHHΨ, h_new, h²_new)
end
