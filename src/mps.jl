using ITensors

# Reconstruct MPS to Hilbert space dimensional vector
function reconstruct(mps::MPS)
    sites  = siteinds(mps)
    vec    = zeros(eltype(mps[1]), Tuple(dim(s) for s in sites))
    combos = Iterators.product(Tuple(1:dim(s) for s in sites)...)

    for c in combos
        val = ITensor(1)
        for i in eachindex(mps)
            val *= mps[i] * state(sites[i], c[i])  # Extract state from MPS at index i
        end
        vec[c...] = scalar(val)
    end
    reshape(vec, length(vec))
end

# RBasis extension using eigenvalue decomposition for MPS RBasis
function extend(basis::RBasis{MPS}, new_snapshot::Vector{MPS}, μ, ed::EigenDecomposition)
    @assert all(length.(new_snapshot) .== length(basis.snapshots[1])) "MPS must have same length as column MPS"
    overlaps = extend_overlaps(basis.snapshot_overlaps, basis.snapshots, new_snapshot)

    # Orthonormalization via eigenvalue decomposition
    Λ, U          = eigen(Hermitian(overlaps))  # Hermitian to automatically sort by smallest λ
    λ_error_trunc = 0.0
    keep          = 1
    if !iszero(ed.cutoff)
        λ²_psums      = reverse(cumsum(Λ.^2))  # Reverse to put largest eigenvector sum first
        λ²_errors     = @. sqrt(λ²_psums / λ²_psums[1])
        idx_trunc     = findlast(err -> err > ed.cutoff, λ²_errors)
        λ_error_trunc = λ²_errors[idx_trunc]
        keep          = idx_trunc - dimension(basis)
        if keep ≤ 0  # Return old basis, if no significant snapshots can be added
            return basis, keep, λ_error_trunc, minimum(Λ)
        end

        if keep != length(new_snapshot)  # Truncate/compress
            Λ = Λ[1:idx_trunc]
            U = U[1:idx_trunc, 1:idx_trunc]
            overlaps = overlaps[1:idx_trunc, 1:idx_trunc]
            λ_error_trunc = λ²_errors[idx_trunc + 1]
        end
    end
    snapshots   = append!(copy(basis.snapshots), new_snapshot[1:keep])  # TODO: use ordering of Λ
    parameters  = append!(copy(basis.parameters), fill(μ, keep))
    vectors_new = U * Diagonal(1 ./ sqrt.(abs.(Λ)))

    RBasis(snapshots, parameters, vectors_new,
           overlaps, vectors_new' * overlaps * vectors_new),
    keep, λ_error_trunc, minimum(Λ)
end

# MPO wrapper struct containing all contraction kwargs
struct ApproxMPO
    mpo::MPO
    opsum::Sum{Scaled{ComplexF64,Prod{Op}}}  # exact operator sum
    cutoff::Float64
    maxdim::Int
    mindim::Int
    truncate::Bool
end
function ApproxMPO(mpo::MPO, opsum; cutoff=1e-9, maxdim=1000, mindim=1, truncate=true)
    ApproxMPO(mpo, opsum, cutoff, maxdim, mindim, truncate)
end

function Base.:*(o::ApproxMPO, mps::MPS)
    apply(
        o.mpo, mps; cutoff=o.cutoff, maxdim=o.maxdim, mindim=o.mindim, truncate=o.truncate,
    )
end
Base.length(mpo::ApproxMPO) = length(mpo.mpo)
Base.size(mpo::ApproxMPO) = size(mpo.mpo)

# AffineDecomposition evaluation with ApproxMPO at parameter point
function (ad::AffineDecomposition{<:AbstractArray{<:ApproxMPO}})(μ)
    θ = ad.coefficient_map(μ)
    opsum = sum([c * term.opsum for (c, term) in zip(θ, ad.terms)])
    MPO(opsum, last.(siteinds(ad.terms[1].mpo)))
end

function compress(mpo::ApproxMPO, basis::RBasis{MPS})
    matel = overlap_matrix(basis.snapshots, map(Ψ -> mpo * Ψ, basis.snapshots))
    basis.vectors' * matel * basis.vectors
end

# DMRG truth solver
Base.@kwdef struct DMRG
    n_target::Int = 1
    tol_degeneracy::Float64 = 0.0
    sweeps::Sweeps = default_sweeps(; cutoff_max=1e-9, bonddim_max=1000)  # contain max. bond dimension and max. SVD cutoff
    observer::Function = () -> DMRGObserver(; energy_tol=1e-9)  # contains energy tol; is called on each solve to create <: AbstractObserver object
    verbose::Bool = false
end

function default_sweeps(; cutoff_max=1e-9, bonddim_max=1000)
    sweeps = Sweeps(100; cutoff=cutoff_max)
    setnoise!(sweeps, [10.0^n for n in -1:-2:-10]..., 0.1cutoff_max)
    setmaxdim!(sweeps, [10n for n in 1:2:10]..., bonddim_max)
    sweeps
end

function solve(H::AffineDecomposition, μ, Ψ₀::Union{Vector{MPS},Nothing}, dm::DMRG)
    if isnothing(Ψ₀)
        # last.(siteinds(...)) for two physical indices per tensor, last for non-primed index
        Ψ₀ = fill(
            randomMPS(last.(siteinds(H.terms[1].mpo)), dm.sweeps.maxdim[1]), dm.n_target,
        )
    end
    observer        = dm.observer()
    H_full          = H(μ)
    E₁, Ψ₁          = dmrg(H_full, Ψ₀[1], dm.sweeps; observer, outputlevel=0)
    values, vectors = [E₁,], [Ψ₁,]

    # Size of initial guess determines target multiplicity
    if length(Ψ₀) > 1 && dm.n_target > 1 && dm.tol_degeneracy > 0.0
        converging, n = true, 2
        while converging
            E_deg, Ψ_deg = dmrg(H_full, vectors, Ψ₀[n], dm.sweeps; observer, outputlevel=0)
            if abs(E_deg - values[end]) < dm.tol_degeneracy
                push!(vectors, Ψ_deg)
                push!(values, E_deg)
                converging = true
            else
                converging = false
            end
            (n == length(Ψ₀)) && break
            n += 1
        end
    end

    variances  = [abs(inner(H_full, Ψ, H_full, Ψ) - inner(Ψ', H_full, Ψ)^2) for Ψ in vectors]
    iterations = length(observer.energies)
    if dm.verbose
        length(vectors) > 1 &&
            println("Degenerate point μ = $μ found with m = $(length(vectors))")
        if iterations / n_target ≥ length(sweeps)
            println("Number of DMRG sweeps has reached maximum: ", iterations)
        end
    end

    (; values, vectors, variances, iterations)
end

function estimate_gs(basis::RBasis{MPS}, h::AffineDecomposition, μ, dm::DMRG, solver_online)
    _, φ_rb = solve(h, basis.metric, μ, solver_online)
    φ_trans = basis.vectors * φ_rb
    map(eachcol(φ_trans)) do col  # Add MPS and multiply by φ coefficients
        mps = col[1] * basis.snapshots[1]
        for k in 2:dimension(basis)
            mps = +(mps, col[k] * basis.snapshots[k]; maxdim=dm.sweeps.maxdim[1])
        end
        mps
    end
end
