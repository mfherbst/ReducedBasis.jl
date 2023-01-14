using ITensors

"""
    reconstruct(mps::MPS)

Explicitly compute Hilbert-space-dimensional vector by reconstructing all MPS coefficients.

!!! warning "Memory restrictions"
    The number of MPS coefficients grows exponentially with system size,
    such that the explicit reconstruction is only possible for small systems.
"""
function reconstruct(mps::MPS)
    # TODO This function is just a conversion to Vector, so it should be a vector
    #      constructor or a convert(Vector, x) routine.
    #
    # Should be upstreamed.

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

"""
Extension type for orthogonalization and compression using eigenvalue decomposition
of the basis overlap matrix. See also [`extend`](@ref).

# Fields
- `cutoff::Float64=1e-6`: cutoff for minimal eigenvalue accuracy.
"""
Base.@kwdef struct EigenDecomposition
    cutoff::Float64 = 1e-6
end

"""
    extend(basis::RBasis{MPS}, new_snapshot::Vector{MPS}, μ, ed::EigenDecomposition)

Extend the MPS reduced basis by orthonormalizing and compressing via eigenvalue decomposition.

The overlap matrix ``S`` in `basis.snapshot_overlaps` is eigenvalue decomposed
``S = U^\\dagger \\Lambda U`` and orthonormalized by computing the vector coefficients
``V = U \\Lambda^{-1/2}``. Modes with an relative squared eigenvalue error smaller than
`ed.cutoff` are dropped.
"""
function extend(basis::RBasis{MPS}, new_snapshot::Vector{MPS}, μ, ed::EigenDecomposition)
    @assert all(length.(new_snapshot) .== length(basis.snapshots[1])) "MPS must have same length as column MPS"
    overlaps = extend_overlaps(basis.snapshot_overlaps, basis.snapshots, new_snapshot)

    # Orthonormalization via eigenvalue decomposition
    Λ, U = eigen(Hermitian(overlaps))  # Hermitian to automatically sort by smallest λ
    λ_error_trunc = 0.0
    keep = 1
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

"""
Carries an `ITensors.MPO` matrix-product operator and possible truncation keyword arguments.
This enables a simple `mpo * mps` syntax while allowing for proper truncation throughout
the basis assembly.

Includes the exact operator sum in `opsum` to be able to produce efficient sums of MPOs
when constructing [`AffineDecomposition`](@ref) sums explicitly.
"""
@kwdef struct ApproxMPO
    mpo::MPO
    opsum::Sum{Scaled{ComplexF64,Prod{Op}}}
    cutoff::Float64 = 1e-9
    maxdim::Int = 1000
    mindim::Int = 1
    truncate::Bool = true
end
"""
    ApproxMPO(mpo::MPO, opsum; <keyword arguments>)
    
Construct an `ApproxMPO` with truncation default settings.

# Arguments

- `mpo::MPO`
- `opsum::Sum{Scaled{ComplexF64,Prod{Op}}}`
- `cutoff::Float64=1e-9`: relative cutoff for singular values.
- `maxdim::Int=1000`: maximal bond dimension.
- `mindim::Int=1`: minimal bond dimension.
- `truncate::Bool=true`: disables all truncate if set to `false`.
"""
function ApproxMPO(mpo::MPO, opsum; kwargs...)
    ApproxMPO(; mpo, opsum, kwargs...)
end

"""
    *(o::ApproxMPO, mps::MPS)

Apply `o.mpo` to `mps` using the truncation arguments contained in `o`.
"""
function Base.:*(o::ApproxMPO, mps::MPS)
    apply(
          # TODO If the extra args are just collected in a kwarg struct, they can just be
          #      passed through, which is more general and extensible
        o.mpo, mps; cutoff=o.cutoff, maxdim=o.maxdim, mindim=o.mindim, truncate=o.truncate,
    )
end
"""
    length(o::ApproxMPO)

Return length of `o.mpo`.
"""
Base.length(o::ApproxMPO) = length(o.mpo)
"""
    size(o::ApproxMPO)

Return size of `o.mpo`.
"""
Base.size(o::ApproxMPO) = size(o.mpo)

"""
    (ad::AffineDecomposition{<:AbstractArray{<:ApproxMPO}})(μ)

Compute sum with `ApproxMPO`s using the exact `ITensors` operator sum.
"""
function (ad::AffineDecomposition{<:AbstractArray{<:ApproxMPO}})(μ)
    θ = ad.coefficient_map(μ)
    # TODO This should work even without making a list explicitly
    opsum = sum([c * term.opsum for (c, term) in zip(θ, ad.terms)])
    MPO(opsum, last.(siteinds(ad.terms[1].mpo)))
end

"""
    compress(mpo::ApproxMPO, basis::RBasis{MPS})

Compress one term of `ApproxMPO` type.
"""
function compress(mpo::ApproxMPO, basis::RBasis{MPS})
    matel = overlap_matrix(basis.snapshots, map(Ψ -> mpo * Ψ, basis.snapshots))
    basis.vectors' * matel * basis.vectors
end

"""
Solver type for the density matrix renormalization group (DMRG) as implemented
in [`ITensors`](https://itensor.github.io/ITensors.jl/stable/DMRG.html).

# Fields

- `n_states::Int=1`: see [`FullDiagonalization`](@ref).
- `tol_degeneracy::Float64=0.0`: see [`FullDiagonalization`](@ref).
- `sweeps::Sweeps=default_sweeps(; cutoff_max=1e-9, bonddim_max=1000)`: set DMRG sweep settings via `ITensors.Sweeps`.
- `observer::Function=() -> DMRGObserver(; energy_tol=1e-9)`: set DMRG exit conditions. At each solve a new `ITensors.AbstractObserver` object is created.
- `verbose::Bool=false`: if `true`, prints info about DMRG solve.
"""
Base.@kwdef struct DMRG
    n_states::Int = 1
    tol_degeneracy::Float64 = 0.0
    sweeps::Sweeps = default_sweeps(; cutoff_max=1e-9, bonddim_max=1000)  # contain max. bond dimension and max. SVD cutoff
    observer::Function = () -> DMRGObserver(; energy_tol=1e-9)  # contains energy tol; is called on each solve to create <: AbstractObserver object
    verbose::Bool = false
end

"""
    FullDiagonalization(dm::DMRG) 

Same for [`DMRG`](@ref).
"""
function FullDiagonalization(dm::DMRG)
    # Fix n_target=1 since DMRG solver cannot target excited states yet
    FullDiagonalization(; n_target=1, tol_degeneracy=dm.tol_degeneracy)
end

"""
    default_sweeps(; cutoff_max=1e-9, bonddim_max=1000)

Return default `ITensors.Sweeps` object for DMRG solves, containing decreasing noise and increasing maximal bond dimension ramps.
"""
function default_sweeps(; cutoff_max=1e-9, bonddim_max=1000)
    sweeps = Sweeps(100; cutoff=cutoff_max)
    setnoise!(sweeps, [10.0^n for n in -1:-2:-10]..., 0.1cutoff_max)
    setmaxdim!(sweeps, [10n for n in 1:2:10]..., bonddim_max)
    sweeps
end

"""
    solve(H::AffineDecomposition, μ, Ψ₀::Union{Vector{MPS},Nothing}, dm::DMRG)

Solve using [`DMRG`](@ref). When `nothing` is provided as an initial guess,
`dm.n_states` random MPS are used.
"""
function solve(H::AffineDecomposition, μ, Ψ₀::Union{Vector{MPS},Nothing}, dm::DMRG)
    if isnothing(Ψ₀)
        # last.(siteinds(...)) for two physical indices per tensor, last for non-primed index
        Ψ₀ = fill(
            randomMPS(last.(siteinds(H.terms[1].mpo)); linkdims=dm.sweeps.maxdim[1]), dm.n_states,
        )
    end
    observer = dm.observer()
    H_full = H(μ)
    E₁, Ψ₁ = dmrg(H_full, Ψ₀[1], dm.sweeps; observer, outputlevel=0)
    values, vectors = [E₁,], [Ψ₁,]

    # Size of initial guess determines target multiplicity
    if length(Ψ₀) > 1 && dm.n_states > 1 && dm.tol_degeneracy > 0.0
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
        if iterations / dm.n_states ≥ length(sweeps)
            println("Number of DMRG sweeps has reached maximum: ", iterations)
        end
    end

    (; values, vectors, variances, iterations)
end

"""
    estimate_gs(basis::RBasis{MPS}, h::AffineDecomposition, μ, dm::DMRG, solver_online)

Compute ground state MPS from the reduced basis by MPS addition in
``| \\Phi(\\mathbf{\\mu}) \\rangle = \\sum_{k=1}^{\\dim B} [V \\varphi(\\mathbf{\\mu_k})]_k\\, | \\Psi(\\mathbf{\\mu_k}) \\rangle``.
"""
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
