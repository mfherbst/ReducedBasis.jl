using ITensors

"""
    reconstruct(mps::MPS)

Explicitly compute Hilbert-space-dimensional vector by reconstructing all MPS coefficients.

!!! warning "Memory restrictions"
    The number of MPS coefficients grows exponentially with system size,
    such that the explicit reconstruction is only possible for small systems.
"""
function reconstruct(mps::MPS)
    # TODO: This function is just a conversion to Vector, so it should be a vector
    #       constructor or a convert(Vector, x) routine.
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
    
Construct an `ApproxMPO` with default truncation settings.

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
          # TODO: If the extra args are just collected in a kwarg struct, they can just be
          #       passed through, which is more general and extensible
        o.mpo, mps; cutoff=o.cutoff, maxdim=o.maxdim, mindim=o.mindim, truncate=o.truncate,
    )
end

Base.length(o::ApproxMPO) = length(o.mpo)

Base.size(o::ApproxMPO) = size(o.mpo)

"""
    (ad::AffineDecomposition{<:AbstractArray{<:ApproxMPO}})(μ)

Compute sum with `ApproxMPO`s using the exact `ITensors` operator sum.
"""
function (ad::AffineDecomposition{<:AbstractArray{<:ApproxMPO}})(μ)
    θ = ad.coefficient_map(μ)
    opsum = sum(c * term.opsum for (c, term) in zip(θ, ad.terms))
    MPO(opsum, last.(siteinds(ad.terms[1].mpo)))
end

"""
Solver type for the density matrix renormalization group (DMRG) as implemented
in [`ITensors.dmrg`](https://itensor.github.io/ITensors.jl/stable/DMRG.html).

# Fields

- `n_states::Int=1`: see [`FullDiagonalization`](@ref).
- `tol_degeneracy::Float64=0.0`: see [`FullDiagonalization`](@ref).
- `sweeps::Sweeps=default_sweeps(; cutoff_max=1e-9, bonddim_max=1000)`: set DMRG sweep
  settings via `ITensors.Sweeps`.
- `observer::Function=() -> DMRGObserver(; energy_tol=1e-9)`: set DMRG exit conditions.
  At each solve a new `ITensors.AbstractObserver` object is created.
- `verbose::Bool=false`: if `true`, prints info about DMRG solve.
"""
@kwdef struct DMRG
    n_states::Int = 1
    tol_degeneracy::Float64 = 0.0
    sweeps::Sweeps = default_sweeps(; cutoff_max=1e-9, bonddim_max=1000)
    observer::Function = () -> DMRGObserver(; energy_tol=1e-9)
    verbose::Bool = false
end

"""
    FullDiagonalization(dm::DMRG) 

Construct [`FullDiagonalization`](@ref) with the same degeneracy tolerance as `dm` and
fix `n_target=1`.
"""
function FullDiagonalization(dm::DMRG)
    # Fix n_target=1 since DMRG solver cannot target excited states yet
    FullDiagonalization(; n_target=1, tol_degeneracy=dm.tol_degeneracy)
end

"""
    default_sweeps(; cutoff_max=1e-9, bonddim_max=1000)

Return default `ITensors.Sweeps` object for DMRG solves, containing decreasing noise and
increasing maximal bond dimension ramps.
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
        Ψ₀ = fill(randomMPS(last.(siteinds(H.terms[1].mpo));
                  linkdims=dm.sweeps.maxdim[1]), dm.n_states)
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
    interpolate(basis::RBasis{MPS}, h::AffineDecomposition, μ, dm::DMRG, solver_online)

Compute ground state MPS at `μ` from the reduced basis by MPS addition in
``| \\Phi(\\bm{\\mu}) \\rangle = \\sum_{k=1}^{\\dim B} [V \\varphi(\\bm{\\mu_k})]_k\\, | \\Psi(\\bm{\\mu_k}) \\rangle``.
"""
function interpolate(basis::RBasis{MPS}, h::AffineDecomposition, μ, dm::DMRG, solver_online)
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
