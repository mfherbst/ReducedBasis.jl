struct DMRG
    n_target::Int
    tol_degeneracy::Float64
    sweeps::Sweeps  # contain max. bond dimension and max. SVD cutoff
    observer::Function  # contains energy tol; is called on each solve to create <: AbstractObserver object
    verbose::Bool
end
function DMRG(;
    n_target=1,
    tol_degeneracy=0.0,
    sweeps=default_sweeps(; cutoff_max=1e-9, bonddim_max=1000),
    observer=() -> default_observer(; energy_tol=1e-9),
    verbose=false,
)
    @assert !(tol_degeneracy == 0.0 && n_target > 1) "Can only target one state, if degeneracy is disabled"
    @assert !(n_target == 1 && tol_degeneracy > 0.0) "Only one state will be targetted, but degeneracy is enabled"
    return DMRG(n_target, tol_degeneracy, sweeps, observer, verbose)
end

function default_sweeps(; cutoff_max=1e-9, bonddim_max=1000)
    # TODO: optimize default settings
    sweeps = Sweeps(100; cutoff=cutoff_max)
    setnoise!(sweeps, [10.0^n for n in -1:-2:-10]..., 0.1cutoff_max)
    setmaxdim!(sweeps, [10n for n in 1:2:10]..., bonddim_max)
    return sweeps
end

default_observer(; energy_tol=1e-9) = DMRGObserver(; energy_tol)

function solve(H::AffineDecomposition, μ, Ψ₀::Union{Vector{MPS},Nothing}, dm::DMRG)
    if isnothing(Ψ₀)
        # last.(siteinds(...)) since MPOs have two physical indices per tensor, last for non-primed index
        Ψ₀ = fill(
            randomMPS(last.(siteinds(H.terms[1].mpo)), dm.sweeps.maxdim[1]), dm.n_target
        )
    end
    observer        = dm.observer()
    H_full          = H(μ)
    E₁, Ψ₁          = dmrg(H_full, Ψ₀[1], dm.sweeps; observer, outputlevel=0)
    values, vectors = Float64[E₁,], MPS[Ψ₁,]

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

    return (; values, vectors, variances, iterations)
end