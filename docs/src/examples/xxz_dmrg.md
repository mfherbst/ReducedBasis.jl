# Basis assembly using DMRG

As a follow-up example, we now want to showcase how to compute an `RBasis` by means of the [density matrix renormalization group](https://tensornetwork.org/mps/algorithms/dmrg/) (DMRG).
To that end, we utilize the `ITensors.jl` library which, among other things, efficiently implements DMRG.
We will see that, while we need to adjust the way we set up the model Hamiltonian as well as our solver, most steps stay the same.

## Hamiltonians as `MPO`s

```@example xxz_dmrg; continued = true
using LinearAlgebra
using ITensors
using Plots
using ReducedBasis
```

```@example xxz_dmrg; continued = true
function xxz_chain(sites::IndexSet; kwargs...)
    xy_term   = OpSum()
    zz_term   = OpSum()
    magn_term = OpSum()
    for i in 1:(length(sites) - 1)
        xy_term   += 0.5, "S+", i, "S-", i + 1
        xy_term   += 0.5, "S-", i, "S+", i + 1
        zz_term   +=      "Sz", i, "Sz", i + 1
        magn_term +=      "Sz", i
    end
    magn_term += "Sz", length(sites)  # Add last magnetization term
    coefficient_map = μ -> [1.0, μ[1], -μ[2]]
    AffineDecomposition(
        [ApproxMPO(MPO(xy_term, sites), xy_term; kwargs...),
         ApproxMPO(MPO(zz_term, sites), zz_term; kwargs...),
         ApproxMPO(MPO(magn_term, sites), magn_term; kwargs...)],
        coefficient_map,
    )
end
```

```@example xxz_dmrg; continued = true
L = 12
sites = siteinds("S=1/2", L)
H = xxz_chain(sites; truncate=true, cutoff=1e-9)
```

## Using the [`DMRG`](@ref) solver for assembly

```@example xxz_dmrg; continued = true
dm = DMRG(;
    n_target=L+1,
    tol_degeneracy=1e-4,
    sweeps=default_sweeps(; cutoff_max=1e-9, bonddim_max=1000),
    observer=() -> DMRGObserver(; energy_tol=1e-9),
)
```

```@example xxz_dmrg; continued = true
edcomp = EigenDecomposition(; cutoff=1e-7)
```

```@example xxz_dmrg; continued = true
Δ = range(-1.0, 2.5, 40) # hide
hJ = range(0.0, 3.5, 40) # hide
grid_train = RegularGrid(Δ, hJ) # hide
greedy = Greedy(; estimator=Residual(), tol=1e-3, init_from_rb=true) # hide
```

```@example xxz_dmrg; continued = true
basis, h, info = assemble(H, grid_train, greedy, dm, edcomp)
```

```@example xxz_dmrg; continued = true
M = AffineDecomposition([H.terms[3]], μ -> [2 / L])
m = compress(M, basis)
m_reduced = m([1])
Δ_online = range(first(Δ), last(Δ), 100)
hJ_online = range(first(hJ), last(hJ), 100)
grid_online = RegularGrid(Δ_online, hJ_online)

fulldiag = FullDiagonalization(; n_target=L+1, tol_degeneracy=1e-4)
magnetization = map(grid_online) do μ
    _, φ_rb = solve(h, basis.metric, μ, fulldiag)
    sum(eachcol(φ_rb)) do u
        abs(dot(u, m_reduced, u)) / size(φ_rb, 2)
    end
end
```

```@example xxz_dmrg
hm = heatmap(grid_online.ranges[1], grid_online.ranges[2], magnetization';
             xlabel=raw"$\Delta$", ylabel=raw"$h/J$", title="magnetization ",
             colorbar=true, clims=(0.0, 1.0), leg=false)
plot!(hm, grid_online.ranges[1], x -> 1 + x; lw=2, ls=:dash, legend=false, color=:fuchsia)
params = unique(basis.parameters)
scatter!(hm, [μ[1] for μ in params], [μ[2] for μ in params];
         markershape=:xcross, color=:springgreen, ms=3.0, msw=2.0)
```
