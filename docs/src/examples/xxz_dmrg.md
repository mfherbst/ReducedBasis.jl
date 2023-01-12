# Greedy basis assembly using DMRG

As a follow-up example, we now want to showcase how to compute an `RBasis` by means of the [density matrix renormalization group](https://tensornetwork.org/mps/algorithms/dmrg/) (DMRG).
To that end, we utilize the `ITensors.jl` library which, among other things, efficiently implements DMRG.
We will see that, while we need to adjust the way we set up the model Hamiltonian as well as our solver, most steps stay the same.
Again, we treat the one-dimensional ``S=1/2`` XXZ model from the previous example.

## Hamiltonians as `MPO`s

Let us begin by building the XXZ Hamiltonian.
Instead of constructing explicit matrices from Kronecker products as we did before, we now use a tensor format called *matrix product operators* (MPOs) to represent the Hamiltonian.
For that purpose, we first import

```@example xxz_dmrg; continued = true
using LinearAlgebra
using ITensors
using Plots
using ReducedBasis
```

now featuring `ITensors`.
To build the Hamiltonian terms as MPOs, we make use of the [`OpSum()`](https://itensor.github.io/ITensors.jl/stable/tutorials/DMRG.html) object that automatically produces an MPO from a string of operators.
The affine MPO terms are then stored in an [`AffineDecomposition`](@ref) as [`ApproxMPO`](@ref)s which also include possible truncation keyword arguments:

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

So let us instantiate such an MPO Hamiltonian where we also specify a singular value `cutoff`, which is passed to the [`ApproxMPO`](@ref) objects:

```@example xxz_dmrg; continued = true
L = 12
sites = siteinds("S=1/2", L)
H = xxz_chain(sites; cutoff=1e-9)
```

We now chose a bigger system size, since the tensor format allows for efficient low rank approximations (hence the `cutoff`) that buy us a substantial performance advantage when going to larger systems.

## Using the [`DMRG`](@ref) solver for basis assembly

Having created our Hamiltonian in MPO format, we now need a solver that is able to compute ground states from MPOs.
The corresponding ground state will also be obtained in a tensor format, namely as *matrix product states* (MPS).
This is achieved by `ITensors.dmrg` which is wrapped in the [`DMRG`](@ref) solver type:

```@example xxz_dmrg; continued = true
dm = DMRG(; n_target=1, tol_degeneracy=0.0,
          sweeps=default_sweeps(; cutoff_max=1e-9, bonddim_max=1000),
          observer=() -> DMRGObserver(; energy_tol=1e-9))
```

While the implemented DMRG solver is capable of solving degenerate ground state, we here opt for non-degenerate settings (i.e. `n_target=1` and `tol_degeneracy=0.0`), since one encounters a ``L+1``-fold degeneracy on the parameter domain, where the degenerate DMRG solver can produce instable results for larger ``L``.

As discussed in the last example, we need a way to orthogonalize the reduced basis.
Due to the MPS format that the snapshots will have, we cannot use QR decompositions anymore and resort to a different method, [`EigenDecomposition`](@ref), featuring an eigenvalue decomposition of the snapshot overlap matrix:

```@example xxz_dmrg; continued = true
edcomp = EigenDecomposition(; cutoff=1e-7)
```

```@example xxz_dmrg; continued = true
Δ = range(-1.0, 2.5, 40) # hide
hJ = range(0.0, 3.5, 40) # hide
grid_train = RegularGrid(Δ, hJ) # hide
greedy = Greedy(; estimator=Residual(), n_truth_max=22, init_from_rb=true) # hide
```

Now with different types for `H`, the solver and the orthogonalizer, we call `assemble` using the `greedy` strategy and training grid from the last example:

```@example xxz_dmrg; continued = true
basis, h, info = assemble(H, grid_train, greedy, dm, edcomp)
```

The returned `basis` now has snapshot vectors of `ITensors.MPS` type, which we have to keep in mind when we want to compress observables.
That is to say, the observables have to be constructed as `AffineDecompositions` with `ApproxMPO` terms as for the Hamiltonian.
Again, we want to compute the magnetization so that we can reuse the third term of `H`:

```@example xxz_dmrg; continued = true
M = AffineDecomposition([H.terms[3]], μ -> [2 / L])
m = compress(M, basis)
```

```@example xxz_dmrg; continued = true
m_reduced = m([1]) # hide
Δ_online = range(first(Δ), last(Δ), 100) # hide
hJ_online = range(first(hJ), last(hJ), 100) # hide
grid_online = RegularGrid(Δ_online, hJ_online) # hide

fulldiag = FullDiagonalization(; n_target=1, tol_degeneracy=0.0)  # hide
magnetization = map(grid_online) do μ # hide
    _, φ_rb = solve(h, basis.metric, μ, fulldiag)  # hide
    sum(eachcol(φ_rb)) do u # hide
        abs(dot(u, m_reduced, u)) / size(φ_rb, 2)  # hide
    end # hide
end # hide
```

And at that point, we can continue as before since we have arrived at the online phase where we only operate in the low-dimensional reduced basis space, agnostic of the previously used solver method.
In the same way as before, we perform the online calculations and arrive at the following magnetization plot:

```@example xxz_dmrg
hm = heatmap(grid_online.ranges[1], grid_online.ranges[2], magnetization';
             xlabel=raw"$\Delta$", ylabel=raw"$h/J$", title="magnetization ",
             colorbar=true, clims=(0.0, 1.0), leg=false)
plot!(hm, grid_online.ranges[1], x -> 1 + x; lw=2, ls=:dash, legend=false, color=:fuchsia)
params = unique(basis.parameters)
scatter!(hm, [μ[1] for μ in params], [μ[2] for μ in params];
         markershape=:xcross, color=:springgreen, ms=3.0, msw=2.0)
```

We reproduce the ground state phase diagram, but this time with more magnetization plateaus (due to increased system size) and we see that the greedy algorithm chose different parameter points to solve using DMRG.
