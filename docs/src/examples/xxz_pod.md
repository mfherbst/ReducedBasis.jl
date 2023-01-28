# Basis assembly using Proper Orthogonal Decomposition

In the last examples we have seen that we can customize the snapshot solvers as well as the compression methods during reduced basis assembly.
What we want to demonstrate in this example is that we can also use different strategies for basis assembly altogether.
In particular, we will show how to use the Proper Orthogonal Decomposition ([`POD`](@ref)) technique in the offline stage.

```@example xxz_pod; continued = true
using LinearAlgebra, SparseArrays, Plots, ReducedBasis # hide

σx = sparse([0.0 1.0; 1.0 0.0]) # hide
σy = sparse([0.0 -im; im 0.0]) # hide
σz = sparse([1.0 0.0; 0.0 -1.0]) # hide

function to_global(op::M, L::Int, i::Int) where {M<:AbstractMatrix} # hide
    d = size(op, 1) # hide

    if i == 1 # hide
        return kron(op, M(I, d^(L - 1), d^(L - 1))) # hide
    elseif i == L # hide
        return kron(M(I, d^(L - 1), d^(L - 1)), op) # hide
    else # hide
        return kron(kron(M(I, d^(i - 1), d^(i - 1)), op), M(I, d^(L - i), d^(L - i))) # hide
    end # hide
end # hide

function xxz_chain(L) # hide
    H1 = 0.25 * sum([to_global(σx, L, i) * to_global(σx, L, i + 1) + # hide
                     to_global(σy, L, i) * to_global(σy, L, i + 1) for i in 1:(L-1)]) # hide
    H2 = 0.25 * sum([to_global(σz, L, i) * to_global(σz, L, i + 1) for i in 1:(L-1)]) # hide
    H3 = 0.5  * sum([to_global(σz, L, i) for i in 1:L]) # hide
    coefficient_map = μ -> [1.0, μ[1], -μ[2]] # hide
    AffineDecomposition([H1, H2, H3], coefficient_map) # hide
end # hide
```

The conceptual difference between POD and the greedy assembly strategy is that with POD, a truth solve is performed at all parameter points in the selected grid, followed by a singular value decomposition of the snapshot matrix.
In this way, we obtain an orthogonal basis by using the singular vectors as our reduced basis.
While this procedure is less complex than the greedy strategy, it comes with the significantly increased cost of having to solve snapshots at all grid points.
Nonetheless, it can be useful to e.g. obtain a reference reduced basis and to compare against a greedy basis.

So let us stay with the example of the XXZ spin chain and initialize the Hamiltonian as before (using the functions defined in the first example) and choose a grid as well as a solver method:

```@example xxz_pod; continued = true
L = 6
H = xxz_chain(L)

Δ = range(-1.0, 2.5; length=20)
hJ = range(0.0, 3.5; length=20)
grid_train = RegularGrid(Δ, hJ)

lobpcg = LOBPCG(; n_target=1, tol_degeneracy=1e-4, tol=1e-9)
```

Notice that we now use a coarser 20 × 20 grid since we perform truth solves on all parameter points and want to keep the computational effort low.
Moreover, we are restricted to exact diagonalization solvers, since we need to explicitly construct the snapshot matrix in order to be able to perform an SVD on it.
To assemble using POD, we create a [`POD`](@ref) object where we specify the number of retained columns, i.e. singular vectors of the snapshot matrix:

```@example xxz_pod; continued = true
pod = POD(; n_vectors=24)
```

We then call `assemble` using our parameters, including `pod`, which selects the POD assembly method:

```@example xxz_pod; continued = true
info = assemble(H, grid_train, pod, lobpcg)
```

Since we do not compute any Hamiltonian compressions during POD, we need to compute them afterwards using the [`HamiltonianCache`](@ref) constructor (recall that `h` is needed in the online stage):

```@example xxz_pod; continued = true
h_cache = HamiltonianCache(H, info.basis)
basis = info.basis; h = h_cache.h;
```

```@example xxz_pod; continued = true
M = AffineDecomposition([H.terms[3]], μ -> [2 / L]) # hide
m, _ = compress(M, basis) # hide
m_reduced = m([1]) # hide

Δ_online = range(first(Δ), last(Δ); length=100) # hide
hJ_online = range(first(hJ), last(hJ); length=100) # hide
grid_online = RegularGrid(Δ_online, hJ_online) # hide
fulldiag = FullDiagonalization(lobpcg) # hide
magnetization = map(grid_online) do μ # hide
    _, φ_rb = solve(h, basis.metric, μ, fulldiag) # hide
    sum(eachcol(φ_rb)) do u # hide
        abs(dot(u, m_reduced, u)) # hide
    end / size(φ_rb, 2) # hide
end # hide
```

Again, we arrive at the online phase which is performed analogously to the first example.
We hence skip directly to the end result:

```@example xxz_pod
hm = heatmap(grid_online.ranges[1], grid_online.ranges[2], magnetization';
             xlabel=raw"$\Delta$", ylabel=raw"$h/J$", title="magnetization",
             colorbar=true, clims=(0.0, 1.0), leg=false)
plot!(hm, grid_online.ranges[1], x -> 1 + x; lw=2, ls=:dash, legend=false, color=:green)
```

The magnetization phase diagram is correctly reproduced, however this time without the parameter point markers being plotted.
This is due to the fact that all points in `grid_train` have been solved and incorporated into the reduced basis according to the POD procedure.
The number of retained singular vectors therefore does not correspond directly to snapshots at certain parameter points but to linear combinations of them.
