# Affine decompositions with multi-indices and additional parameters

In this example, we want to explore the capabilities of the central [`AffineDecomposition`](@ref) type.
To advance the previous examples where we covered the magnetization — a very simple observable that consists of only one affine term and a parameter-independent coefficient — we now turn to observables where the indices are multi-indices ``r = (r_1, \dots, r_d)`` and the coefficients can depend on additional parameters ``p``, aside from the ``\bm{\mu}`` parameter points that are present in the Hamiltonian:

```math
O(\bm{\mu}, p) = \sum_{q=1}^Q \alpha_q(\bm{\mu}, p)\, O_q
```

To stay within the realm of spin physics, we will consider the so-called *spin structure factor*

```math
\mathcal{S}(k) = \frac{1}{L} \sum_{r,r'=1}^L e^{-i (r - r') k} S^z_r S^z_{r'},\quad
\alpha_{r,r'}(k) = \frac{e^{-i (r - r') k}}{L}, \quad
O_{r,r'} =  S^z_r S^z_{r'}
```

with a wavevector parameter ``k``, to discuss the implementation of a more complicated observable.

```@example multi_ad; continued = true
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

L = 6 # hide
H = xxz_chain(L) # hide
Δ = range(-1.0, 2.5; length=40) # hide
hJ = range(0.0, 3.5; length=40) # hide
grid_train = RegularGrid(Δ, hJ) # hide
greedy = Greedy(; estimator=Residual(), tol=1e-3, init_from_rb=true) # hide
lobpcg = LOBPCG(; n_target=1, tol_degeneracy=1e-4, tol=1e-9) # hide
qrcomp = QRCompress(; tol=1e-9) # hide

info = assemble(H, grid_train, greedy, lobpcg, qrcomp) # hide
basis = info.basis; h = info.h_cache.h; # hide

Δ_online = range(first(Δ), last(Δ); length=100) # hide
hJ_online = range(first(hJ), last(hJ); length=100) # hide
grid_online = RegularGrid(Δ_online, hJ_online) # hide
fulldiag = FullDiagonalization(lobpcg) # hide
```

So let us continue the first example where we have generated an ``L=6`` XXZ surrogate `basis` with a reduced Hamiltonian `h` using an exact diagonalization solver.
Now the task is to implement the double-sum in ``\mathcal{S}``, as well as the ``k``-dependency in the coefficients.
The double-sum can be encoded by putting all ``S^z_r S^z_{r'}`` combinations into a ``L \times L`` matrix:

``` @example multi_ad; continued = true
terms = map(idx -> to_global(σz, L, first(idx.I)) * to_global(σz, L, last(idx.I)),
            CartesianIndices((1:L, 1:L)))
```

Correspondingly, the coefficient function now has to map one ``k`` value to a matrix of coefficients of the same size as the `terms` matrix:

``` @example multi_ad; continued = true
coefficient_map = k -> map(idx -> cis(-(first(idx.I) - last(idx.I)) * k) / L,
                           CartesianIndices((1:L, 1:L)))
```

One feature of the structure factor that also shows up in many other affine decompositions with double-sums is that the term indices commute, i.e. ``O_{r,r'} = O_{r',r}``.
In that case, only the upper triangular matrix has to be computed since ``B^\dagger O_{r,r'} B = B^\dagger O_{r',r} B`` are the same in the compressed affine decomposition.
So let's create the [`AffineDecomposition`](@ref) and compress, exploiting this symmetry using the `symmetric_terms` keyword argument:

``` @example multi_ad; continued = true
SFspin = AffineDecomposition(terms, coefficient_map)
sfspin, _ = compress(SFspin, basis; symmetric_terms=true)
```

In the online evaluation of the structure factor, we then need to define some wavevector values and compute the structure factor at each of them.
With the `grid_online` from before, this reads:

``` @example multi_ad; continued = true
wavevectors = [0.0, π/4, π/2, π]
sf = [zeros(size(grid_online)) for _ in 1:length(wavevectors)]
for (idx, μ) in pairs(grid_online)
    _, φ_rb = solve(h, basis.metric, μ, fulldiag)
    for (i, k) in enumerate(wavevectors)
        sf[i][idx] = sum(eachcol(φ_rb)) do u
            real(dot(u, sfspin(k), u))
        end / size(φ_rb, 2)
    end
end
```

Here we again see the convenience of measuring observables in the online stage;
adding more wavevector values does not significantly increase the computational cost, since it corresponds to a mere reevaluation of the coefficient functions and small vector-matrix products.
Finally, let us see how the structure factor behaves for the different wavevector values:

``` @example multi_ad
kwargs = (; xlabel=raw"$\Delta$", ylabel=raw"$h/J$", colorbar=true, leg=false) # hide
hms = []
for (i, q) in enumerate(wavevectors)
    push!(hms, heatmap(grid_online.ranges[1], grid_online.ranges[2], sf[i]'; 
                       title="\$k = $(round(q/π; digits=3))\\pi\$", kwargs...))
end
plot(hms...)
```

It can be nicely seen that the spin structure factor indicates the ferromagnetic phase at ``k=0`` and then moves through the magnetization plateaus until it reaches the antiferromagnetic plateau at ``k=\pi``.