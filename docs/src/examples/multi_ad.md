# Affine decompositions with multi-indices and additional parameters

In this example, we want to explore the capabilities of the central [`AffineDecomposition`](@ref) type.
To advance the previous examples where we covered the magnetization — a very simple observable that consists of only one affine term and a parameter-independent coefficient — we now turn to observables where the indices are multi-indices ``r = (r_1, \dots, r_d)`` and the coefficients can depend on additional parameters ``p``, aside from the ``\mathbf{\mu}`` parameter points that are present in the Hamiltonian:

```math
O(\bm{\mu}, p) = \sum_{r=1}^R \alpha_r(\bm{\mu}, p)\, O_r
```

To stay within the realm of spin physics, we will consider the so-called *spin structure factor*

```math
\mathcal{S}(k) = \sum_{r, r'} e^{-i (r - r') k} S^z_r S^z_{r'}
```

with ``k`` wavevector parameter to discuss the implementation of a more complicated observable.

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

The double-sum can be encoded by putting all ``S^z_r S^z_{r'}`` combinations into a matrix.
Since the indices ``r`` and ``r'`` commute, that matrix will by symmetric, hence we use the special `LinearAlgebra.Symmetric` matrix type:

``` @example multi_ad; continued = true
terms = Symmetric(map(idx -> to_global(σz, L, first(idx.I)) * to_global(σz, L, last(idx.I)),
                      CartesianIndices((1:L, 1:L))))
```

Correspondingly, the coefficient function now has to map one ``k`` value to a matrix of coefficients of the same size as the `terms` matrix.
Here we can use `LinearAlgebra.Hermitian` as an abstract matrix type, since the coefficients satisfy ``\alpha_{r,r'}^* = \alpha_{r',r}``:

``` @example multi_ad; continued = true
coefficient_map = k -> Hermitian(map(idx -> cis(-(first(idx.I) - last(idx.I)) * k) / L,
                                     CartesianIndices((1:L, 1:L))))
```

Of course, one could also go without using `Symmetric` and `Hermitian`, but the advantage of using special matrix types is that the observable compression can now be performed much more efficiently.
Namely, only the upper triangular matrix has to computed since the matrix elements satisify ``(\Psi_i^\dagger O_r \Psi_j)^* = \Psi_j^\dagger O_r \Psi_i``.
Let's create the [`AffineDecomposition`](@ref) and compress:

``` @example multi_ad; continued = true
SF_zz = AffineDecomposition(terms, coefficient_map)
sf_zz = compress(SF_zz, basis)
```

In online evaluation of the structure factor, we need to first define some wavevector values and then compute the structure factor at each of them.
With the `grid_online` from before, this reads:

``` @example multi_ad; continued = true
wavevectors = [0.0, π/4, π/2, π]
sf = [zeros(size(grid_online)) for _ in 1:length(wavevectors)]
for (idx, μ) in pairs(grid_online)
    _, φ_rb = solve(h, basis.metric, μ, fulldiag)
    for (i, k) in enumerate(wavevectors)
        sf[i][idx] = sum(eachcol(φ_rb)) do u
            abs(dot(u, sf_zz(k), u))
        end / size(φ_rb, 2)
    end
end
```

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

It can be nicely seen that the spin structure factor indicates the ferromagnetic phase at ``k=0`` and continously moves through the magnetization plateaus until it reaches the antiferromagnetic plateau at ``k=\pi``.
