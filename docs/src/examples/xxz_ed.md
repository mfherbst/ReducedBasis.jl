# The S=1/2 XXZ chain using exact diagonalization

In this example we want to provide a more in-depth introduction to the reduced basis framework as applied to quantum spin systems.
The reduced basis workflow roughly boils down to three steps:

1. Model setup: As a first step, we need to initialize the model Hamiltonian and the associated physical parameters.
2. Offline phase: Secondly, an assembly strategy and a truth solving method is chosen, with which we are able to generate the reduced basis surrogate.
3. Online phase: Lastly, using the surrogate, we measure observables with reduced computational cost.

Let us see, how to perform these steps using `ReducedBasis`.
As a first application, we will explore a canonical model from quantum spin physics, the *one-dimensional XXZ chain*

```math
H = \sum_{i=1}^{L-1} \big[ S_i^x S_{i+1}^y + S_i^y S_{i+1}^z + \Delta S_i^z S_{i+1}^z \big] - \frac{h}{J} \sum_{i=1}^L S_i^z .
```

To keep things simple, we will use exact diagonalization techniques to perform the truth solves.
This means, the ``H`` will be represented by a (sparse) matrix and the snapshots by vectors of Hilbert space dimension.
Alternatively, one could also provide ``H`` in so-called matrix-product-operator form and obtain ground states of ``H`` as matrix product states.
This however is reserved for later examples.

## Model setup

To implement the XXZ Hamiltonian matrix, we first need a way to create global many-body operators

```math
S_i^\gamma = (\otimes^{i-1} I) \otimes \frac{1}{2}\sigma^\gamma \otimes (\otimes^{N-i} I) ,
```

which in this case are Spin-1/2 operators featuring the Pauli matrices

```math
\sigma^x = \begin{bmatrix}
    0 & 1 \\
    1 & 0
\end{bmatrix},\quad
\sigma^y = \begin{bmatrix}
    0 & -i \\
    -i & 0
\end{bmatrix},\quad
\sigma^z = \begin{bmatrix}
    1 & 0 \\
    0 & -1
\end{bmatrix} .
```

So, let us first define the Pauli matrices as sparse matrices

```@example xxz_ed; continued = true
using LinearAlgebra
using SparseArrays
using Plots
using ReducedBasis

σx = sparse([0.0 1.0; 1.0 0.0])
σy = sparse([0.0 -im; im 0.0])
σz = sparse([1.0 0.0; 0.0 -1.0])
```

and create a function to make single-site operators global operators at site ``i`` for a many-body system of length ``L``:

```@example xxz_ed; continued = true
function to_global(L::Int, op::M, i::Int) where {M<:AbstractMatrix}
    d = size(op, 1)

    if i == 1
        return kron(op, M(I, d^(L - 1), d^(L - 1)))
    elseif i == L
        return kron(M(I, d^(L - 1), d^(L - 1)), op)
    else
        return kron(kron(M(I, d^(i - 1), d^(i - 1)), op), M(I, d^(L - i), d^(L - i)))
    end
end
```

In the course of the reduced basis assembly, we will need to access the Hamiltonian matrix as a linear combination of matrices and parameter-dependent coefficients, which is known as an *affine decomposition*

```math
H (\mathbf{\mu}) = \sum_{r=1}^R \theta_r(\mathbf{\mu})\, H_r .
```

In our specific case, we can identify the parameter vector ``\mathbf{\mu} = (1, \Delta, h/J)`` and the coefficient function ``\mathbf{\theta}(\mathbf{\mu}) = (1, \mu_1, -\mu_2)``.
The affine decomposition is realized in the [`AffineDecomposition`](@ref) type which allows us to implement the XXZ Hamiltonian as defined above:

```@example xxz_ed; continued = true
function xxz_chain(L)
    H1 = 0.25 * sum([to_global(L, σx, i) * to_global(L, σx, i + 1) +
                     to_global(L, σy, i) * to_global(L, σy, i + 1) for i in 1:(L-1)])
    H2 = 0.25 * sum([to_global(L, σz, i) * to_global(L, σz, i + 1) for i in 1:(L-1)])
    H3 = 0.5  * sum([to_global(L, σz, i) for i in 1:L])
    coefficient_map = μ -> [1.0, μ[1], -μ[2]]
    AffineDecomposition([H1, H2, H3], coefficient_map)
end
```

## Offline phase

```@example xxz_ed; continued = true
L = 6
H = xxz_chain(L)
```

```@example xxz_ed; continued = true
lobpcg = LOBPCG(; n_target=L+1, tol_degeneracy=1e-4, tol=1e-9)
```

```@example xxz_ed; continued = true
Δ = range(-1.0, 2.5, 40)
hJ = range(0.0, 3.5, 40)
grid_train = RegularGrid(Δ, hJ)
```

```@example xxz_ed; continued = true
qrcomp = QRCompress(; tol=1e-10)
```

```@example xxz_ed
greedy = Greedy(; estimator=Residual(), tol=1e-3, init_from_rb=true)
dfbuilder = DFBuilder()
basis, h, info = assemble(
    H, grid_train, greedy, lobpcg, qrcomp; callback=dfbuilder ∘ print_callback
)
diagnostics = dfbuilder.df
```

```@example xxz_ed; continued = true
M = AffineDecomposition([H.terms[3]], μ -> [2 / L])
m = compress(M, basis)
```

## Online phase

```@example xxz_ed; continued = true
Δ_online = range(first(Δ), last(Δ), 100)
hJ_online = range(first(hJ), last(hJ), 100)
grid_online = RegularGrid(Δ_online, hJ_online)
```

```@example xxz_ed; continued = true
fulldiag = FullDiagonalization(; n_target=L+1, tol_degeneracy=1e-4)
magnetization = Matrix{Float64}(undef, size(grid_online))
m_reduced = m([1])  # Save observable, since coefficients do not depend on μ 
for (idx, μ) in pairs(grid_online)
    λ_rb, φ_rb = solve(h, basis.metric, μ, fulldiag)
    magnetization[idx] = sum(eachcol(φ_rb)) do φ
        abs(dot(φ, m_reduced, φ)) / size(φ_rb, 2)
    end
end
```

```@example xxz_ed
hm = heatmap(grid_online.ranges[1], grid_online.ranges[2], magnetization';
             xlabel=raw"$\Delta$", ylabel=raw"$h/J$", title="magnetization",
             colorbar=true, clims=(0.0, 1.0), leg=false)
plot!(hm, grid_online.ranges[1], x -> 1 + x; lw=2, ls=:dash, legend=false, color=:fuchsia)
scatter!(hm, [μ[1] for μ in diagnostics.parameter], [μ[2] for μ in diagnostics.parameter];
         markershape=:xcross, color=:springgreen, ms=3.0, msw=2.0)
```
