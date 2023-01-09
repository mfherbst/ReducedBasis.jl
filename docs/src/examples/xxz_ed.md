# The S=1/2 XXZ chain using exact diagonalization

In this example we want to provide a more in-depth introduction to the reduced basis framework as applied to quantum spin systems.
The reduced basis workflow roughly boils down to three steps:

1. Model setup: As a first step, we need to initialize the model Hamiltonian and the associated physical parameters.
2. Offline phase: Secondly, an assembly strategy and a truth solving method is chosen, with which we are able to generate the reduced basis surrogate.
3. Online phase: Lastly, using the surrogate, we measure observables with reduced computational cost.

Let us see, how to perform these steps using `ReducedBasis`.
As a first application, we will explore a canonical model from quantum spin physics, the *one-dimensional XXZ chain*

```math
H = \sum_{i=1}^{L-1} \big[ S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + \Delta S_i^z S_{i+1}^z \big] - \frac{h}{J} \sum_{i=1}^L S_i^z .
```

To keep things simple, we will use exact diagonalization techniques to perform the truth solves.
This means, the ``H`` will be represented by a (sparse) matrix and the snapshots by vectors of Hilbert space dimension.
Alternatively, one could also provide ``H`` in so-called matrix-product-operator form and obtain ground states of ``H`` as matrix product states.
This however is reserved for later examples.

!!! note "Simulation parameters"
    In the following we choose the simulation parameters in such a way to keep the
    computational load small. We do this to be able to automatically run all example code
    during documentation compilation.

## Model setup

To implement the XXZ Hamiltonian matrix, we first need a way to create global many-body operators

```math
S_i^\gamma = (\otimes^{i-1} I) \otimes \frac{1}{2}\sigma^\gamma \otimes (\otimes^{N-i} I) ,
```

which in this case are Spin-1/2 operators featuring the [Pauli matrices](https://en.wikipedia.org/wiki/Pauli_matrices) ``\sigma^\gamma``.
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

and create a function to make single-site operators global operators at site `i` for a many-body system of length `L`:

```@example xxz_ed; continued = true
function to_global(op::M, L::Int, i::Int) where {M<:AbstractMatrix}
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
    H1 = 0.25 * sum([to_global(σx, L, i) * to_global(σx, L, i + 1) +
                     to_global(σy, L, i) * to_global(σy, L, i + 1) for i in 1:(L-1)])
    H2 = 0.25 * sum([to_global(σz, L, i) * to_global(σz, L, i + 1) for i in 1:(L-1)])
    H3 = 0.5  * sum([to_global(σz, L, i) for i in 1:L])
    coefficient_map = μ -> [1.0, μ[1], -μ[2]]
    AffineDecomposition([H1, H2, H3], coefficient_map)
end
```

Using these function, we initialize a small system of length ``L=6`` with a Hamiltonian matrix dimension of ``2^6 = 64``:

```@example xxz_ed
L = 6
H = xxz_chain(L)
@show size(H)
```

## Offline phase

Now we can proceed by assembling the reduced basis.
To that end we first choose a solver to find the lowest eigenvectors of ``H``, which in this case is the [`LOBPCG`](@ref) solver.
Since the XXZ model as defined above harbors degenerate ground states at some parameter points, we need to target more than one eigenvector during truth solves using the `n_target` argument.
Different eigenvalues are then distinguished up some tolerance `tol_degeneracy`.
The general solver accuracy is set via the `tol` keyword argument:

```@example xxz_ed; continued = true
lobpcg = LOBPCG(; n_target=L+1, tol_degeneracy=1e-4, tol=1e-9)
```

Next, we need to restrict our surrogate to a certain domain in the ``(\Delta, h/J)`` parameter space and define a discrete grid of points on that domain.
This is achieved, for example, by defining a 2-dimensional regular grid of parameter points using [`RegularGrid`](@ref):

```@example xxz_ed; continued = true
Δ = range(-1.0, 2.5, 40)
hJ = range(0.0, 3.5, 40)
grid_train = RegularGrid(Δ, hJ)
```

For reasons of numerical stability, it is important orthogonalize the reduced basis during assembly (or use similar methods to keep the problem well-conditioned).
Hence there are different protocols to extend a reduced basis by a new snapshot.
An numerically efficient way to realize this is to use QR decomposition methods as implemented in [`QRCompress`](@ref).
Note that we choose a tolerance `tol` to discard snapshot vectors that do not significantly contribute to the basis:

```@example xxz_ed; continued = true
qrcomp = QRCompress(; tol=1e-10)
```

We lastly need to set the parameters for the greedy basis assembly by creating a [`Greedy`](@ref) object.
This includes choosing an error estimate, as well as an error tolerance below which we stop the basis assembly.
With that, we gathered all elements to be able generate the reduced basis:

```@example xxz_ed; continued = true
greedy = Greedy(; estimator=Residual(), tol=1e-3, init_from_rb=true)
basis, h, info = assemble(H, grid_train, greedy, lobpcg, qrcomp) # TODO: show output!
```

To close up the offline phase, we want to prepare an observable by compressing an [`AffineDecomposition`](@ref).
We will use the *magnetization* ``M = 2L^{-1} \sum_{i=1}^L S_i^z`` that serves as a so-called order parameter to distinguish different phases of matter in the parameter space.
Conveniently, the magnetization already is contained in the third term of ``H``:

```@example xxz_ed; continued = true
M = AffineDecomposition([H.terms[3]], μ -> [2 / L])
m = compress(M, basis)
```

And since the coefficient ``2L^{-1}`` is actually parameter-independent, we can just construct `m` at some parameter point to obtain the reduced magnetization matrix for all parameters:

```@example xxz_ed; continued = true
m_reduced = m([1])
```

## Online phase

Having assembled a reduced basis surrogate, we now want to scan in the parameter domain by measuring observables.
We have finished all Hilbert-space dimension dependent steps and only operate in the low dimensional reduced basis space.
This allows us to now compute observables on a much finer grid:

```@example xxz_ed; continued = true
Δ_online = range(first(Δ), last(Δ), 100)
hJ_online = range(first(hJ), last(hJ), 100)
grid_online = RegularGrid(Δ_online, hJ_online)
```

Instead of solving for the ground states of ``H``, we solve for the lowest eigenvectors of the reduced Hamiltonian in the online phase.
Again, we need a solver, which in this case is [`FullDiagonalization`](@ref), i.e. a wrapper around `LinearAlgebra.eigen`.

```@example xxz_ed; continued = true
fulldiag = FullDiagonalization(; n_target=L+1, tol_degeneracy=1e-4)
```

Note that we use the same degeneracy settings as we do for the offline [`LOBPCG`](@ref) solver.
To compute expectation values on all grid points, it is convenient to use a `map`:

```@example xxz_ed; continued = true
magnetization = map(grid_online) do μ
    _, φ_rb = solve(h, basis.metric, μ, fulldiag)
    sum(eachcol(φ_rb)) do u
        abs(dot(u, m_reduced, u)) / size(φ_rb, 2)
    end
end
```

Finally, we can take a look at the results.
Note that we need to transpose the `magnetization` matrix, to use the rows as the x-axis.
In addition to the magnetization, let us also plot the parameter points at which we performed truth solves in the offline stage:

```@example xxz_ed
hm = heatmap(grid_online.ranges[1], grid_online.ranges[2], magnetization';
             xlabel=raw"$\Delta$", ylabel=raw"$h/J$", title="magnetization",
             colorbar=true, clims=(0.0, 1.0), leg=false)
plot!(hm, grid_online.ranges[1], x -> 1 + x; lw=2, ls=:dash, legend=false, color=:fuchsia)
params = unique(basis.parameters)
scatter!(hm, [μ[1] for μ in params], [μ[2] for μ in params];
         markershape=:xcross, color=:springgreen, ms=3.0, msw=2.0)
```
