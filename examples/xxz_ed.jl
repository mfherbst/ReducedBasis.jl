# # The reduced basis workflow
#
# In this first example we want to provide an introduction to the RB framework
# as applied to quantum spin systems. We want to see, from start to finish, how to set up
# a physical model, how to generate a surrogate basis and how to finally compute observable
# quantities. For that purpose, we cover the three basic steps of the RB workflow:
#
# 1. Model setup: We first need to initialize the model Hamiltonian and the associated physical parameters.
# 2. Offline phase: An assembly strategy and a truth solving method is chosen, with which we generate the RB surrogate and we prepare observables for later measurement.
# 3. Online phase: Using the surrogate, we measure observables with reduced computational cost.
#
# Let us see, how to perform these steps using ReducedBasis.jl.
# As a first application, we will explore a canonical model from quantum spin physics,
# the *one-dimensional spin-1/2 XXZ chain*
#
# ```math
# H = \sum_{i=1}^{L-1} \big[ S_i^x S_{i+1}^x + S_i^y S_{i+1}^y +
# \Delta S_i^z S_{i+1}^z \big] - \frac{h}{J} \sum_{i=1}^L S_i^z .
# ```
#
# To assemble the basis, a greedy algorithm will be used that tries to select as few
# snapshots as possible to generate a good surrogate. And to keep things simple, we will
# utilize exact diagonalization techniques to perform the eigenvalue solves to obtain
# snapshots at the desired parameter points. This means, ``H`` will be represented by a
# (sparse) matrix and the snapshots by vectors of Hilbert space dimension. Alternatively,
# one could, e.g., provide ``H`` and its ground states in a tensor-based format allowing for
# low-rank approximations, which is reserved for a later example.
#
# !!! note "Simulation parameters"
#     In the following we choose the simulation parameters in such a way as to keep the
#     computational load small. We do this to be able to automatically run all example code
#     during documentation compilation. As a result, the physical results are characteristic
#     for small systems, i.e., they exhibit strong finite-size effects.
#
# ## Model setup
#
# Let us first set up the parametrized Hamiltonian matrix. In this specific example, we will
# need some utility functions to generate many-body spin Hamiltonians, but in other
# applications, possibly not connected to physics, the Hamiltonian setup will of course
# differ. However, all parametrized Hamiltonians will need to be cast into the form of an
# *affine decomposition*
#
# ```math
# H (\bm{\mu}) = \sum_{q=1}^Q \theta_q(\bm{\mu})\, H_q .
# ```
#
# The corresponding type in ReducedBasis is the [`AffineDecomposition`](@ref) which,
# as we will see, will account for both Hamiltonians and other observables that one
# would want to measure.
# 
# Now coming to the XXZ chain, we want to implement the parametrized Hamiltonian matrix,
# for which we first need a way to create global many-body operators
#
# ```math
# S_i^\gamma = (\otimes^{i-1} I) \otimes \frac{1}{2}\sigma^\gamma \otimes (\otimes^{N-i} I),
# ```
#
# which in this case are ``S=1/2`` operators featuring the
# [Pauli matrices](https://en.wikipedia.org/wiki/Pauli_matrices) ``\sigma^\gamma``.
# So, let us first define the Pauli matrices as sparse matrices

using LinearAlgebra
using SparseArrays
using ReducedBasis

σx = sparse([0.0 1.0; 1.0 0.0])
σy = sparse([0.0 -im; im 0.0])
σz = sparse([1.0 0.0; 0.0 -1.0]);

# and create a function to promote single-site operators to global operators at site `i` for a
# many-body system of length `L`:

function to_global(op::M, L::Int, i::Int) where {M<:AbstractMatrix}
    d = size(op, 1)
    if i == 1
        kron(op, M(I, d^(L - 1), d^(L - 1)))
    elseif i == L
        kron(M(I, d^(L - 1), d^(L - 1)), op)
    else
        kron(kron(M(I, d^(i - 1), d^(i - 1)), op), M(I, d^(L - i), d^(L - i)))
    end
end;

# To be able to create an [`AffineDecomposition`](@ref), we first need to identify the terms
# ``H_q`` and the coefficient functions ``\theta_q(\bm{\mu})``. In our specific case, we can
# identify the parameter vector ``\bm{\mu} = (\Delta, h/J)`` and the associated coefficient
# function as ``\bm{\theta}(\bm{\mu}) = (1, \mu_1, -\mu_2)``. Note that the first entry
# ``\theta_1 = 1`` since the first two summands of ``H`` do not have any non-trivial
# coefficients. Hence we arrive at the following Hamiltonian implementation:

function xxz_chain(L)
    H1 = 0.25 * sum(1:L-1) do i
        to_global(σx, L, i) * to_global(σx, L, i + 1) +
        to_global(σy, L, i) * to_global(σy, L, i + 1)
    end
    H2 = 0.25 * sum(1:L-1) do i
        to_global(σz, L, i) * to_global(σz, L, i + 1)
    end
    H3 = 0.5  * sum(1:L) do i
        to_global(σz, L, i)
    end
    AffineDecomposition([H1, H2, H3], μ -> [1.0, μ[1], -μ[2]])
end;

# Using these functions, we initialize a small system of length ``L=6`` with a matrix
# dimension of ``2^6 = 64``:

L = 6
H = xxz_chain(L);

# ## Offline phase
#
# Now we can proceed by assembling the reduced basis. To that end we first choose a solver
# to find the lowest eigenvectors of ``H``, for which in this case we use the
# [`LOBPCG`](@ref) solver. Since the XXZ model as defined above harbors degenerate ground
# states at some parameter points, we need to choose the right solver settings to account
# for that. To obtain the ground-state subspace, with different eigenvalues are being
# distinguished up to some tolerance `tol_degeneracy`, we set:

lobpcg = LOBPCG(; tol_degeneracy=1e-4);

# Next, we need to restrict our surrogate to a certain domain in the ``(\Delta, h/J)``
# parameter space and define a discrete grid of points on that domain. This is achieved,
# for example, by defining a 2-dimensional regular grid of parameter points using
# [`RegularGrid`](@ref):

Δ = range(-1.0, 2.5; length=40)
hJ = range(0.0, 3.5; length=40)
grid_train = RegularGrid(Δ, hJ);

# !!! note "Choice of training grid"
#     While in many cases the choice of the training grid does not have a large impact on
#     the greedy assembly, as long as the resolution is high enough to resolve the physical
#     features underlying the Hamiltonian, there are some cases where special care is
#     required. In particular, the initial parameter point that is used to assemble the
#     basis (see [`assemble`](@ref)) can make crucial differences, when the Hamiltonian
#     possesses conserved quantities. For instance, for the XXZ chain, we need to choose
#     ``\bm\mu = (-1,0)`` since ``H`` has plateaus of conserved magnetization (see below),
#     and any other choice would lock the basis to one particular magnetization sector.
#     As this is the first parameter point in `grid_train` anyway, we do not need to
#     further specify the initial parameter point below.
#
# For reasons of numerical stability, it is important orthogonalize the RB during
# assembly (or use similar methods to keep the problem well-conditioned). Hence there are
# different protocols to extend a RB by a new snapshot. A numerically efficient
# way to realize this is to use QR decomposition methods as implemented in
# [`QRCompress`](@ref). Note that we choose a tolerance `tol` to discard snapshot vectors
# that do not significantly contribute to the basis:

qrcomp = QRCompress(; tol=1e-9);

# We lastly need to set the parameters for the greedy basis assembly by creating a
# [`Greedy`](@ref) object. This includes choosing an error estimate, as well as an error
# tolerance below which we use to stop the basis assembly:

greedy = Greedy(; estimator=Residual(), tol=1e-3);

# With that, we gathered all elements to be able generate the reduced basis:

rbres = assemble(H, grid_train, greedy, lobpcg, qrcomp);

# To finish up the offline phase, we want to define an observable, again as an
# [`AffineDecomposition`](@ref) and then compress it, to be able to measure it efficiently
# in the online stage. We will use the *magnetization* ``M = 2L^{-1} \sum_{i=1}^L S_i^z``
# that serves as a so-called order parameter to distinguish different phases of the system
# in the parameter space. Conveniently, the magnetization already is contained in the third
# term of ``H``:

M    = AffineDecomposition([H.terms[3]], [2 / L])
m, _ = compress(M, rbres.basis);

# Note that the compression again produces an [`AffineDecomposition`](@ref) which now contains
# only the low-dimensional matrices that operate in RB space. In addition to the
# compressed observable, [`compress`](@ref) also returns a second decomposition for analysis
# purposes, which we will not cover in this example and hence did not assign. Since the
# coefficient ``2L^{-1}`` is actually parameter-independent, we can just construct `m`
# without passing any parameter point to obtain the constant reduced magnetization matrix:

m_reduced = m();

# ## Online phase
#
# Having assembled a RB surrogate, we now want to scan the parameter domain by
# measuring observables, in particular the magnetization from above. Fortunately, we have
# finished all Hilbert-space-dimension dependent steps and only operate in the low
# dimensional RB space. This allows us to now compute observables on a much finer grid:

Δ_online = range(first(Δ), last(Δ); length=100)
hJ_online = range(first(hJ), last(hJ); length=100)
grid_online = RegularGrid(Δ_online, hJ_online);

# Instead of solving for the ground states of ``H``, we solve for the lowest eigenvectors of
# the reduced Hamiltonian in the online phase. Again, we need a solver, which in this case
# is [`FullDiagonalization`](@ref), i.e., a wrapper around `LinearAlgebra.eigen`. To use
# degeneracy settings that match `lobpcg` from above, we can use the matching constructor:

fulldiag = FullDiagonalization(lobpcg);

# To compute expectation values on all online grid points — which we mean by "scanning" the
# parameter domain — it is convenient to use a `map`:

using Statistics
magnetization = map(grid_online) do μ
    _, φ_rb = solve(rbres.h, rbres.basis.metric, μ, fulldiag)
    mean(u -> abs(dot(u, m_reduced, u)), eachcol(φ_rb))
end;

# Finally, we can take a look at the results. Note that, to plot a magnetization heatmap,
# we need to pass the transposed matrix `magnetization'`, in order to use the rows as the
# x-axis. In addition to the magnetization, let us also plot the parameter points at which
# we performed truth solves in the offline stage:

using Plots
hm = heatmap(grid_online.ranges[1], grid_online.ranges[2], magnetization';
             xlabel=raw"$\Delta$", ylabel=raw"$h/J$", title="magnetization",
             colorbar=true, clims=(0.0, 1.0), leg=false)
params = unique(rbres.basis.parameters)
scatter!(hm, [μ[1] for μ in params], [μ[2] for μ in params];
         markershape=:xcross, color=:springgreen, ms=3.0, msw=2.0)

# As expected from theory, we observe ``L/2+1`` discrete magnetization plateaus, as well as
# a ferromagnetic (``M=1``) and an antiferromagnetic phase (``M=0``) in the ground state
# phase diagram.
