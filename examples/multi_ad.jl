# # Affine decompositions with multi-indices and additional parameters
#
# In this example, we want to explore the capabilities of the central
# [`AffineDecomposition`](@ref) type. To advance the previous examples where we covered
# the magnetization — a very simple observable that consists of only one affine term and a
# parameter-independent coefficient — we now turn to observables where the indices are
# multi-indices ``r = (r_1, \dots, r_d)`` and the coefficients can depend on additional
# parameters ``p``, aside from the ``\bm{\mu}`` parameter points that are present in the
# Hamiltonian:
#
# ```math
# O(\bm{\mu}, p) = \sum_{q=1}^Q \alpha_q(\bm{\mu}, p)\, O_q
# ```
#
# To stay within the realm of spin physics, we will consider the so-called
# *spin structure factor*
#
# ```math
# \mathcal{S}(k) = \frac{1}{L} \sum_{r,r'=1}^L e^{-i (r - r') k} S^z_r S^z_{r'},\quad
# \alpha_{r,r'}(k) = \frac{e^{-i (r - r') k}}{L}, \quad
# O_{r,r'} =  S^z_r S^z_{r'}
# ```
#
# with a wavevector parameter ``k``, to discuss the implementation of a more complicated
# observable.
#
# To provide a physical setup, we again use the XXZ chain from [The reduced basis workflow](@ref):

using LinearAlgebra
using SparseArrays
using ReducedBasis

σx = sparse([0.0 1.0; 1.0 0.0])
σy = sparse([0.0 -im; im 0.0])
σz = sparse([1.0 0.0; 0.0 -1.0])

function to_global(op::M, L::Int, i::Int) where {M<:AbstractMatrix}
    d = size(op, 1)
    if i == 1
        kron(op, M(I, d^(L - 1), d^(L - 1)))
    elseif i == L
        kron(M(I, d^(L - 1), d^(L - 1)), op)
    else
        kron(kron(M(I, d^(i - 1), d^(i - 1)), op), M(I, d^(L - i), d^(L - i)))
    end
end

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

# Construct Hamiltonian for XXZ chain with 6 sites ...

L = 6
H = xxz_chain(L);

# ... and generate a surrogate reduced basis using an exact diagonalization solver.

Δ  = range(-1.0, 2.5; length=40)
hJ = range( 0.0, 3.5; length=40)
grid_train = RegularGrid(Δ, hJ)
greedy = Greedy(; estimator=Residual())
lobpcg = LOBPCG(; tol_degeneracy=1e-4)
qrcomp = QRCompress(; tol=1e-9)
rbres = assemble(H, grid_train, greedy, lobpcg, qrcomp);

# Now the task is to implement the double-sum in ``\mathcal{S}``, as well as the
# ``k``-dependency in the coefficients. The double-sum can be encoded by putting all
# ``S^z_r S^z_{r'}`` combinations into a ``L \times L`` matrix:

terms = map(idx -> to_global(σz, L, first(idx.I)) * to_global(σz, L, last(idx.I)),
            CartesianIndices((1:L, 1:L)));

# Correspondingly, the coefficient function now has to map one ``k`` value to a matrix of
# coefficients of the same size as the `terms` matrix:

coefficient_map = k -> map(idx -> cis(-(first(idx.I) - last(idx.I)) * k) / L,
                           CartesianIndices((1:L, 1:L)));

# One feature of the structure factor that also shows up in many other affine decompositions
# with double-sums is that the term indices commute, i.e. ``O_{r,r'} = O_{r',r}``. In that
# case, only the upper triangular matrix has to be computed since
# ``B^\dagger O_{r,r'} B = B^\dagger O_{r',r} B`` are the same in the compressed affine
# decomposition. So let's create the [`AffineDecomposition`](@ref) and compress, exploiting
# this symmetry using the `symmetric_terms` keyword argument:

SFspin    = AffineDecomposition(terms, coefficient_map)
sfspin, _ = compress(SFspin, rbres.basis; symmetric_terms=true);

# In the online evaluation of the structure factor, we then need to define some wavevector
# values and compute the structure factor at each of them. As usual, we first define a
# finer online grid of points and the matching online solver:

Δ_online    = range(first(Δ), last(Δ); length=100)
hJ_online   = range(first(hJ), last(hJ); length=100)
grid_online = RegularGrid(Δ_online, hJ_online)
fulldiag    = FullDiagonalization(lobpcg);

# And then we map the grid points to the corresponding structure factor values for a set
# of different wavevectors:

using Statistics
wavevectors = [0.0, π/4, π/2, π]
sf = [zeros(size(grid_online)) for _ in 1:length(wavevectors)]
for (idx, μ) in pairs(grid_online)
    _, φ_rb = solve(rbres.h_cache.h, rbres.basis.metric, μ, fulldiag)
    for (i, k) in enumerate(wavevectors)
        sf[i][idx] = mean(u -> real(dot(u, sfspin(k), u)), eachcol(φ_rb))
    end
end

# Here we again see the convenience of measuring observables in the online stage; adding
# more wavevector values does not significantly increase the computational cost, since it
# corresponds to a mere reevaluation of the coefficient functions and small vector-matrix
# products. Finally, let us see how the structure factor behaves for the different
# wavevector values:

using Plots
kwargs = (; xlabel=raw"$\Delta$", ylabel=raw"$h/J$", colorbar=true, leg=false)
hms = []
for (i, q) in enumerate(wavevectors)
    push!(hms, heatmap(grid_online.ranges[1], grid_online.ranges[2], sf[i]';
                       title="\$k = $(round(q/π; digits=3))\\pi\$", kwargs...))
end
plot(hms...)

# It can be nicely seen that the spin structure factor indicates the ferromagnetic phase at
# ``k=0`` and then moves through the magnetization plateaus until it reaches the
# antiferromagnetic plateau at ``k=\pi``.
