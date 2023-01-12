using LinearAlgebra
using SparseArrays
using ITensors
using Plots
using ReducedBasis

## Setting up the physical model
# Define sparse Pauli matrices
σx = sparse([0.0 1.0; 1.0 0.0])
σy = sparse([0.0 -im; im 0.0])
σz = sparse([1.0 0.0; 0.0 -1.0])

# Convert local-site to many-body operator
function to_global(op::M, L::Int, i::Int) where {M<:AbstractMatrix}
    d = size(op, 1)
    @assert d == size(op, 2) "Operator has to be a square matrix."

    if i == 1
        return kron(op, M(I, d^(L - 1), d^(L - 1)))
    elseif i == L
        return kron(M(I, d^(L - 1), d^(L - 1)), op)
    else
        return kron(kron(M(I, d^(i - 1), d^(i - 1)), op), M(I, d^(L - i), d^(L - i)))
    end
end

function xxz_chain(L)
    H1 = 0.25 * sum([to_global(σx, L, i) * to_global(σx, L, i + 1) +
                     to_global(σy, L, i) * to_global(σy, L, i + 1) for i in 1:(L-1)])
    H2 = 0.25 * sum([to_global(σz, L, i) * to_global(σz, L, i + 1) for i in 1:(L-1)])
    H3 = 0.5  * sum([to_global(σz, L, i) for i in 1:L])
    coefficient_map = μ -> [1.0, μ[1], -μ[2]]
    AffineDecomposition([H1, H2, H3], coefficient_map)
end
function xxz_chain(sites::IndexSet; kwargs...)
    !all(hastags.(sites, "S=1/2")) && error("Site type must be S=1/2")
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

## Offline parameters
L        = 6
sites    = siteinds("S=1/2", L)
H_matrix = xxz_chain(L)
H_mpo    = xxz_chain(sites; truncate=true, cutoff=1e-9)

# Assembly strategies
greedy = Greedy(; estimator=Residual(), tol=1e-3, n_truth_max=32, init_from_rb=true)
pod    = POD(; n_truth=64, verbose=true)

# Truth solvers (degeneracy: m = L + 1 at (Δ, h/J) = (-1, 0))
fulldiag = FullDiagonalization(; n_target=L+1, tol_degeneracy=1e-4)
lobpcg = LOBPCG(; n_target=L+1, tol_degeneracy=1e-4, tol=1e-9)
dm = DMRG(;
    n_target=1,
    tol_degeneracy=0.0,
    sweeps=default_sweeps(; cutoff_max=1e-9, bonddim_max=1000),
    observer=() -> DMRGObserver(; energy_tol=1e-9),
)

# Algorithms for orthogonalization and mode compression
qrcomp = QRCompress(; tol=1e-10)
edcomp = EigenDecomposition(; cutoff=1e-7)
nocomp = NoCompress()

# Training grid
Δ = range(-1.0, 2.5, 40)
hJ = range(0.0, 3.5, 40)
grid_train = RegularGrid(Δ, hJ);

# Offline phase (RB assembly)
collector = InfoCollector(:err_grid, :λ_grid, :μ)
basis, h, info = assemble(
    H_matrix, grid_train, greedy, lobpcg, qrcomp; callback=collector ∘ print_callback
    # H_mpo, grid_train, greedy, dm, edcomp; callback=collector ∘ print_callback,
);
# basis, info = assemble(H_matrix, grid_train, pod, lobpcg)
# h_cache = HamiltonianCache(H, basis)
# h = h_cache.h

# Offline phase (observable compression)
M = AffineDecomposition([H_matrix.terms[3]], μ -> [2 / L])
# M = AffineDecomposition([H_mpo.terms[3]], μ -> [2 / L])
m = compress(M, basis)
m_reduced = m([1]);  # Save observable, since coefficients do not depend on μ 

E_grids = [map(maximum, λ_grid) for λ_grid in collector.data[:λ_grid]]
varcheck = BitMatrix(undef, size(grid_train))
for (idx, λ) in pairs(E_grids[end])
    varcheck[idx] = round.(E_grids[end-1][idx] - λ; digits=10) .≥ 0.0
end
@show all(varcheck)

# Online phase
Δ_online    = range(first(Δ), last(Δ), 150)
hJ_online   = range(first(hJ), last(hJ), 150)
grid_online = RegularGrid(Δ_online, hJ_online)

magnetization = map(grid_online) do μ
    _, φ_rb = solve(h, basis.metric, μ, fulldiag)
    sum(eachcol(φ_rb)) do u
        abs(dot(u, m_reduced, u)) / size(φ_rb, 2)  # Divide by multiplicity
    end
end

# Plot observables and snapshot points
hm = heatmap(grid_online.ranges[1], grid_online.ranges[2], magnetization';  # Transpose to have rows as x-axis
             xlabel=raw"$\Delta$", ylabel=raw"$h/J$", title="magnetization ",
             colorbar=true, clims=(0.0, 1.0), leg=false)
plot!(hm, grid_online.ranges[1], x -> 1 + x; lw=2, ls=:dash, legend=false, color=:fuchsia)
params = unique(basis.parameters)
scatter!(hm, [μ[1] for μ in params], [μ[2] for μ in params];
         markershape=:xcross, color=:springgreen, ms=3.0, msw=2.0)