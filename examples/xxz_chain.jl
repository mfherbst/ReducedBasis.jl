using MKL
using Revise
using LinearAlgebra
using SparseArrays
using ProgressMeter
using ITensors
using Plots
using ReducedBasis

## Setting up the physical model
# Define sparse Pauli matrices
σx = sparse([0.0 1.0; 1.0 0.0])
σy = sparse([0.0 -im; im 0.0])
σz = sparse([1.0 0.0; 0.0 -1.0])

# Convert local-site to many-body operator
function to_global(L::Int, op::M, i::Int) where {M<:AbstractMatrix}
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
    H1 =
        0.25 * sum([
            to_global(L, σx, i) * to_global(L, σx, i + 1) +
            to_global(L, σy, i) * to_global(L, σy, i + 1) for i in 1:(L - 1)
        ])
    H2 = 0.25 * sum([to_global(L, σz, i) * to_global(L, σz, i + 1) for i in 1:(L - 1)])
    H3 = 0.5 * sum([to_global(L, σz, i) for i in 1:L])
    coefficient_map = μ -> [1.0, μ[1], -μ[2]]
    return AffineDecomposition([H1, H2, H3], coefficient_map)
end
function xxz_chain(sites::IndexSet; kwargs...)
    @assert all(hastags.(sites, "S=1/2")) "Site type must be S=1/2"
    xy_term   = OpSum()
    zz_term   = OpSum()
    magn_term = OpSum()
    for i in 1:(length(sites) - 1)
        xy_term   += 0.5, "S+", i, "S-", i + 1
        xy_term   += 0.5, "S-", i, "S+", i + 1
        zz_term   += "Sz", i, "Sz", i + 1
        magn_term += "Sz", i
    end
    magn_term += "Sz", length(sites) # Add last magnetization term
    coefficient_map = μ -> [1.0, μ[1], -μ[2]]
    return AffineDecomposition(
        [
            ApproxMPO(MPO(xy_term, sites), xy_term; kwargs...),
            ApproxMPO(MPO(zz_term, sites), zz_term; kwargs...),
            ApproxMPO(MPO(magn_term, sites), magn_term; kwargs...),
        ],
        coefficient_map,
    )
end

## Offline parameters
L        = 6
sites    = siteinds("S=1/2", L)
H_matrix = xxz_chain(L)
H_mpo    = xxz_chain(sites; cutoff=1e-9)

greedy = Greedy(; estimator=Residual(), tol=1e-3, n_truth_max=64, init_from_rb=false)
pod    = POD(; n_truth=64)

# Degeneracy: m = L + 1 at (Δ, h/J) = (-1, 0)
fulldiag = FullDiagonalization(; n_target=1, tol_degeneracy=0.0)
lobpcg = LOBPCG(; n_target=1, tol_degeneracy=0.0, tol=1e-9)  # TODO: fix wrong results for n_target=1 and tol_degeneracy = 0.0
dm = DMRG(;
    n_target=L + 1,
    tol_degeneracy=1e-4,
    sweeps=default_sweeps(; cutoff_max=1e-9, bonddim_max=1000),
    observer=() -> default_observer(; energy_tol=1e-9),
)

qrcomp = QRCompress(; full_orthogonalize=false, tol_qr=1e-10)  # TODO: fix `full_orthogonalize` (wrong residuals lead to doubly solved parameter points)
edcomp = EigenDecomposition(; cutoff=1e-9)
nocomp = nothing

Δ  = range(-1.0, 2.5, 40)
hJ = range(0.0, 3.5, 40)
grid_train = RegularGrid(Δ, hJ);

## Offline phase (RB assembly)
dfbuilder = DFBuilder()
basis, h, info = assemble(
    H_matrix, grid_train, greedy, lobpcg, qrcomp; callback=dfbuilder ∘ print_callback
)
# basis, h, info = assemble(
#     H_mpo, grid_train, greedy, dm, edcomp; callback=dfbuilder ∘ print_callback
# )
# basis, h, info = assemble(H_matrix, grid_train, pod, lobpcg)
diagnostics = dfbuilder.df;

## Offline phase (observable compression)
M = AffineDecomposition([H_matrix.terms[3]], μ -> [2 / L])
# M = AffineDecomposition([H_mpo.terms[3]], μ -> [2 / L])
m = compress(M, basis);

## Online phase
Δ_online    = range(first(Δ), last(Δ), 150)
hJ_online   = range(first(hJ), last(hJ), 150)
grid_online = RegularGrid(Δ_online, hJ_online)

magnetization = Matrix{Float64}(undef, size(grid_online))
m_reduced = m([1])  # Save observable, since coefficients do not depend on μ 
@showprogress for (idx, μ) in pairs(grid_online)
    λ_rb, φ_rb = solve(h, basis.metric, μ, fulldiag)
    magnetization[idx] = sum(eachcol(φ_rb)) do φ
        dot(φ, m_reduced, φ) / size(φ_rb, 2)  # Divide by multiplicity
    end
end

## Plot observables and snapshot points
hm = heatmap(
    grid_online.ranges[1],
    grid_online.ranges[2],
    magnetization';  # Transpose to have rows as x-axis
    xlabel=raw"$\Delta$",
    ylabel=raw"$h/J$",
    title="magnetization " * raw"$\langle M \rangle = \frac{2}{L}\sum_{i=1}^L S^z_i$",
    colorbar=true,
    clims=(0.0, 1.0),
    leg=false,
)
plot!(hm, grid_online.ranges[1], x -> 1 + x; lw=2, ls=:dash, legend=false, color=:fuchsia)
scatter!(
    hm,
    [μ[1] for μ in diagnostics.parameter],
    [μ[2] for μ in diagnostics.parameter];
    markershape=:xcross,
    color=:springgreen,
    ms=3.0,
    msw=2.0,
)