using LinearAlgebra
using SparseArrays
using Plots
using ReducedBasis

# Define Pauli matrices
σx = sparse([0.0 1.0; 1.0 0.0])
σy = sparse([0.0 -im; im 0.0])
σz = sparse([1.0 0.0; 0.0 -1.0]);

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

function xxz_chain(L)
    H1 = 0.25 * sum([to_global(σx, L, i) * to_global(σx, L, i + 1) +
                     to_global(σy, L, i) * to_global(σy, L, i + 1) for i in 1:(L-1)])
    H2 = 0.25 * sum([to_global(σz, L, i) * to_global(σz, L, i + 1) for i in 1:(L-1)])
    H3 = 0.5  * sum([to_global(σz, L, i) for i in 1:L])
    coefficient_map = μ -> [1.0, μ[1], -μ[2]]
    AffineDecomposition([H1, H2, H3], coefficient_map)
end

# Offline parameters
L = 6
H = xxz_chain(L)
Δ = range(-1.0, 2.5; length=20)
hJ = range(0.0, 3.5; length=20)
grid_train = RegularGrid(Δ, hJ)
pod = POD(; n_vectors=24)
lobpcg = LOBPCG(; n_target=1, tol_degeneracy=1e-4, tol=1e-9);

# Assemble
info = assemble(H, grid_train, pod, lobpcg)
h_cache = HamiltonianCache(H, info.basis)
basis = info.basis; h = h_cache.h;

# Compress observable
M = AffineDecomposition([H.terms[3]], μ -> [2 / L])
m, _ = compress(M, basis)
m_reduced = m([])

# Online phase
Δ_online = range(first(Δ), last(Δ); length=100)
hJ_online = range(first(hJ), last(hJ); length=100)
grid_online = RegularGrid(Δ_online, hJ_online)
fulldiag = FullDiagonalization(lobpcg)
magnetization = map(grid_online) do μ
    _, φ_rb = solve(h, basis.metric, μ, fulldiag)
    sum(eachcol(φ_rb)) do u
        abs(dot(u, m_reduced, u))
    end / size(φ_rb, 2)
end

# Plot magnetization heatmap and snapshot points
hm = heatmap(grid_online.ranges[1], grid_online.ranges[2], magnetization';
             xlabel=raw"$\Delta$", ylabel=raw"$h/J$", title="magnetization",
             colorbar=true, clims=(0.0, 1.0), leg=false)
plot!(hm, grid_online.ranges[1], x -> 1 + x; lw=2, ls=:dash, legend=false, color=:green)
