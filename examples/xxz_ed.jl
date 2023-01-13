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
Δ = range(-1.0, 2.5, 40)
hJ = range(0.0, 3.5, 40)
grid_train = RegularGrid(Δ, hJ)
greedy = Greedy(; estimator=Residual(), tol=1e-3, init_from_rb=true)
lobpcg = LOBPCG(; n_target=1, tol_degeneracy=1e-4, tol=1e-9);
qrcomp = QRCompress(; tol=1e-10)

# Assemble
basis, h, info = assemble(H, grid_train, greedy, lobpcg, qrcomp)

# Compress observable
M = AffineDecomposition([H.terms[3]], μ -> [2 / L])
m = compress(M, basis)
m_reduced = m([1])

# Online phase
Δ_online = range(first(Δ), last(Δ), 100)
hJ_online = range(first(hJ), last(hJ), 100)
grid_online = RegularGrid(Δ_online, hJ_online)
fulldiag = FullDiagonalization(lobpcg)
magnetization = map(grid_online) do μ
    _, φ_rb = solve(h, basis.metric, μ, fulldiag)
    sum(eachcol(φ_rb)) do u
        abs(dot(u, m_reduced, u)) / size(φ_rb, 2)
    end
end

# Plot magnetization heatmap and snapshot points
hm = heatmap(grid_online.ranges[1], grid_online.ranges[2], magnetization';
             xlabel=raw"$\Delta$", ylabel=raw"$h/J$", title="magnetization",
             colorbar=true, clims=(0.0, 1.0), leg=false)
plot!(hm, grid_online.ranges[1], x -> 1 + x; lw=2, ls=:dash, legend=false, color=:fuchsia)
params = unique(basis.parameters)
scatter!(hm, [μ[1] for μ in params], [μ[2] for μ in params];
         markershape=:xcross, color=:springgreen, ms=3.0, msw=2.0)
