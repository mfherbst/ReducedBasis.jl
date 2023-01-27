using LinearAlgebra, SparseArrays, Plots, ReducedBasis

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
Δ = range(-1.0, 2.5; length=40)
hJ = range(0.0, 3.5; length=40)
grid_train = RegularGrid(Δ, hJ)
greedy = Greedy(; estimator=Residual(), tol=1e-3, init_from_rb=true)
lobpcg = LOBPCG(; n_target=1, tol_degeneracy=1e-4, tol=1e-9)
qrcomp = QRCompress(; tol=1e-9)

# Assemble
info = assemble(H, grid_train, greedy, lobpcg, qrcomp)
basis = info.basis; h = info.h_cache.h;

# Observables
terms = map(idx -> to_global(σz, L, first(idx.I)) * to_global(σz, L, last(idx.I)),
            CartesianIndices((1:L, 1:L)))
coefficient_map = k -> map(idx -> cis(-(first(idx.I) - last(idx.I)) * k) / L,
                           CartesianIndices((1:L, 1:L)))
SFspin = AffineDecomposition(terms, coefficient_map)
sfspin, _ = compress(SFspin, basis; symmetric_terms=true)

# Online phase
Δ_online = range(first(Δ), last(Δ); length=100)
hJ_online = range(first(hJ), last(hJ); length=100)
grid_online = RegularGrid(Δ_online, hJ_online)
fulldiag = FullDiagonalization(lobpcg)

wavevectors = [0.0, π/4, π/2, π]
sf = [zeros(size(grid_online)) for _ in 1:length(wavevectors)]
for (idx, μ) in pairs(grid_online)
    _, φ_rb = solve(h, basis.metric, μ, fulldiag)
    for (i, k) in enumerate(wavevectors)
        sf[i][idx] = sum(eachcol(φ_rb)) do u
            abs(dot(u, sfspin(k), u))
        end / size(φ_rb, 2)
    end
end

# Plot magnetization heatmap and snapshot points
kwargs = (; xlabel=raw"$\Delta$", ylabel=raw"$h/J$", colorbar=true, leg=false)
hms = []
for (i, q) in enumerate(wavevectors)
    push!(hms, heatmap(grid_online.ranges[1], grid_online.ranges[2], sf[i]'; 
                       title="\$k = $(round(q/π; digits=3))\\pi\$", kwargs...))
end
plot(hms...)