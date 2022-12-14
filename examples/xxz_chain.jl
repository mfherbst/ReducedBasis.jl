using MKL
using Revise
using LinearAlgebra
using SparseArrays
using ProgressMeter
using Plots
using ReducedBasis

## Setting up the physical model
# Define sparse Pauli matrices
σx = sparse([0.0 1.0; 1.0 0.0])
σy = sparse([0.0 -im; im 0.0])
σz = sparse([1.0 0.0; 0.0 -1.0])

# Convert local-site to many-body operator
function local_to_global(L::Int, op::M, i::Int) where M <: AbstractMatrix
    d = size(op, 1)
    @assert d == size(op, 2) "Operator has to be a square matrix."

    if i == 1
        return kron(op, M(I, d^(L - 1), d^(L - 1)))
    elseif i == L
        return kron(M(I, d^(L - 1), d^(L - 1)), op)
    else
        return kron(
            kron(M(I, d^(i - 1), d^(i - 1)), op), M(I, d^(L - i), d^(L - i))
        )
    end
end

function xxz_chain(L)
    H1 = 0.25 * sum([
        local_to_global(L, σx, i) * local_to_global(L, σx, i + 1) +
        local_to_global(L, σy, i) * local_to_global(L, σy, i + 1) for i = 1:L-1
    ])
    H2 = 0.25 * sum([
        local_to_global(L, σz, i) * local_to_global(L, σz, i + 1) for i = 1:L-1
    ])
    H3 = 0.5 * sum([local_to_global(L, σz, i) for i = 1:L])
    coefficient_map = μ -> [1.0, μ[1], -μ[2]]
    return AffineDecomposition([H1, H2, H3], coefficient_map)
end

## Offline parameters
L = 10
H_XXZ = xxz_chain(L)
greedy = Greedy(; estimator=Residual(), tol=1e-3, n_truth_max=64)
fulldiag = FullDiagonalization(; n_target=L+1, tol_degeneracy=1e-4) # m = L + 1 degeneracy at (Δ, h/J) = (-1, 0)
# TODO: fix iteration for n_target=1 (terminates after n=2)
lobpcg = LOBPCG(; tol=1e-9, n_target=1, tol_degeneracy=0.0)
compressalg = QRCompress(; full_orthogonalize=false, tol_qr=1e-10)
# compressalg = nothing

Δ = range(-1.0, 2.5, 40)
hJ = range(0.0, 3.5, 40)
grid_train = RegularGrid(Δ, hJ);

## Offline phase (RB assembly)
diagnostics = DFBuilder()
basis, h, info = assemble(
    H_XXZ, grid_train, greedy, lobpcg, compressalg;
    callback=diagnostics ∘ print_callback, init_from_rb=false,
)
diagnostics = diagnostics.df  # TODO: improve DataFrame management

## Offline phase (observable compression)
M = AffineDecomposition([H_XXZ.terms[3]], μ -> [2 / L])
m = compress(M, basis);

## Online phase
Δ_online = range(first(Δ), last(Δ), 500)
hJ_online = range(first(hJ), last(hJ), 500)
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
    magnetization';  # transpose to have rows as x-axis
    xlabel=raw"$\Delta$",
    ylabel=raw"$h/J$",
    title="magnetization "*raw"$\langle M \rangle = \frac{2}{L}\sum_{i=1}^L S^z_i $",
    colorbar=true,
    clims=(0.0, 1.0),
    leg=false,
)
plot!(hm, grid_online.ranges[1], x -> 1 + x; lw=2, ls=:dash, legend=false, color=:green3)
scatter!(
    hm,
    [μ[1] for μ in diagnostics.parameter],
    [μ[2] for μ in diagnostics.parameter];
    markershape=:xcross,
    color=:springgreen,
    ms=3.0,
    msw=2.0,
)