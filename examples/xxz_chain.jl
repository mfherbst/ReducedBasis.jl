using MKL
using Revise
using LinearAlgebra
using SparseArrays
using Plots
using ReducedBasis

## Setting up the physical model
# Define (dense) Pauli matrices
σx = [0.0 1.0; 1.0 0.0]
σy = [0.0 -im; im 0.0]
σz = [1.0 0.0; 0.0 1.0]

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

function xxz_chain(L, ::Type{Matrix})
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
L = 8
H_XXZ = xxz_chain(L, Matrix)

greedy = Greedy(; estimator=Residual(), tol_residual=1e-16, n_truth_max=64)

solver = FullDiagonalization(;
    n_states_max=L+1, tol_degeneracy=1e-10, full_orthogonalize=false, tol_qr=1e-10
) # m = L + 1 degeneracy at (Δ, h/J) = (-1, 0)

Δ = range(-1.0, 2.5, 40)
h = range(0.0, 3.5, 40)
grid_train = RegularGrid(Δ, h);

## Assembly
basis, h, info = assemble(
    H_XXZ, grid_train, greedy, solver; solver_online=solver
);

##
h = info.h_cache.h
hh = info.h_cache.h²
μ = [1.0, 0.0]
λ, φ = online_solve(h, basis.metric, μ, solver)
err = estimate_error(greedy.estimator, μ, hh, basis.metric, λ, φ)

@show err;

## Construct and compress observables
M = AffineDecomposition([H.terms[3]], μ -> [2 / L])
m = compress(M, basis)

## Online phase
Δ_online = range(first(Δ), last(Δ), 200)
h_online = range(first(h), last(h), 200)
grid_online = RegularGrid(Δ_online, h_online)

magnetization = Matrix{Float64}(undef, size(grid_online))
m_reduced = m([1])  # Save observable, since coefficients do not depend on μ 
for (idx, μ) in pairs(grid_online)
    λ_rb, φ_rb = online_solve(h, basis.metric, μ, solver)
    magnetization[idx] = sum(eachcol(φ_rb)) do φ
        dot(φ, m_reduced, φ) / size(φ_rb, 2)  # Divide by multiplicity
    end
end

## Plot observables and snapshot points
hm = heatmap(
    grid_online.ranges[1],
    grid_online.ranges[2],
    magnetization';  # transpose to have rows as x-axis
    title="\$\\langle M \\rangle\$",
    aspect_ratio=:equal,
    clims=(0.0, 1.0),
)
# plot!(hm, grid_online.ranges[1], x -> 1 + x; lw=2, ls=:dash, legend=false)
scatter!(
    hm,
    [μ[1] for μ in diagnostics.snapshot],
    [μ[2] for μ in diagnostics.snapshot];
    markershape=:xcross,
    color=:green,
    msw=2,
)