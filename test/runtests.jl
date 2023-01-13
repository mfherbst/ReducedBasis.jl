using Random, LinearAlgebra, SparseArrays
using ReducedBasis

# Initialize seed
Random.seed!(0)

# Convert local-site to many-body operator
function to_global(L::Int, op::M, i::Int) where {M<:AbstractMatrix}
    d = size(op, 1)
    if i == 1
        return kron(op, M(I, d^(L - 1), d^(L - 1)))
    elseif i == L
        return kron(M(I, d^(L - 1), d^(L - 1)), op)
    else
        return kron(kron(M(I, d^(i - 1), d^(i - 1)), op), M(I, d^(L - i), d^(L - i)))
    end
end

# Construct XXZ Hamiltonian matrix
function xxz_chain(N)
    σx = sparse([0.0 1.0; 1.0 0.0])
    σy = sparse([0.0 -im; im 0.0])
    σz = sparse([1.0 0.0; 0.0 -1.0])
    H1 = 0.25 * sum([to_global(N, σx, i) * to_global(N, σx, i + 1) +
                        to_global(N, σy, i) * to_global(N, σy, i + 1) for i in 1:(N-1)])
    H2 = 0.25 * sum([to_global(N, σz, i) * to_global(N, σz, i + 1) for i in 1:(N-1)])
    H3 = 0.5  * sum([to_global(N, σz, i) for i in 1:N])
    coefficient_map = μ -> [1.0, μ[1], -μ[2]]

    AffineDecomposition([H1, H2, H3], coefficient_map)
end

# Convenience function for fast generation of XXZ RBasis
# TODO: replace by some generic, XXZ-independent model
function fast_assemble(n_truth=20)
    L    = 6
    H    = xxz_chain(L)
    Δ    = range(-1.0, 2.5; length=30)
    hJ   = range(0.0, 3.5; length=30)
    grid = RegularGrid(Δ, hJ)
    
    pod = POD(; n_truth, verbose=false)
    fulldiag = FullDiagonalization(; n_target=1, tol_degeneracy=1e-4)
    basis, = assemble(H, grid, pod, fulldiag)
    basis
end

include("grid.jl")
include("affine_decomposition.jl")
include("rbasis.jl")
include("xxz.jl")
include("mps.jl")
