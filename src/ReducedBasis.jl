module ReducedBasis

using LinearAlgebra
using SparseArrays
using StaticArrays
using TimerOutputs
using DataFrames
using Printf
using ProgressMeter
using ITensors

include("grid.jl")
export RegularGrid, bounds, shift, in_bounds

include("rbasis.jl")
export RBasis, QRCompress, EigenDecomposition
export dimension, n_truthsolve, multiplicity, overlap_matrix, extend!

include("mps_snapshots.jl")
export ApproxMPO, matrix_size, reconstruct

include("affine_decomposition.jl")
export AffineDecomposition, n_terms, compress

include("hamiltonian_cache.jl")
export HamiltonianCache

include("full_diag.jl")
include("lobpcg.jl")
export FullDiagonalization, LOBPCG, solve

include("dmrg.jl")
export DMRG, default_sweeps, default_observer

include("callback.jl")
export DFBuilder, print_callback

include("greedy.jl")
export Greedy, ErrorEstimate, Residual, estimate_error, assemble, estimate_gs

include("pod.jl")
export POD

end
