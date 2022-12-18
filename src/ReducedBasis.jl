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
export RBasis, QRCompress, dimension, n_truthsolve, multiplicity, extend!

include("affine_decomposition.jl")
export AffineDecomposition, n_terms, compress

include("full_diag.jl")
include("lobpcg.jl")
export FullDiagonalization, LOBPCG, solve

include("mpsbasis.jl")
export MPSColumns, EigenDecomposition, overlap_matrix, reconstruct, estimate_gs
include("dmrg.jl")
export DMRG, default_sweeps, default_observer

include("callback.jl")
export DFBuilder, print_callback

include("hamiltonian_cache.jl")
export HamiltonianCache

include("greedy.jl")
export Greedy, ErrorEstimate, Residual, estimate_error, assemble

include("pod.jl")
export POD

end
