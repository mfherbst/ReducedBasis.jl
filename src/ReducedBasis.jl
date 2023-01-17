module ReducedBasis

using LinearAlgebra
using SparseArrays
using StaticArrays
using TimerOutputs
using Printf
using ProgressMeter

export RegularGrid
export bounds, in_bounds, shift
include("grid.jl")

export RBasis, NoCompress, QRCompress, EigenDecomposition
export dimension, n_truthsolve, multiplicity, overlap_matrix, extend
include("rbasis.jl")

export AffineDecomposition
export n_terms, compress
include("affine_decomposition.jl")

export HamiltonianCache
include("hamiltonian_cache.jl")

export FullDiagonalization, LOBPCG
export solve
include("full_diag.jl")
include("lobpcg.jl")

export ApproxMPO, DMRG
export default_sweeps
include("mps.jl")

export InfoCollector
export print_callback
include("callback.jl")

export Greedy, ErrorEstimate, Residual
export estimate_error, assemble, interpolate
include("greedy.jl")

export POD
include("pod.jl")

end
