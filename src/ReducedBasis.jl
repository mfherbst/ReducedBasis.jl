module ReducedBasis

using LinearAlgebra
using SparseArrays
using StaticArrays
using TimerOutputs
using DataFrames
using Printf

include("grid.jl")
export RegularGrid, bounds, shift, in_bounds

include("compressalg.jl")
export QRCompress

include("rbasis.jl")
export RBasis, dim, n_truthsolve, extend, reconstruct

include("affine_decomposition.jl")
export AffineDecomposition, n_terms, compress

include("full_diag.jl")
include("lobpcg.jl")
export FullDiagonalization, solve, LOBPCG


include("callback.jl")
export print_callback, DFBuilder

include("greedy.jl")
export Greedy, ErrorEstimate, Residual, estimate_error, assemble

end
