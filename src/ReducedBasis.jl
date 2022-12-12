module ReducedBasis

using LinearAlgebra
using SparseArrays
using StaticArrays
using TimerOutputs
using DataFrames
using Printf

include("grid.jl")
export RegularGrid, bounds, shift, in_bounds

include("rbasis.jl")
export RBasis, dim, n_truthsolve, extend, reconstruct

include("affine_decomposition.jl")
export AffineDecomposition, n_terms, compress

include("full_diag.jl")
export FullDiagonalization, truth_solve, online_solve
# include("lobpcg.jl")

include("callback.jl")
export default_callback

include("greedy.jl")
export Greedy, ErrorEstimate, Residual, estimate_error, assemble

end
