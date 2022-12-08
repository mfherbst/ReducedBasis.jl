module ReducedBasis

using LinearAlgebra
using SparseArrays
using StaticArrays
using TimerOutputs
using DataFrames

include("grid.jl")
export RegularGrid, bounds, shift, in_bounds

include("basis.jl")
export ReducedBasis, dim, n_truthsolve, extend, reconstruct

include("affine_decomposition.jl")
export AffineDecomposition, n_terms, compress

include("full_diag.jl")
include("lobpcg.jl")

include("greedy.jl")
export Greedy, ErrorEstimate, Residual, assemble

end
