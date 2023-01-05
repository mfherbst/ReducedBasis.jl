using Random

# Initialize seed
Random.seed!(0)

include("grid.jl")
include("affine_decomposition.jl")
include("rbasis.jl")
include("errors.jl")
include("xxz_chain.jl")
include("mps.jl")
