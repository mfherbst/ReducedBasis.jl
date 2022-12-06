# Parametric type for parameter vector?
struct Grid{D} <: AbstractArray{NTuple{D,Float64},D}
    points::Array{NTuple{D,Float64},D}
    bounds::Vector{Vector{Float64}}
    ranges::Vector{StepRangeLen}
end
function Grid(bounds::Vector{<:Vector{<:Number}}, lengths::Vector{<:Integer}; shifted=false)
    bounds = Vector{Float64}.(bounds)
    dim = length(bounds)

    if shifted
        offsets = [diff(bounds[d]) / 2(lengths[d] - 1) for d = 1:dim]
        ranges = [range(bounds[d]..., lengths[d] - 1) .+ offsets[d] for d = 1:dim]
    else
        ranges = [range(bounds[d]..., lengths[d]) for d = 1:dim]
    end

    points = Array{NTuple{dim,Float64},dim}(undef, lengths...)
    for idx in CartesianIndices(points)
        points[idx] = Tuple(ranges[d][idx[d]] for d = 1:dim)
    end
    Grid{dim}(points, bounds, ranges)
end

# AbstractArray interface
Base.length(grid::Grid) = prod(grid.len)
Base.size(grid::Grid) = size(grid.points)
Base.size(grid::Grid, i::Int) = size(grid.points, i)
Base.getindex(grid::Grid{D}, i::Int) where {D} = getindex(grid.points, i)
Base.getindex(grid::Grid{D}, I::Vararg{Int,D}) where {D} = getindex(grid.points, I...)
Base.setindex!(grid::Grid{D}, μ, i::Int) where {D} = Base.setindex!(grid.points, μ, i)
Base.setindex!(grid::Grid{D}, μ, I::Vararg{Int,D}) where {D} = Base.setindex!(grid.points, μ, I...)

# Check if given parameter point is in convex hull of grid
function in_bounds(μ, grid::Grid)
    contained = [p ∈ grid.bounds[d] for (d, p) in enumerate(μ)]
    return all(contained) ? true : false
end