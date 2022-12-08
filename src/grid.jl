# Parametric type for parameter vector?
struct RegularGrid{D} <: AbstractArray{<:SVector{D,Number},D}
    points::Array{SVector{D,Number},D}
    ranges::Vector{StepRangeLen}
end
function RegularGrid(ranges::Vector{StepRangeLen})
    Dim = length(ranges)
    @assert all(eltype.(range)) # Enforce same type in all dimensions?
    T = eltype(ranges[1])

    points = Array{SVector{Dim,T},Dim}(undef, lengths...)
    for idx in CartesianIndices(points)
        points[idx] = SVector([ranges[d][idx[d]] for d = 1:Dim]...)
    end
    Grid{Dim}(points, ranges)
end

# AbstractArray interface
Base.length(grid::RegularGrid) = length(grid.points)
Base.size(grid::RegularGrid) = size(grid.points)
Base.size(grid::RegularGrid, i::Int) = size(grid.points, i)
Base.getindex(grid::RegularGrid{D}, i::Int) where {D} = getindex(grid.points, i)
Base.getindex(grid::RegularGrid{D}, I::Vararg{Int,D}) where {D} = getindex(grid.points, I...)

# Return grid boundaries (convex hull)
bounds(grid::RegularGrid) = [[first(r), last(r)] for r in grid.ranges]

# Shift grid points by D-dimensional offset vector
shift(grid::RegularGrid) = missing

# Check if given parameter point is in convex hull of grid
function in_bounds(μ, grid::RegularGrid)
    all([first(r) ≤ p ≤ last(r) for (p, r) in zip(μ, grid.ranges)])
end