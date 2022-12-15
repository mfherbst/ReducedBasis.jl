struct RegularGrid{D,N<:Number} <: AbstractArray{SVector{D,N},D}
    points::Array{SVector{D,N},D}
    ranges::Vector{StepRangeLen}
end
function RegularGrid(ranges::Vararg{<:StepRangeLen})
    Dim = length(ranges)
    @assert all(t -> t == eltype(ranges[1]), eltype.(ranges)) # TODO: Enforce same type in all dimensions?
    T = eltype(ranges[1])

    points = Array{SVector{Dim,T},Dim}(undef, length.(ranges)...)
    for idx in CartesianIndices(points)
        points[idx] = SVector([ranges[d][idx[d]] for d = 1:Dim]...)
    end
    RegularGrid{Dim,T}(points, [ranges...])
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
# TODO: shift function
shift(grid::RegularGrid, μ_shift) = error()

# Check if given parameter point is in convex hull of grid
function in_bounds(μ, grid::RegularGrid)
    all([first(r) ≤ p ≤ last(r) for (p, r) in zip(μ, grid.ranges)])
end