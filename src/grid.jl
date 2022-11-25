struct RegularGrid{D,N<:Number} <: AbstractArray{SVector{D,N},D}
    points::Array{SVector{D,N},D}
    ranges::Vector{StepRangeLen}
end
function RegularGrid(ranges::Vararg{<:StepRangeLen})
    @assert all(eltype.(ranges) .== eltype(ranges[1]))
    Dim = length(ranges)
    T = eltype(ranges[1])

    points = Array{SVector{Dim,T},Dim}(undef, length.(ranges)...)
    for idx in CartesianIndices(points)
        points[idx] = SVector([ranges[d][idx[d]] for d in 1:Dim]...)
    end
    return RegularGrid{Dim,T}(points, [ranges...])
end

# AbstractArray interface
Base.length(grid::RegularGrid) = length(grid.points)
Base.size(grid::RegularGrid) = size(grid.points)
Base.size(grid::RegularGrid, i::Int) = size(grid.points, i)
Base.getindex(grid::RegularGrid{D}, i::Int) where {D} = getindex(grid.points, i)
function Base.getindex(grid::RegularGrid{D}, I::Vararg{Int,D}) where {D}
    return getindex(grid.points, I...)
end

# Return grid boundaries (convex hull)
bounds(grid::RegularGrid) = [[first(r), last(r)] for r in grid.ranges]

# Shift grid points by D-dimensional offset vector
function shift(grid::RegularGrid{D,N}, μ_shift; stay_in_bounds=false) where {D,N}
    if stay_in_bounds
        @assert all(μ_shift .< step.(grid.ranges)) "Shift vector must shift by less than one unit cell if `stay_in_bound=true`"
        ranges_shifted = [
            (ra .+ μ_shift[d])[1:(end - 1)] for (d, ra) in enumerate(grid.ranges)
        ]
        return RegularGrid(ranges_shifted...)
    else
        points_shifted = grid.points .+ fill(μ_shift, size(grid))
        ranges_shifted = [ra .+ μ_shift[d] for (d, ra) in enumerate(grid.ranges)]
        return RegularGrid{D,N}(points_shifted, ranges_shifted)
    end
end

# Check if given parameter point is in convex hull of grid
function in_bounds(μ, grid::RegularGrid)
    return all([first(r) ≤ p ≤ last(r) for (p, r) in zip(μ, grid.ranges)])
end