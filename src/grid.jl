struct RegularGrid{D,N<:Number} <: AbstractArray{SVector{D,N},D}
    points::Array{SVector{D,N},D}
    ranges::Vector{StepRangeLen}
end
function RegularGrid(ranges::Vararg{<:StepRangeLen})
    @assert all(r -> eltype(r) == eltype(ranges[1]), ranges)
    Dim = length(ranges)
    T = eltype(ranges[1])

    points = map(CartesianIndices(length.(ranges)...)) do idx
        SVector([ranges[d][idx[d]] for d in 1:Dim]...)
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
function bounds(grid::RegularGrid{D}) where {D}
    SVector{D}([(first(r), last(r)) for r in grid.ranges])
end

# Shift grid points by D-dimensional offset vector
function shift(grid::RegularGrid{D,N}, μ_shift; stay_in_bounds=false) where {D,N}
    if stay_in_bounds
        if !all(μ_shift .< step.(grid.ranges))
            throw(ArgumentError("shift vector extends out of unit cell"))
        end
        ranges_shifted = [
            (ra .+ μ_shift[d])[1:(end - 1)] for (d, ra) in enumerate(grid.ranges)
        ]
    else
        ranges_shifted = [ra .+ μ_shift[d] for (d, ra) in enumerate(grid.ranges)]
    end
    RegularGrid(ranges_shifted...)
end

# Check if given parameter point is in convex hull of grid
function in_bounds(μ, grid::RegularGrid)
    return all([first(r) ≤ p ≤ last(r) for (p, r) in zip(μ, grid.ranges)])
end