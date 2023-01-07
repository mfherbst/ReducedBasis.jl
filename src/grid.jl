"""
Stores equidistant grid of parameter points as well as its interval ranges.
"""
struct RegularGrid{D,N<:Number} <: AbstractArray{SVector{D,N},D}
    points::Array{SVector{D,N},D}
    ranges::Vector{StepRangeLen}
end
"""
    RegularGrid(ranges::Vararg{<:StepRangeLen})

Construct a ``D``-dimensional [`RegularGrid`](@ref) from ``D`` ranges.
"""
function RegularGrid(ranges::Vararg{<:StepRangeLen})
    @assert all(r -> eltype(r) == eltype(ranges[1]), ranges)
    Dim = length(ranges)
    T = eltype(ranges[1])

    points = map(CartesianIndices(Tuple(length.(ranges)))) do idx
        SVector([ranges[d][idx[d]] for d in 1:Dim]...)
    end
    RegularGrid{Dim,T}(points, [ranges...])
end

# AbstractArray interface
Base.length(grid::RegularGrid) = length(grid.points)
Base.size(grid::RegularGrid) = size(grid.points)
Base.size(grid::RegularGrid, i::Int) = size(grid.points, i)
Base.getindex(grid::RegularGrid, args...) = getindex(grid.points, args...)

"""
    bounds(grid::RegularGrid{D}) where {D}

Return `SVector` of grid boundaries.
"""
function bounds(grid::RegularGrid{D}) where {D}
    SVector{D}([(first(r), last(r)) for r in grid.ranges])
end

"""
    in_bounds(μ, grid::RegularGrid)
    
Check whether a given parameter point `μ` is in the convex hull of the grid.
"""
function in_bounds(μ, grid::RegularGrid)
    all([first(r) ≤ p ≤ last(r) for (p, r) in zip(μ, grid.ranges)])
end

"""
    shift(grid::RegularGrid{D,N}, μ_shift; stay_in_bounds=false) where {D,N}
    
Shift a regular grid by a shift vector `μ_shift`. If `stay_in_bounds=true`,
the shifted grid will stay in the convex hull of the unshifted grid.
Note that the shift vector elements cannot be larger than the `grid.ranges` steps.
"""
function shift(grid::RegularGrid{D,N}, μ_shift; stay_in_bounds=false) where {D,N}
    if stay_in_bounds
        if !all(μ_shift .< step.(grid.ranges))
            throw(ArgumentError("shift vector extends out of unit cell"))
        end
        ranges_shifted = [
            (ra .+ μ_shift[d])[1:(end-1)] for (d, ra) in enumerate(grid.ranges)
        ]
    else
        ranges_shifted = [ra .+ μ_shift[d] for (d, ra) in enumerate(grid.ranges)]
    end
    RegularGrid(ranges_shifted...)
end