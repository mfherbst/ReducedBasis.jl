using Test, Random
using LinearAlgebra: norm
using ReducedBasis: RegularGrid, bounds, in_bounds, shift

@testset "Basic RegularGrid functionality" begin
    for D in 1:3
        ranges = [range(sort(rand(2))..., rand(10:10:30)) for _ in 1:D]
        grid   = RegularGrid(ranges...)
        μ      = rand.(ranges)

        @test length(grid) == prod(length.(ranges)) 
        @test size(grid) == Tuple(length.(ranges))
        @test in_bounds(μ, grid) == true
        @test in_bounds(2μ / norm(μ), grid) == false
    end
end

@testset "RegularGrid shifting" begin
    for D in 1:3
        ranges  = [range(sort(rand(2))..., rand(10:10:30)) for _ in 1:D]
        grid    = RegularGrid(ranges...)
        μ_shift = step.(ranges) / 2

        grid_shift_out = shift(grid, μ_shift)
        @test all(
            [all(b .< b_shift) for (b, b_shift) in zip(bounds(grid), bounds(grid_shift_out))]
        )
        @test length(grid) == length(grid_shift_out)
        @test size(grid) == size(grid_shift_out)

        grid_shift_in = shift(grid, μ_shift; stay_in_bounds=true)
        @test size(grid_shift_in) == size(grid) .- Tuple(fill(1, D))
    end
end