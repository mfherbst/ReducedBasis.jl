using DataFrames

"""
    print_callback(info)

Print diagnostic information in each assembly iterations.
"""
function print_callback(info)
    info = merge(info, (; iter_time=TimerOutputs.prettytime(time_ns() - info.t)))

    if info.state == :iterate
        info = merge(info, (; metric_norm=norm(info.basis.metric - I)))
        if isone(info.iteration)
            @printf("%-3s    %-8s    %-8s    %-6s    %-8s\n",
                "n", "max. err", "‖BᵀB-I‖", "time", "μ")
            println("-"^60)
        end
        μ_round = round.(info.μ; digits=3) # Print rounded parameter vector
        @printf("%-3s    %-8.3g    %-8.3g    %-6s    %-8s\n", info.iteration,
            info.err_max, info.metric_norm, info.iter_time, μ_round)
    elseif info.state == :finalize
        println("-"^60, "\ntotal time elapsed: $(info.iter_time)\n", "-"^60)
    else
        @warn "Invalid info state:" info.state
    end

    flush(stdout)  # Flush e.g. to enable live printing on cluster
    info
end

"""
Carries a `DataFrame` in which assembly information is gathered.
"""
struct DFBuilder
    df::DataFrame
end
function DFBuilder()
    DFBuilder(
        DataFrame(
            "iteration"   => Int[],
            "max_error"   => Float64[],
            "metric_norm" => Float64[],
            "time"        => String[],
            "parameter"   => SVector[],
        )
    )
end

"""
    (builder::DFBuilder)(info)

Push assembly information into `DataFrame`. Note that the functor can be chained with
different callback functions using `∘`.
"""
function (builder::DFBuilder)(info)
    if info.state == :iterate
        push!( # Push standard data into DataFrame
            builder.df,
            (iteration=info.iteration, max_error=info.err_max,
             metric_norm=info.metric_norm, time=info.iter_time, parameter=info.μ),
        )
    end
    merge(info, (; builder)) # Insert DFBuilder into info and return
end