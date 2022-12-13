function print_callback(info::NamedTuple)
    info = merge(info, (; iter_time=TimerOutputs.prettytime(time_ns() - info.t)))
    if info.state == :run
        info = merge(info, (; metric_norm=norm(info.basis.metric - I)))
        if isone(info.iteration)
            @printf("%-3s    %-8s    %-8s    %-6s    %-8s\n",
                    "n", "max. err", "‖BᵀB - I‖", "time", "μ")
            println("-"^60)
        end
        @printf("%-3s    %-8.3g    %-8.3g    %-6s    %-8s\n",
                info.iteration, info.err_max, info.metric_norm, info.iter_time, info.μ)
    elseif info.state == :finalize
        println("-"^60, "\ntotal time elapsed: $(info.iter_time)\n", "-"^60)
    else
        @warn "Invalid info state:" info.state
    end
    return info
end

struct DFBuilder
    df::DataFrame
end
function DFBuilder()
    DFBuilder(DataFrame(
        "iteration" => Int[],
        "max_error" => Float64[],
        "metric_norm" => Float64[],
        "time" => String[],
        "snapshot" => SVector[]
    ))
end

function (builder::DFBuilder)(info::NamedTuple)
    if info.state == :run
        # push standard data into DataFrame
        push!(builder.df, [info.iteration, info.err_max, info.metric_norm, info.iter_time, info.μ])
    end
    merge(info, (; builder)) # Insert DFbuilder into info and return
end