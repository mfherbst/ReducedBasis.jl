function print_callback(info::NamedTuple)
    info = merge(info, (; iter_time=TimerOutputs.prettytime(time_ns() - info.t)))
    if info.state == :start
        @printf("%-3s    %-8s    %-6s    %-8s\n", "n", "max. err", "time", "μ")
        println("-"^40)
        @printf(
            "%-3s    %-8.3g    %-6s    %-8s\n",
            info.iteration, info.err_max, info.iter_time, info.μ
        )
    elseif info.state == :run
        @printf(
            "%-3s    %-8.3g    %-6s    %-8s\n",
            info.iteration, info.err_max, info.iter_time, info.μ
        )
    elseif info.state == :finalize
        println("-"^40, "\ntotal time elapsed: $(info.iter_time)\n", "-"^40)
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
        "time" => String[],
        "snapshot" => SVector[],
    ))
end

function (builder::DFBuilder)(info::NamedTuple)
    if info.state == :start || info.state == :run
        push!(builder.df, [info.iteration, info.err_max, info.iter_time, info.μ])
    end 
    merge(info, (; builder))
end