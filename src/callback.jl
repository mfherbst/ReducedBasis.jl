function default_callback(info::NamedTuple)
    if info.state == :starting
        @printf("%-3s    %-8s    %-6s    %-8s\n", "n", "max. err", "time", "μ")
        @printf("%-3s    %-8.3g    %-6s    %-8s\n", info.iteration, info.err_max, info.iter_time, info.μ)
    elseif info.state == :running
        @printf("%-3s    %-8.3g    %-6s    %-8s\n", info.iteration, info.err_max, info.iter_time, info.μ)
    elseif info.state == :stopping
        # How to build DataFrame from accumulated info?
        return nothing
    end
end

function diagnostics_callback(info::NamedTuple, ::FullDiagonalization)

end