function default_callback(info::NamedTuple)

    if state == :starting
        @printf("%3s %8.3g %6s %8", "n", "max. err", "time", "μ")
        @printf("%3s %8.3g %6s %8", info.iteration, info.err_max, info.itertime, info.μ)
    elseif state == :running
        @printf("%3s %8.3g %6s %8", info.iteration, info.err_max, info.itertime, info.μ)
    elseif state == :stopping
        # How to build DataFrame from accumulated info?
        return nothing
    end
end

function diagnostics_callback(info::NamedTuple, ::FullDiagonalization)

end