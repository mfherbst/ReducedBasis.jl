"""
    print_callback(info)

Print diagnostic information in each assembly iterations.
"""
function print_callback(info)
    t_now = time_ns()

    if info.state == :start || info.state == :end
        @printf("%-3s    %-8s    %-8s    %-6s    %-8s\n",
                "n", "max. err", "‖BᵀB-I‖", "time", "μ")
        println("-"^60)
    elseif info.state == :iterate
        info = merge(info, (; iter_time=TimerOutputs.prettytime(t_now - info.cache.t_now),
                              metric_norm=norm(info.basis.metric - I)))
        μ_round = round.(info.μ; digits=3)  # Print rounded parameter vector
        @printf("%-3s    %-8.3g    %-8.3g    %-6s    %-8s\n", info.iteration,
                info.err_max, info.metric_norm, info.iter_time, μ_round)
    else
        @warn "Invalid info state:" info.state
    end
    info = merge(info, (; cache=merge(info.cache, (; t_now))))

    flush(stdout)  # Flush e.g. to enable live printing on cluster
    info
end

"""
Carries a `Dict` of `Vector`s which contain information from greedy assembly iterations.
"""
struct InfoCollector
    data::Dict{Symbol,Vector}
end
"""
    InfoCollector(fields::Symbol...)

Construct `InfoCollector` from fields that are contained in the `info` iteration
state object. Possible fields to select from are:

- `iteration`: number of iteration at which the information was obtained.
- `err_grid`: error estimate on all parameter points of the training grid.
- `err_max`: maximal error estimate on the grid.
- `λ_grid`: RB energies on all training grid points.
- `μ`: parameter point at which truth solve has been performed.
- `basis`: `RBasis` at the current iteration.
- `h_cache`: `HamiltonianCache` at the current iteration.
- `extend_info`: info that is specific to the chosen extension procedure.

"""
function InfoCollector(fields::Symbol...)
    InfoCollector(Dict(f => [] for f in fields))
end

"""
    (collector::InfoCollector)(info)

Push iteration information into `InfoCollector` and return `info` object,
containing the `InfoCollector` itself.
"""
function (collector::InfoCollector)(info)
    for (key, val) in collector.data
        !haskey(info, key) && continue  # If key is not contained, do not push
        push!(val, info[key])
    end
    merge(info, (; collector))  # Insert InfoCollector into info and return
end