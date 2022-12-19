# Size of matrix with Vector{MPS} column
function matrix_size(v::Vector{MPS})
    @assert all(length.(v) .== length(v[1])) "All MPS must have same length"
    (prod(dim.(siteinds(v[1]))), length(v))
end

# Reconstruct MPS in vector to Hilbert space dimensional vectors
function reconstruct(snapshots::Vector{MPS})
    vecs = reconstruct.(snapshots)
    hcat(vecs...)
end
function reconstruct(mps::MPS)
    sites = siteinds(mps)
    vec = zeros(eltype(mps[1]), Tuple(dim(s) for s in sites))
    combos = Iterators.product(Tuple(1:dim(s) for s in sites)...)

    for c in combos
        val = ITensor(1)
        for i in eachindex(mps)
            val *= mps[i] * state(sites[i], c[i])  # Extract state from MPS at index i
        end
        vec[c...] = scalar(val)
    end
    reshape(vec, length(vec))
end

# MPO wrapper struct containing all contraction kwargs
struct ApproxMPO
    mpo::MPO
    cutoff::Float64
    maxdim::Int
    mindim::Int
    alg::String
    truncate::Bool
end
function ApproxMPO(mpo::MPO; cutoff=1e-9, maxdim=1000, mindim=1, alg="zipup", truncate=true)
    @assert alg == "zipup" || alg == "naive"
    ApproxMPO(mpo, cutoff, maxdim, mindim, alg, truncate)
end

function Base.:*(o::ApproxMPO, mps::MPS)
    apply(o.mpo, mps; cutoff=o.cutoff, maxdim=o.maxdim,
          mindim=o.mindim, truncate=o.truncate)
end
