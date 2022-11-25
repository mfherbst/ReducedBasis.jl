# Size of matrix with Vector{MPS} column
function matrix_size(v::Vector{MPS})
    @assert all(length.(v) .== length(v[1])) "All MPS must have same length"
    return (prod(dim.(siteinds(v[1]))), length(v))
end

# Reconstruct MPS in vector to Hilbert space dimensional vectors
function reconstruct(snapshots::Vector{MPS})
    vecs = reconstruct.(snapshots)
    return hcat(vecs...)
end
function reconstruct(mps::MPS)
    sites  = siteinds(mps)
    vec    = zeros(eltype(mps[1]), Tuple(dim(s) for s in sites))
    combos = Iterators.product(Tuple(1:dim(s) for s in sites)...)

    for c in combos
        val = ITensor(1)
        for i in eachindex(mps)
            val *= mps[i] * state(sites[i], c[i])  # Extract state from MPS at index i
        end
        vec[c...] = scalar(val)
    end
    return reshape(vec, length(vec))
end

# MPO wrapper struct containing all contraction kwargs
struct ApproxMPO
    mpo::MPO
    opsum::Sum{Scaled{ComplexF64,Prod{Op}}}  # exact operator sum
    cutoff::Float64
    maxdim::Int
    mindim::Int
    truncate::Bool
end
function ApproxMPO(
    mpo::MPO, opsum; cutoff=1e-9, maxdim=1000, mindim=1, truncate=true
)
    return ApproxMPO(mpo, opsum, cutoff, maxdim, mindim, truncate)
end

function Base.:*(o::ApproxMPO, mps::MPS)
    return apply(
        o.mpo, mps; cutoff=o.cutoff, maxdim=o.maxdim, mindim=o.mindim, truncate=o.truncate
    )
end
