"""
Central type containing snapshots and associated objects that make a reduced basis.

The snapshot vector can be of a generic type `V` and are stored in an `AbstractVector{V}`.
The associated parameter points of type `P` are contained in `parameters`.
Note that for snapshots ``\\mathbf{\\Psi}(\\mathbf{\\mu}) = (\\Psi_1(\\mathbf{\\mu}),\\dots,\\Psi_m(\\mathbf{\\mu}))`` of multiplicity ``m``
the parameter point ``\\mathbf{\\mu}`` is contained ``m`` times.

Treated as a matrix, the reduced basis ``B = \\Upsilon V`` is made up of the snapshot
vectors as column vectors in ``\\Upsilon``  and vector coefficients ``V``.
The latter are stored in `vectors`. 
In the simple case of ``B = \\Upsilon``, one sets `vectors=I`.

Since the matrix ``B^\\dagger B = V^\\dagger \\Upsilon^\\dagger \\Upsilon V``
is frequently needed, both the `snapshot_overlaps` ``\\Upsilon^\\dagger \\Upsilon``
and the `metric` ``B^\\dagger B`` are stored with generic floating-point type `T`.
"""
struct RBasis{V,T<:Number,P,N}
    snapshots::AbstractVector{V}  # TODO Short comment on what these are
    parameters::Vector{P}
    vectors::N
    snapshot_overlaps::Matrix{T}
    metric::Matrix{T}
end

"""
    dimension(basis::RBasis)

Return the basis dimension ``\\dim B``.
"""
dimension(basis::RBasis) = length(basis.snapshots)
"""
    n_truthsolve(basis::RBasis)

Return the number of truth solves (snapshots) contained in the basis.
"""
n_truthsolve(basis::RBasis) = length(unique(basis.parameters))
"""
    multiplicity(basis::RBasis)

Return the multiplicity of each truth solve.
"""
function multiplicity(basis::RBasis)
    [count(isequal(μ), basis.parameters) for μ in unique(basis.parameters)]
end

"""
    overlap_matrix(v1::Vector, v2::Vector)

Compute the overlap matrix of two sets of vector-like objects `v1` and `v2`.

The computed matrix elements are the dot products `dot(v1[i], v2[j])`.
Correspondingly, the elements of `v1` and `v2` must support a `LinearAlgebra.dot` method.
In the case where `v1 = v2`, the Gram matrix is computed.
"""
function overlap_matrix(v1::Vector, v2::Vector)
    if length(v1) == length(v2)
        overlaps = map(CartesianIndices((1:length(v1), 1:length(v2)))) do idx
            # Strongs zeros on lower triangle
            (first(idx.I) > last(idx.I)) ? false : dot(v1[first(idx.I)], v2[last(idx.I)])
        end  # Returns Matrix{Number}
        for j in 1:length(v2), i in j:length(v1)  # Use hermiticity to fill lower triangle
            overlaps[i, j] = overlaps[j, i]'
        end
        return promote_type.(overlaps)  # Convert to floating-point type
    else  # Do not use hermiticity
        overlaps = map(CartesianIndices((1:length(v1), 1:length(v2)))) do idx
            dot(v1[first(idx.I)], v2[last(idx.I)])
        end
        return overlaps
    end
end

"""
    extend_overlaps(old_overlaps::Matrix, old_snapshots::Vector, new_snapshot::Vector)

Extend an overlap matrix by one snapshot, where only the necessary dot products
are computed.
"""
function extend_overlaps(old_overlaps::Matrix, old_snapshots::Vector, new_snapshot::Vector)
    d_old = length(old_snapshots)
    m = length(new_snapshot)  # Multiplicity of new snapshot
    overlaps = zeros(eltype(old_overlaps), d_old + m, d_old + m)
    overlaps[1:d_old, 1:d_old] = old_overlaps
    for (j, Ψ_new) in enumerate(new_snapshot)
        for (i, Ψ_old) in enumerate(old_snapshots)
            overlaps[i, d_old+j] = dot(Ψ_old, Ψ_new)
            overlaps[d_old+j, i] = overlaps[i, d_old+j]'
        end
        for (i, Ψ_new′) in enumerate(new_snapshot)
            overlaps[d_old+i, d_old+j] = dot(Ψ_new′, Ψ_new)
            overlaps[d_old+j, d_old+i] = overlaps[d_old+i, d_old+j]'
        end
    end
    overlaps
end

"""Extension type for no orthonormalization or compression. See also [`extend`](@ref)."""
struct NoCompress end

"""
Extension type for QR orthonormalization and compression. See [`extend`](@ref) for details.

# Fields

- `tol::Float64`: tolerance for compressing insignificant basis snapshots.
"""
Base.@kwdef struct QRCompress
    tol::Float64 = 1e-10
end

"""
    extend(basis::RBasis, new_snapshot, μ, ::NoCompress)

Extend the reduced basis by one snapshot without any orthogonalization or
compression procedure.
"""
function extend(basis::RBasis, new_snapshot, μ, ::NoCompress)
    # TODO What grinds me here is that new_snapshot is singular,
    #      but actually contains a vector of stuff. Either we make this plural
    #      or we type-annotate by ::AbstractVector to make this clear
    overlaps   = extend_overlaps(basis.snapshot_overlaps, basis.snapshots, new_snapshot)
    snapshots  = append!(copy(basis.snapshots), new_snapshot)
    parameters = append!(copy(basis.parameters), fill(μ, length(new_snapshot)))

    RBasis(snapshots, parameters, I, overlaps, overlaps), length(new_snapshot)
end
"""
    extend(basis::RBasis, new_snapshot, μ, qrcomp::QRCompress)

Extend using QR orthonormalization and compression.

The orthonormalization is performed by QR decomposing the orthogonal projection
``\\mathbf{\\Psi}(\\mathbf{\\mu}_{n+1}) - B_n^\\dagger [B_n^\\dagger B_n]^{-1} B_n \\mathbf{\\Psi}(\\mathbf{\\mu}_n)``
and appending ``Q`` to snapshots. Modes that have an ``R`` column maximum falling below
the `qrcomp.tol` tolerance are dropped.
"""
function extend(basis::RBasis, new_snapshot, μ, qrcomp::QRCompress)
    B    = hcat(basis.snapshots...)
    Ψ    = hcat(new_snapshot...)
    fact = qr(Ψ - B * (cholesky(basis.metric) \ (B' * Ψ)), Val(true))  # pivoted QR

    # Keep orthogonalized vectors of significant norm
    max_per_row = dropdims(maximum(abs, fact.R; dims=2); dims=2)
    keep        = findlast(max_per_row .> qrcomp.tol)
    isnothing(keep) && (return basis, keep)

    v         = [fact.Q[:, i] for i in 1:keep]
    v_norm    = abs(fact.R[keep, keep])
    new_basis = extend(basis, v, μ, NoCompress())

    new_basis, keep, v_norm
end
