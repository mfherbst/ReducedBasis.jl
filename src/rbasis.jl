import Base: @kwdef

"""
Central type containing snapshots and associated objects that make a reduced basis.

The snapshot vectors are contained in `snapshots::AbstractVector{V}` where the snapshots
are of type `V`. Here a "snapshot" refers to all vectors that were obtained from a solve
at a specific parameter point. For greedy-generated bases, each element in `snapshots` has
a parameter point in `parameter::Vector{P}`, i.e. for snapshots
``\\bm{\\Psi}(\\bm{\\mu_i}) = (\\Psi_1(\\bm{\\mu_i}),\\dots,\\Psi_m(\\bm{\\mu_i}))``
of multiplicity ``m`` the parameter point ``\\bm{\\mu}`` is contained ``m`` times.
However, for general basis-assembly strategies this one-to-one correspondence might not
be true, e.g. for [`POD`](@ref) where `snapshots` contains singular vectors based on
all truth solves.

Treated as a matrix, the reduced basis ``B = \\Upsilon V`` is made up of the snapshot
vectors as column vectors in ``\\Upsilon`` and vector coefficients ``V``. The latter are
stored in `vectors`. In the simple case of ``B = \\Upsilon``, one sets `vectors=I`.

Since the matrix ``B^\\dagger B = V^\\dagger \\Upsilon^\\dagger \\Upsilon V``
is frequently needed, both the `snapshot_overlaps` ``\\Upsilon^\\dagger \\Upsilon``
and the `metric` ``B^\\dagger B`` are stored with generic floating-point type `T`.
"""
struct RBasis{V,T<:Number,P,N}
    snapshots::AbstractVector{V}
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
    overlap_matrix(f, v1::Vector, v2::Vector)

Compute the overlap matrix of two sets of vector-like objects `v1` and `v2`.

The computed matrix elements are the dot products `dot(v1[i], v2[j])`.
Correspondingly, the elements of `v1` and `v2` must support a `LinearAlgebra.dot` method.
In the case where `v1 = v2`, the Gram matrix is computed.

Optionally, a function `f` can be provided which is applied to each element of `v2`.
"""
function overlap_matrix(f, v1::Vector, v2::Vector)
    if length(v1) == length(v2)
        overlaps = map(Iterators.product(1:length(v1), 1:length(v2))) do (i, j) 
            # Strong zeros on lower triangle
            (i > j) ? false : dot(v1[i], f(v2[j]))
        end  # Returns Matrix{Number}
        for j in 1:length(v2), i in j:length(v1)  # Use hermiticity to fill lower triangle
            overlaps[i, j] = overlaps[j, i]'
        end
        return promote_type.(overlaps)  # Convert to floating-point type
    else  # Do not use hermiticity
        overlaps = map(Iterators.product(1:length(v1), 1:length(v2))) do (i, j)
            dot(v1[i], f(v2[j]))
        end
        return overlaps
    end
end

overlap_matrix(v1::Vector, v2::Vector) = overlap_matrix(identity, v1, v2)

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
@kwdef struct QRCompress
    tol::Float64 = 1e-10
end

"""
Extension type for orthogonalization and compression using eigenvalue decomposition
of the basis overlap matrix. See also [`extend`](@ref).

# Fields

- `cutoff::Float64=1e-6`: cutoff for minimal eigenvalue accuracy.
"""
@kwdef struct EigenDecomposition
    cutoff::Float64 = 1e-6
end

"""
    extend(basis::RBasis, new_snapshot, μ, ::NoCompress)

Extend the reduced basis by one snapshot without any orthogonalization or
compression procedure.
"""
function extend(basis::RBasis, new_snapshot::AbstractVector, μ, ::NoCompress)
    overlaps   = extend_overlaps(basis.snapshot_overlaps, basis.snapshots, new_snapshot)
    snapshots  = append!(copy(basis.snapshots), new_snapshot)
    parameters = append!(copy(basis.parameters), fill(μ, length(new_snapshot)))

    (; basis=RBasis(snapshots, parameters, I, overlaps, overlaps),
       keep=length(new_snapshot))
end

"""
    extend(basis::RBasis, new_snapshot::AbstractVector, μ, qrcomp::QRCompress)

Extend using QR orthonormalization and compression.

The orthonormalization is performed by QR decomposing the orthogonal projection
``\\bm{\\Psi}(\\bm{\\mu}_{n+1}) - B_n^\\dagger [B_n^\\dagger B_n]^{-1} B_n \\bm{\\Psi}(\\bm{\\mu}_n)``
and appending ``Q`` to snapshots. Modes that have an ``R`` column maximum falling below
the `qrcomp.tol` tolerance are dropped.
"""
function extend(basis::RBasis, new_snapshot::AbstractVector, μ, qrcomp::QRCompress)
    B = hcat(basis.snapshots...)
    Ψ = hcat(new_snapshot...)
    if VERSION ≥ v"1.7"  # Pivoted QR without deprecation warning
        fact = qr(Ψ - B * (cholesky(basis.metric) \ (B' * Ψ)), ColumnNorm())
    else
        fact = qr(Ψ - B * (cholesky(basis.metric) \ (B' * Ψ)), Val(true))
    end

    # Keep orthogonalized vectors of significant norm
    max_per_row = dropdims(maximum(abs, fact.R; dims=2); dims=2)
    keep        = findlast(max_per_row .> qrcomp.tol)
    isnothing(keep) && (return (; basis, keep))

    v      = [fact.Q[:, i] for i in 1:keep]
    v_norm = abs(fact.R[keep, keep])
    ext    = extend(basis, v, μ, NoCompress())

    (; basis=ext.basis, keep, v_norm)
end

"""
    extend(basis::RBasis, new_snapshot::AbstractVector, μ, ed::EigenDecomposition)

Extend the reduced basis by orthonormalizing and compressing via eigenvalue decomposition.

The overlap matrix ``S`` in `basis.snapshot_overlaps` is eigenvalue decomposed
``S = U^\\dagger \\Lambda U`` and orthonormalized by computing the vector coefficients
``V = U \\Lambda^{-1/2}``. Modes with an relative squared eigenvalue error smaller than
`ed.cutoff` are dropped.
"""
function extend(basis::RBasis, new_snapshot::AbstractVector, μ, ed::EigenDecomposition)
    overlaps = extend_overlaps(basis.snapshot_overlaps, basis.snapshots, new_snapshot)

    # Orthonormalization via eigenvalue decomposition
    Λ, U = eigen(Hermitian(overlaps))  # Hermitian to automatically sort by smallest λ
    Λ = abs.(Λ)  # Remove zero elements with negative sign
    λ_error_trunc = 0.0
    keep = 1
    if !iszero(ed.cutoff)
        λ²_psums      = reverse(cumsum(Λ.^2))  # Reverse to put largest eigenvector sum first
        λ²_errors     = @. sqrt(λ²_psums / λ²_psums[1])
        idx_trunc     = findlast(err -> err > ed.cutoff, λ²_errors)
        λ_error_trunc = λ²_errors[idx_trunc]
        keep          = idx_trunc - dimension(basis)
        if keep ≤ 0  # Return old basis, if no significant snapshots can be added
            return (; basis, keep, Λ, λ_error_trunc)
        end

        if keep != length(new_snapshot)  # Truncate/compress
            Λ = Λ[1:idx_trunc]
            U = U[1:idx_trunc, 1:idx_trunc]
            overlaps = overlaps[1:idx_trunc, 1:idx_trunc]
            λ_error_trunc = λ²_errors[idx_trunc + 1]
        end
    end
    snapshots   = append!(copy(basis.snapshots), new_snapshot[1:keep])  # TODO: use ordering of Λ
    parameters  = append!(copy(basis.parameters), fill(μ, keep))
    vectors_new = U * Diagonal(1 ./ sqrt.(Λ))

    (; basis=RBasis(snapshots, parameters, vectors_new,
                    overlaps, vectors_new' * overlaps * vectors_new),
     keep, Λ, λ_error_trunc)
end
