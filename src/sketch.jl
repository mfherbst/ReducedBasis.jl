# T -> floating-point type
# P -> type of the parameter vector (e.g. SVector{3, T})

# Represents an ∑_i α_i(μ) M_i
struct AffineDecomposition{D,M,F}
    terms::Array{M,D}
    # is a function mapping from a parameter point μ_i to the vector
    # of coeffients for each term
    coeffientmap::F
end
(d::AffineDecomposition)(μ) = sum(d.coeffientmap(μ) .* d.terms)


struct ReducedBasis{T <: AbstractFloat, P <: AbstractVector, M <: AbstractMatrix{T}, N <: AbstractMatrix{T}}
    # Column-wise truth solves yᵢ at certain parameter values μᵢ
    truthsolves::M  # = Y
    # Parameter values μᵢ associated with truth solve yᵢ
    snapshot_locations::Vector{P}

    # Coefficients in the truth solves making up the reduced basis vectors
    # as truthsolves * vectors
    vectors::N  # = V
    # overlap between basis vectors, equivalent to (V'*Y'*Y*V)
    overlap::Matrix{T}  # = V' * Y' * Y * V
end

function compress(basis::ReducedBasis, a::AffineDecomposition)
    AffineDecomposition([compress(basis, term) for term in a.terms], a.coeffientmap)
end

function compress(basis::ReducedBasis, m::AbstractMatrix)
    basis.truthsolves' * basis.vectors' * m * basis.vectors * basis.truthsolves
end

struct Greedy
    # error metric -> residual squared
end

struct Lobpcg
    # preconditioner ??
end

function assemble(model, greedy::Greedy; tol, truthsolver=Lobpcg(), )
    H = hamiltonian(model)


    Hrbij = AffineDecomposition([
            # B'*Hi'*Hj*B
    ], μ -> (H.coefficientmap(μ) * H.coefficientmap(μ)'))


    newbasis = extend_basis(basis, truthsolver, ...)


    # ....
    basis::ReducedBasis

    (; h=compress(basis, H), basis)
end


function offline()
    model = ...
    ares = assemble(model, Greedy( ... ))

    observable = ...
    (; ares.h, ares.basis.overlap, )
end




dot(v::AbstractVector, d::AffineDecomposition, w::AbstractVector)::AffineDecomposition





A * (∑_i α_i(μ) M_i) * B   = ∑_i α_i(μ) (A * M_i * B)
