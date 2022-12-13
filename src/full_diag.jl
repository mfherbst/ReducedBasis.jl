struct FullDiagonalization
    n_target::Int
    tol_degeneracy::Float64
end
function FullDiagonalization(; n_target=1, tol_degeneracy=0.0)
    FullDiagonalization(n_target, tol_degeneracy)
end

function solve(H::AffineDecomposition, μ, Ψ₀, fd::FullDiagonalization)
    Λ, Ψ = eigen(Hermitian(H(μ)), 1:fd.n_target)
    # TODO: n_target correct terminology here?
    n_target = fd.n_target
    if fd.tol_degeneracy > 0.0
        n_target = findlast(abs.(Λ .- Λ[1]) .< fd.tol_degeneracy)
    end
    (values=Λ[1:n_target], vectors=Ψ[:, 1:n_target])
end

function solve(h::AffineDecomposition, b::Matrix, μ, fd::FullDiagonalization)
    # TODO: force Hermitian?
    Λ, φ = eigen(Hermitian(h(μ)), Hermitian(b))
    n_target = fd.n_target
    if fd.tol_degeneracy > 0.0
        n_target = findlast(abs.(Λ .- Λ[1]) .< fd.tol_degeneracy)
    end
    (values=Λ[1:n_target], vectors=φ[:, 1:n_target])
end