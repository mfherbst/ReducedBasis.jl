struct FullDiagonalization
    n_target::Int
    tol_degeneracy::Float64
end
function FullDiagonalization(; n_target=1, tol_degeneracy=0.0)
    FullDiagonalization(n_target, tol_degeneracy)
end

function solve(H::AffineDecomposition, μ, _, fd::FullDiagonalization)
    H_matrix = Hermitian(H(μ))  # Hamiltonian are Hermitian by assumption
    # Convert to dense matrix if sparse
    issparse(H_matrix) && (H_matrix = Hermitian(Matrix(H_matrix)))
    Λ, Ψ = eigen(H_matrix, 1:fd.n_target)
    n_target = fd.n_target
    if fd.tol_degeneracy > 0.0
        n_target = findlast(abs.(Λ .- Λ[1]) .< fd.tol_degeneracy)
    end
    (values=Λ[1:n_target], vectors=Ψ[:, 1:n_target])
end
# TODO: combine into one function using args...?
function solve(h::AffineDecomposition, b::Matrix, μ, fd::FullDiagonalization)
    Λ, φ = eigen(Hermitian(h(μ)), Hermitian(b)) 
    n_target = fd.n_target
    if fd.tol_degeneracy > 0.0
        n_target = findlast(abs.(Λ .- Λ[1]) .< fd.tol_degeneracy)
    end
    (values=Λ[1:n_target], vectors=φ[:, 1:n_target])
end