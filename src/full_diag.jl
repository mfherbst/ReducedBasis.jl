struct FullDiagonalization
    n_states_max::Int
    tol_degeneracy::Float64
    full_orthogonalize::Bool
    tol_qr::Float64
end
function FullDiagonalization(; n_states_max=1, tol_degeneracy=0.0,
                             full_orthogonalize=false, tol_qr=1e-10)
    FullDiagonalization(n_states_max, tol_degeneracy, full_orthogonalize, tol_qr)
end

function truth_solve(H::AffineDecomposition, μ, Ψ₀, fd::FullDiagonalization)
    Λ, Ψ = eigen(Hermitian(H(μ)), 1:fd.n_states_max)
    n_target = fd.n_states_max
    if fd.tol_degeneracy > 0.0
        n_target = findlast(λ -> abs(λ - Λ[fd.n_states_max]) < fd.tol_degeneracy, Λ)
    end
    (values=Λ[1:n_target], vectors=Ψ[:, 1:n_target])
end

function online_solve(h::AffineDecomposition, b::Matrix, μ, fd::FullDiagonalization)
    Λ, φ = eigen(Hermitian(h(μ)), Hermitian(b))
    n_target = fd.n_states_max
    if fd.tol_degeneracy > 0.0
        n_target = findlast(λ -> abs(λ - Λ[fd.n_states_max]) < fd.tol_degeneracy, Λ)
    end
    (values=Λ[1:n_target], vectors=φ[:, 1:n_target])
end