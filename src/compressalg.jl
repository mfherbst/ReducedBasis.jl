struct QRCompress
    full_orthogonalize::Bool
    tol_qr::Float64
end
function QRCompress(; full_orthogonalize=false, tol_qr=1e-10)
    QRCompress( full_orthogonalize, tol_qr)
end
