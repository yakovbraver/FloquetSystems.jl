module GaugeFields

using FFTW, SparseArrays, KrylovKit

export GaugeField,
    𝑈,
    spectrum

mutable struct GaugeField{Float<:AbstractFloat}
    ϵ::Float
    ϵc::Float
    χ::Float
    H_rows::Vector{Int}
    H_cols::Vector{Int}
    H_vals::Vector{Float}
end

"""
Construct a `GaugeField` object.
`n_harmonics` is the number of positive harmonics; coordinates will be discretised using `2n_harmonics` points.
"""
function GaugeField(ϵ::Float, ϵc::Real, χ::Real; n_harmonics::Integer=32, fft_threshold::Real=1e-2) where {Float<:AbstractFloat}
    gf = GaugeField(ϵ, Float(ϵc), Float(χ), Int[], Int[], Float[])
    constructH!(gf, n_harmonics, fft_threshold)
    return gf
end

"Return the 2D gauge potential 𝑈."
function 𝑈(gf::GaugeField{Float}, xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real}) where {Float<:AbstractFloat}
    (;ϵ, ϵc) = gf
    U = Matrix{Float}(undef, length(xs), length(ys))
    for (iy, y) in enumerate(ys)
        for (ix, x) in enumerate(xs)
            β₋ = sin(x-y); β₊ = sin(x+y)
            U[ix, iy] = (β₊^2 + (ϵc*β₋)^2) / 𝛼(gf, x, y)^2 * 2ϵ^2*(1+ϵc^2)
        end
    end
    return U
end

"Helper function for calculating the gauge potential 𝑈."
function 𝛼(gf::GaugeField, x::Real, y::Real)
    (;ϵ, ϵc, χ) = gf
    η₋ = cos(x-y); η₊ = cos(x+y)
    return ϵ^2 * (1 + ϵc^2) + η₊^2 + (ϵc*η₋)^2 - 2ϵc*η₊*η₋*cos(χ)  
end

"""
Construct the Hamiltonian matrix by filling `gf.H_rows`, `gf.H_cols`, and `gf.H_vals`.
`M` is the number of positive harmonics; coordinates will be discretised using 2M points.
"""
function constructH!(gf::GaugeField{Float}, M::Integer, fft_threshold::Real) where {Float<:AbstractFloat}
    L = π # periodicity of the potential
    dx = L / 2M
    x = range(0, L-dx, 2M)
    U = 𝑈(gf, x, x) .* (dx/L)^2
    u = rfft(U) |> real # guaranteed to be real (and even) because `U` is real and even
    u[1, 1] = 0 # remove the secular component -- it has no physical significance but breaks the structure in `filter_count!` if included
    n_elem = filter_count!(u, factor=fft_threshold) # filter small values and calculate the number of elements in the final Hamiltonian
    
    n_diag = (M+1)^2 # number of diagonal elements in 𝐻
    n_elem += n_diag # reserve space for the diagonal elements coming from the other terms of the Hamiltonian
    
    gf.H_rows = Vector{Int}(undef, n_elem)
    gf.H_cols = Vector{Int}(undef, n_elem)
    gf.H_vals = Vector{Float}(undef, n_elem)
    fft_to_matrix!(gf.H_rows, gf.H_cols, gf.H_vals, u)

    # fill positions of the diagonal elements
    gf.H_rows[end-n_diag+1:end] .= 1:n_diag
    gf.H_cols[end-n_diag+1:end] .= 1:n_diag
    gf.H_vals[end-n_diag+1:end] .= Inf # push Inf's to later locate the diagonal values in `nonzeros(H)` easily
end

"""
Set to zero values of `u` that are `threshold` times smaller (by absolute magnitude) than the largest.
Based on the resulting number of nonzero elements in `u`, count the number of values that will be stored in 𝐻.
"""
function filter_count!(u::AbstractMatrix{<:Number}; factor::Real)
    n_elem = 0
    M = size(u, 2) ÷ 2
    N = size(u, 1) # if `u` is really the result of `rfft`, then `N == M+1`, but we keep the calculation a bit more general
    max_u = maximum(abs, u)
    # do the first row of `u`, i.e. the diagonal blocks of 𝐻, separately
    for c in axes(u, 2)
        r = 1
        if abs(u[r, c]) < factor * max_u
            u[r, c] = 0
        else
            if c == 1
                n_elem += (N - (r-1)) * (M+1 - (c-1)) # number of blocks in which `u[r, c]` will appear × number of times it will appear within each block
            elseif c < M+1
                n_elem += (N - (r-1)) * (M+1 - (c-1))
            elseif c == M+1
                n_elem += 2(N - (r-1)) * (M+1 - (c-1))
            else
                n_elem += (N - (r-1)) * (c - M)
            end
        end
    end
    for c in axes(u, 2), r in 2:size(u, 1)
        if abs(u[r, c]) < factor * max_u
            u[r, c] = 0
        else
            if c < M+1
                n_elem += 2(N - (r-1)) * (M+1 - (c-1))
            elseif c == M+1
                n_elem += 4(N - (r-1)) * (M+1 - (c-1))
            else
                n_elem += 2(N - (r-1)) * (c - M)
            end
        end
    end
    return n_elem
end

"Based on results of a real 2D fft `u`, return `rows, cols, vals` tuple for constructing a sparse matrix."
function fft_to_matrix!(rows, cols, vals, u)
    M = size(u, 2) ÷ 2 # M/2 + 1 gives the size of each block; `size(u, 1)` gives the number of block-rows (= number of block-cols)
    counter = 1

    # it is assumed that u[1, 1] == 0 -- otherwise, one would also need to prevent double pushing of the diagonal elements
    for c_u in axes(u, 2), r_u in axes(u, 1) # iterate over columns and rows of `u`
        u[r_u, c_u] == 0 && continue
        val = u[r_u, c_u]
        for r_b in r_u:size(u, 1) # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `M+1`th. For actual applications, `size(u, 1) == M+1`
            c_b = r_b - r_u + 1 # block-column where to place the value
            if c_u <= M # for `c_u` ≤ `M`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal of the block
                for (r, c) in zip(c_u:M+1, 1:M+2-c_u)
                    push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, M, val)
                    counter += 2
                end
            elseif c_u == M+1 # for `c_u` = `M+1`, the value from `c_u`th column of `u` will be put to lower left and upper right corners of the block
                push_vals!(rows, cols, vals, counter; r_b, c_b, r=M+1, c=1, M, val)
                counter += 2
                if r_b != c_b
                    push_vals!(rows, cols, vals, counter; r_b, c_b, r=1, c=M+1, M, val) # if we're in the diagonal block, then the upper right corner is conjugate to lower left and has already been pushed
                    counter += 2
                end
            else # for `c_u` ≥ `M+2`, the value from `c_u`th column of `u` will be put to the `2M+2-c_u`th upper diagonal of the block
                if r_b != c_b # if `r_b == c_b`, then upper diagonal of the block has already been filled by pushing the conjugate element
                    c_u_inv = 2M+2 - c_u
                    for (r, c) in zip(1:M+2-c_u_inv, c_u_inv:M+1)
                        push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, M, val)
                        counter += 2
                    end
                end
            end
        end
    end
end

"Push `val` and its complex-conjugate."
function push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, M, val)
    i = (r_b-1)*(M+1) + r
    j = (c_b-1)*(M+1) + c
    rows[counter] = i
    cols[counter] = j
    vals[counter] = val
    # conjugate
    rows[counter+1] = j
    cols[counter+1] = i
    vals[counter+1] = val'
end

"""
Calculate ground state energies.
Return the enegy matrix for a quarter of the BZ, since ℤ₄ symmetry is assumed.
"""
function spectrum(gf::GaugeField{Float}, n_q::Integer) where {Float<:AbstractFloat}
    E = Matrix{Float}(undef, n_q, n_q)

    H = sparse(gf.H_rows, gf.H_cols, gf.H_vals)
    H_vals = nonzeros(H)
    diagidx = findall(==(Inf), H_vals) # find indices of diagonal elements

    M = Int(√size(H, 1))
    qs = range(0, 1, length=n_q) # BZ is (-1 ≤ 𝑞ₓ, 𝑞𝑦 ≤ 1), but it's enough to consider a triangle 0 ≤ 𝑞ₓ ≤ 1, 0 ≤ 𝑞𝑦 ≤ 𝑞ₓ
    for (j, qx) in enumerate(qs), i in j:n_q
        qy = qs[i]
        for nx in 1:M, ny in 1:M
            H_vals[diagidx[(nx-1)M+ny]] = qx^2 + qy^2 + 4(qx*nx + qy*ny) + 4(nx^2 + ny^2)
        end
        vals, _, _ = eigsolve(H, 1, :SR, tol=(Float == Float32 ? 1e-6 : 1e-12))
        E[i, j] = E[j, i] = vals[1]
    end
    return E
end

end