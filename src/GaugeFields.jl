module GaugeFields

using FFTW, SparseArrays, KrylovKit

export GaugeField,
    𝑈,
    spectrum,
    FloquetGaugeField

struct GaugeField{Float<:AbstractFloat}
    ϵ::Float
    ϵc::Float
    χ::Float
    δ::Tuple{Float,Float} # shift (δ𝑥, δ𝑦)
    u₀₀::Float # zeroth harmonic (= average) of 𝑈
    H_rows::Vector{Int}
    H_cols::Vector{Int}
    H_vals::Vector{Complex{Float}}
end

"""
Construct a `GaugeField` object.
`n_harmonics` is the number of positive harmonics; coordinates will be discretised using `2n_harmonics` points.
"""
function GaugeField(ϵ::Float, ϵc::Real, χ::Real, δ::Tuple{<:Real,<:Real}=(0, 0); n_harmonics::Integer=32, fft_threshold::Real=1e-2) where {Float<:AbstractFloat}
    H, u₀₀ = constructH(ϵ, ϵc, χ, δ, n_harmonics, fft_threshold)
    return GaugeField(ϵ, Float(ϵc), Float(χ), Float.(δ), u₀₀, H...)
end

"Return the 2D gauge potential 𝑈."
function 𝑈(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real}; ϵ::Real, ϵc::Real, χ::Real)
    U = Matrix{typeof(ϵ)}(undef, length(xs), length(ys))
    for (iy, y) in enumerate(ys)
        for (ix, x) in enumerate(xs)
            β₋ = sin(x-y); β₊ = sin(x+y)
            U[ix, iy] = (β₊^2 + (ϵc*β₋)^2) / 𝛼(x, y; ϵ, ϵc, χ)^2 * 2ϵ^2 * (1+ϵc^2)
        end
    end
    return U
end

"Helper function for calculating the gauge potential 𝑈."
function 𝛼(x::Real, y::Real; ϵ, ϵc, χ)
    η₋ = cos(x-y); η₊ = cos(x+y)
    return ϵ^2 * (1 + ϵc^2) + η₊^2 + (ϵc*η₋)^2 - 2ϵc*η₊*η₋*cos(χ)  
end

"""
Construct the Hamiltonian matrix by filling `gf.H_rows`, `gf.H_cols`, and `gf.H_vals`.
Coordinates will be discretised using 2M points, yielding spatial harmonics from `-M`th to `M`th.
"""
function constructH(ϵ::Float, ϵc::Real, χ::Real, δ::Tuple{<:Real,<:Real}, M::Integer, fft_threshold::Real) where {Float<:AbstractFloat}
    L = π # periodicity of the potential
    dx = Float(L / 2M)
    x = range(0, L-dx, 2M)
    U = 𝑈(x, x; ϵ, ϵc, χ) .* (dx/L)^2
    u = rfft(U) |> real # guaranteed to be real (and even) because `U` is real and even
    n_elem = filter_count!(u, factor=fft_threshold) # filter small values and calculate the number of elements in the final Hamiltonian
    
    H_rows = Vector{Int}(undef, n_elem)
    H_cols = Vector{Int}(undef, n_elem)
    H_vals = Vector{Complex{Float}}(undef, n_elem)
    u₀₀ = u[1, 1] # save the secular component.
    u[1, 1] = 0 # remove because it breaks the structure in `filter_count!` if included
    fft_to_matrix!(H_rows, H_cols, H_vals, u, δ)
    
    n_diag = (M+1)^2 # number of diagonal elements in 𝐻
    # fill positions of the diagonal elements
    H_rows[end-n_diag+1:end] .= 1:n_diag
    H_cols[end-n_diag+1:end] .= 1:n_diag
    H_vals[end-n_diag+1:end] .= Inf # push Inf's to later locate the diagonal values in `nonzeros(H)` easily

    return (H_rows, H_cols, H_vals), u₀₀
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
            if c < M+1
                n_elem += (N - (r-1)) * (M+1 - (c-1)) # number of blocks in which `u[r, c]` will appear × number of times it will appear within each block
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

"""
Based on results of a real 2D fft `u`, return `rows, cols, vals` tuple for constructing a sparse matrix.
Optionally, a tuple `δ` of shifts in 𝑥 and 𝑦 directions can be supplied.
"""
function fft_to_matrix!(rows, cols, vals, u, δ::Tuple{<:Real,<:Real})
    L = π # periodicity of the potential
    M = size(u, 2) ÷ 2 # M + 1 gives the size of each block; `size(u, 1)` gives the number of block-rows (= number of block-cols)
    counter = 1

    # it is assumed that u[1, 1] == 0 -- otherwise, one would also need to prevent double pushing of the diagonal elements
    for c_u in axes(u, 2), r_u in axes(u, 1) # iterate over columns and rows of `u`
        u[r_u, c_u] == 0 && continue
        e = c_u <= M ÷ 2 ? cispi(2/L*(r_u-1)*δ[1]) : cispi(2/L*(2M-r_u-1)*δ[1])
        val = u[r_u, c_u] * e * cispi(2/L*(r_u-1)*δ[2])
        for r_b in r_u:size(u, 1) # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `M+1`th. For actual applications, `size(u, 1) == M+1`
            c_b = r_b - r_u + 1 # block-column where to place the value
            if c_u <= M # for `c_u` ≤ `M`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal of the block
                for (r, c) in zip(c_u:M+1, 1:M+2-c_u)
                    push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize=M+1, val)
                    counter += 2
                end
            elseif c_u == M+1 # for `c_u` = `M+1`, the value from `c_u`th column of `u` will be put to lower left and upper right corners of the block
                push_vals!(rows, cols, vals, counter; r_b, c_b, r=M+1, c=1, blocksize=M+1, val)
                counter += 2
                if r_b != c_b
                    push_vals!(rows, cols, vals, counter; r_b, c_b, r=1, c=M+1, blocksize=M+1, val) # if we're in the diagonal block, then the upper right corner is conjugate to lower left and has already been pushed
                    counter += 2
                end
            else # for `c_u` ≥ `M+2`, the value from `c_u`th column of `u` will be put to the `2M+2-c_u`th upper diagonal of the block
                if r_b != c_b # if `r_b == c_b`, then upper diagonal of the block has already been filled by pushing the conjugate element
                    c_u_inv = 2M+2 - c_u
                    for (r, c) in zip(1:M+2-c_u_inv, c_u_inv:M+1)
                        push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize=M+1, val)
                        counter += 2
                    end
                end
            end
        end
    end
end

"""
Push value `val` stored at (`r`, `c`) in some matrix to the block (`r_b`, `c_b`) of a sparse matrix encoded in `rows`, `cols`, `vals`.
`counter` shows where to push. The complex-conjugate element is also pushed.
"""
function push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize, val)
    i = (r_b-1)*blocksize + r
    j = (c_b-1)*blocksize + c
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
            H_vals[diagidx[(nx-1)M+ny]] = gf.u₀₀ + qx^2 + qy^2 + 4(qx*nx + qy*ny) + 4(nx^2 + ny^2)
        end
        vals, _, _ = eigsolve(H, 1, :SR, tol=(Float == Float32 ? 1e-6 : 1e-12))
        E[i, j] = E[j, i] = vals[1]
    end
    return E
end

struct FloquetGaugeField{Float<:AbstractFloat}
    Q_rows::Vector{Int}
    Q_cols::Vector{Int}
    Q_vals::Vector{Complex{Float}}
    gaugefields::Vector{GaugeField{Float}}
end

"Construct a `FloquetGaugeField` object. The cell edges will be reduced `subfactor` times."
function FloquetGaugeField(ϵ::Float, ϵc::Real, χ::Real, ω::Real, qx::Real, qy::Real; subfactor::Integer, n_floquet_harmonics=10, n_fourier_harmonics::Integer=32, fft_threshold::Real=1e-2) where {Float<:AbstractFloat}
    L = π # periodicity of the potential
    δ = L / subfactor
    gaugefields = Vector{GaugeField{Float}}(undef, subfactor^2)
    M = n_fourier_harmonics + 1 # size of the blocks in 𝐻
    for i in 0:subfactor-1, j in 0:subfactor-1
        gf = GaugeField(ϵ, ϵc, χ, (δ*i, δ*j); n_harmonics=n_fourier_harmonics, fft_threshold)
        diagidx = findall(==(Inf), gf.H_vals) # find indices of diagonal elements
        for nx in 1:M, ny in 1:M
            gf.H_vals[diagidx[(nx-1)M+ny]] = qx^2 + qy^2 + 4(qx*nx + qy*ny) + 4(nx^2 + ny^2)
        end
        gaugefields[i*subfactor+j+1] = gf
    end
    Q = constructQ(gaugefields, ω, n_floquet_harmonics)
    return FloquetGaugeField(Q..., gaugefields)
end

"Construct the quasienergy operator using temporal harmonics from `-M`th to `M`th."
function constructQ(gaugefields::Vector{GaugeField{Float}}, ω::Real, M::Integer) where Float<:AbstractFloat
    blocksize = maximum(gaugefields[1].H_rows) # size of 𝐻
    N = length(gaugefields) # number of steps in the driving sequence
    n_elems = length(gaugefields[1].H_vals) * (M+1)^2 # total number of elements in 𝑄: (M+1)^2 blocks each holding `length(gaugefields[1].H_vals)` values
    Q_rows = Vector{Int}(undef, n_elems)
    Q_cols = Vector{Int}(undef, n_elems)
    Q_vals = Vector{Complex{Float}}(undef, n_elems)
    block_vals = similar(gaugefields[1].H_vals) # for storing summed stroboscopic drives
    
    # block-diagonal of 𝑄
    for i in eachindex(block_vals)
        block_vals[i] = sum(gaugefields[j].H_vals[i] for j in 1:N)
    end
    counter = 1
    fill_blockband!(Q_rows, Q_cols, Q_vals, gaugefields[1].H_rows, gaugefields[1].H_cols, block_vals, 0, blocksize, M+1, counter, ω)
    counter += (M+1) * length(gaugefields[1].H_vals)
    # all remaining block-bands
    for m in 1:M
        for i in eachindex(block_vals)
            block_vals[i] = sum(gaugefields[j].H_vals[i] * cispi(-2m*j/N) for j in 1:N) * (cispi(2m/N)-1) / (2π*im*m)
        end
        fill_blockband!(Q_rows, Q_cols, Q_vals, gaugefields[1].H_rows, gaugefields[1].H_cols, block_vals, m, blocksize, M+1, counter, 0)
        counter += 2(M+1-m) * length(gaugefields[1].H_vals)
    end
    return Q_rows, Q_cols, Q_vals
end

"Fill the `m`th block-off-diagonal of `Q` with matrices `q`. `counter` shows where to start pushing."
function fill_blockband!(Q_rows, Q_cols, Q_vals, q_rows, q_cols, q_vals, m, blocksize, nblockrows, counter, ω)
    if m == 0 # for the block-diagonal can't use `push_vals`
        for i in eachindex(q_vals) # take a value and put it into all required blocks
            for r_b in 0:nblockrows-1 # for each diagonal block whose block-coordinates are (r_b, r_b)
                Q_rows[counter] = r_b*blocksize + q_rows[i]
                Q_cols[counter] = r_b*blocksize + q_cols[i]
                Q_vals[counter] = q_vals[i]
                q_rows[i] == q_cols[i] && (Q_vals[counter] += (r_b - nblockrows÷2) * ω)
                counter += 1
            end
        end
    else
        for i in eachindex(q_vals) # take a value and put it into all required blocks
            for (r_b, c_b) in zip(m+1:nblockrows, 1:nblockrows-m) # for each block whose block-coordinates are (r_b, c_b)
                push_vals!(Q_rows, Q_cols, Q_vals, counter; r_b, c_b, r=q_rows[i], c=q_cols[i], blocksize, val=q_vals[i])
                counter += 2
            end
        end
    end
end

end