module GaugeFields

using FFTW

export GaugeField,
    𝑈

mutable struct GaugeField{Float<:AbstractFloat}
    ϵ::Float
    ϵc::Float
    χ::Float
    H_rows::Vector{Int}
    H_cols::Vector{Int}
    H_vals::Vector{Complex{Float}}
end

"Construct a `GaugeField` object."
function GaugeField(ϵ::Float, ϵc::Real, χ::Real) where {Float<:AbstractFloat}
    gf = GaugeField(ϵ, Float(ϵc), Float(χ), Int[], Int[], Complex{Float}[])
    constructH!(gf)
    return gf
end

"Return the 2D gauge potential 𝑈."
function 𝑈(gf::GaugeField{Float}, xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})  where {Float<:AbstractFloat}
    (;ϵ, ϵc) = gf
    U = Matrix{Float}(undef, length(xs), length(ys))
    for (iy, y) in enumerate(ys)
        for (ix, x) in enumerate(xs)
            β₋ = sin((x-y)/√2); β₊ = sin((x+y)/√2)
            U[ix, iy] = (β₊^2 + (ϵc*β₋)^2) / 𝛼(gf, x, y)^2 * 2ϵ^2*(1+ϵc^2)
        end
    end
    return U
end

"Helper function for calculating the gauge potential 𝑈."
function 𝛼(gf::GaugeField, x::Real, y::Real)
    (;ϵ, ϵc, χ) = gf
    η₋ = cos((x-y)/√2); η₊ = cos((x+y)/√2)
    return ϵ^2 * (1 + ϵc^2) + η₊^2 + (ϵc*η₋)^2 - 2ϵc*η₊*η₋*cos(χ)  
end

"Construct the Hamiltonian matrix by filling `gf.H_rows`, `gf.H_cols`, and `gf.H_vals`."
function constructH!(gf::GaugeField{Float}) where {Float<:AbstractFloat}
    a = 2π  # wavelength, in units of 1/kᵣ
    L = a√2 # periodicity of the potential
    M = 32  # number of positive harmonics; coordinates will be discretised using 2M points
    x = range(0, L, 2M)
    dx = x[2] - x[1]
    
    U = 𝑈(gf, x, x) .* (dx/L)^2
    u = rfft(U)
    u[1, 1] = 0 # remove the secular component -- it has no physical significance but breaks the structure in `filter_count!` if included
    n_elem = filter_count!(u, factor=1e-3) # filter small values and calculate the number of elements in the final Hamiltonian
    n_elem += (M+1)^2 # reserve space for the diagonal elements coming from the other terms of the Hamiltonian
    
    gf.H_rows = Vector{Int}(undef, n_elem)
    gf.H_cols = Vector{Int}(undef, n_elem)
    gf.H_vals = Vector{Float}(undef, n_elem)
    
    fft_to_matrix!(gf.H_rows, gf.H_cols, gf.H_vals, u)
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

end