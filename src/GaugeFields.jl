module GaugeFields

using FFTW

export LightField,
    𝑈

mutable struct LightField
    ϵ::Real
    ϵc::Real
    χ::Real
end

function 𝑈(lf::LightField, xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    (;ϵ, ϵc) = lf
    U = Matrix{Float64}(undef, length(xs), length(ys))
    for (iy, y) in enumerate(ys)
        for (ix, x) in enumerate(xs)
            β₋ = sin((x-y)/√2); β₊ = sin((x+y)/√2)
            U[ix, iy] = (β₊^2 + (ϵc*β₋)^2) / 𝛼(lf, x, y)^2 * 2ϵ^2*(1+ϵc^2)
        end
    end
    return U
end

function 𝛼(lf::LightField, x::Real, y::Real)
    (;ϵ, ϵc, χ) = lf
    η₋ = cos((x-y)/√2); η₊ = cos((x+y)/√2)
    return ϵ^2 * (1 + ϵc^2) + η₊^2 + (ϵc*η₋)^2 - 2ϵc*η₊*η₋*cos(χ)  
end

function construct_H(lf::LightField)
    a = 2π  # wavelength, in units of 1/kᵣ
    L = a√2 # periodicity of the potential
    M = 32  # number of positive harmonics; coordinates will be discretised using 2M points
    x = range(0, L, 2M)
    dx = x[2] - x[1]
    
    U = 𝑈(lf, x, x) .* (dx/L)^2
    u = rfft(U)
    # filter small values
    max_u = maximum(abs2, u)
    threshold_factor = 0.001
    for i in eachindex(u)
        if abs2(u[i]) < threshold_factor * max_u
            u[i] = 0
        end
    end

    rows, cols, vals = fft_to_matrix(u)

end

"Based on results of a real 2D fft `u`, return `rows, cols, vals` tuple for constructing a sparse matrix."
function fft_to_matrix(u)
    M = size(u, 2) ÷ 2 # M/2 + 1 gives the size of each block; `size(u, 1)` gives the number of block-rows (= number of block-cols)
    rows, cols, vals = Int[], Int[], ComplexF64[]
    sizehint!(rows, (M+1)^4); sizehint!(cols, (M+1)^4); sizehint!(vals, (M+1)^4);

    for c_u in axes(u, 2), r_u in axes(u, 1) # iterate over columns and rows of `u`
        u[r_u, c_u] == 0 && continue
        val = u[r_u, c_u]
        for r_b in r_u:size(u, 1) # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `M`th
            c_b = r_b - r_u + 1 # block-column where to place the value
            if c_u <= M # for `c_u` ≤ `M`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal of the block
                for (r, c) in zip(c_u:M+1, 1:M+2-c_u)
                    push_vals!(rows, cols, vals; r_b, c_b, r, c, M, val)
                end
            elseif c_u == M+1 # for `c_u` = `M+1`, the value from `c_u`th column of `u` will be put to lower left and upper right corners of the block
                push_vals!(rows, cols, vals; r_b, c_b, r=M+1, c=1, M, val)
                r_b != c_b && push_vals!(rows, cols, vals; r_b, c_b, r=1, c=M+1, M, val)
            else # for `c_u` ≥ `M+2`, the value from `c_u`th column of `u` will be put to the `2M+2-c_u`th upper diagonal of the block
                if r_b != c_b # if `r_b == c_b`, then upper diagonal of the block has already been filled by pushing the conjugate element
                    c_u_inv = 2M+2 - c_u
                    for (r, c) in zip(1:M+2-c_u_inv, c_u_inv:M+1)
                        push_vals!(rows, cols, vals; r_b, c_b, r, c, M, val)
                    end
                end
            end
        end
    end
    return rows, cols, vals
end

"Push `val` and its complex-conjugate."
function push_vals!(rows, cols, vals; r_b, c_b, r, c, M, val)
    i = (r_b-1)*(M+1) + r
    j = (c_b-1)*(M+1) + c
    push!(rows, i)
    push!(cols, j)
    push!(vals, val)
    # conjugate
    push!(rows, j)
    push!(cols, i)
    push!(vals, val')
end

end