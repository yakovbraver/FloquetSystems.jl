module GaugeFields

using FFTW, SparseArrays, Arpack, KrylovKit, DelimitedFiles
using LinearAlgebra: eigvals, Hermitian, diagind, eigen

export GaugeField,
    𝑈,
    𝐴,
    spectrum,
    make_wavefunction,
    q0_states,
    FloquetGaugeField

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

"Return the 2D vector potential 𝐴(𝑥, 𝑦) as a matrix of tuples of 𝑥- and 𝑦-components. The values are normalised to length given by `normalisation`."
function 𝐴(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real}; ϵ::Real, ϵc::Real, χ::Real, normalisation::Real=1)
    A = [(sin(2y), sin(2x)) .* ϵc .* sin(χ) ./ 𝛼(x, y; ϵ, ϵc, χ) for y in ys, x in xs]
    max_A = √maximum(x -> x[1]^2 + x[2]^2, A)
    map!(x -> x ./ max_A .* normalisation, A, A)
    return A
end

"Helper function for calculating the gauge potential 𝑈."
function 𝛼(x::Real, y::Real; ϵ::Real, ϵc::Real, χ::Real)
    η₋ = cos(x-y); η₊ = cos(x+y)
    return ϵ^2 * (1 + ϵc^2) + η₊^2 + (ϵc*η₋)^2 - 2ϵc*η₊*η₋*cos(χ)  
end

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
    if isodd(n_harmonics)
        @warn "`n_harmonics` must be even. Reducing `n_harmonics` by one."
        n_harmonics -= 1
    end
    H, u₀₀ = constructH(ϵ, ϵc, χ, δ, n_harmonics, fft_threshold)
    return GaugeField(ϵ, Float(ϵc), Float(χ), Float.(δ), u₀₀, H...)
end

"""
Construct the Hamiltonian matrix by filling `gf.H_rows`, `gf.H_cols`, and `gf.H_vals`.
Coordinates will be discretised using 2M points, yielding spatial harmonics from `-M`th to `M`th.
The resulting Hamiltonian will be (M+1) × (M+1).
"""
function constructH(ϵ::Float, ϵc::Real, χ::Real, δ::Tuple{<:Real,<:Real}, M::Integer, fft_threshold::Real) where {Float<:AbstractFloat}
    L = π # periodicity of the potential
    dx = Float(L / 2M) # without a cast, `U` below will always be Float64 
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
    H_vals[end-n_diag+1:end] .= 0 # mark with zeros to later locate the diagonal values in `nonzeros(H)` easily. `fft_to_matrix!` does not save the zero entries so that the only zeros will be the diagonal ones

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
        e = c_u <= M+1 ? cispi(2/L*(c_u-1)*δ[1]) : cispi(2/L*(c_u-(2M+1))*δ[1])
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
Calculate ground state energy dispersion for a quarter of the BZ, since ℤ₄ symmetry is assumed.
This quarter is discretised with `n_q` points in each direction.
"""
function spectrum(gf::GaugeField{Float}; n_q::Integer) where {Float<:AbstractFloat}
    E = Matrix{Float}(undef, n_q, n_q)

    H = sparse(gf.H_rows, gf.H_cols, gf.H_vals)
    H_vals = nonzeros(H)
    diagidx = findall(==(0), H_vals) # find indices of diagonal elements -- we saved zeros there (see `constructH`)

    B = Int(√size(H, 1)) # size of each block in `H`
    j_max = (B - 1) ÷ 2  # index for each block in `H` will run in `-j_max:j_max`, giving `B` values in total
    qs = range(0, 1, length=n_q) # BZ is (-1 ≤ 𝑞ₓ, 𝑞𝑦 ≤ 1), but it's enough to consider a triangle 0 ≤ 𝑞ₓ ≤ 1, 0 ≤ 𝑞𝑦 ≤ 𝑞ₓ
    for (iqx, qx) in enumerate(qs), iqy in iqx:n_q
        qy = qs[iqy]
        for (j, jx) in enumerate(-j_max:j_max), (i, jy) in enumerate(-j_max:j_max)
            H_vals[diagidx[(j-1)B+i]] = gf.u₀₀ + qx^2 + qy^2 + 4(qx*jx + qy*jy) + 4(jx^2 + jy^2)
        end
        # vals, _, _ = eigsolve(H, 1, :SR, tol=(Float == Float32 ? 1e-6 : 1e-12))
        vals = eigvals(Hermitian(Matrix(H)))
        E[iqy, iqx] = E[iqx, iqy] = vals[1]
    end
    return E
end

"""
Calculate energies and wavefunctions at zero quasimomenta for `nbands` lowest bands.
"""
function q0_states(gf::GaugeField{Float}) where {Float<:AbstractFloat}
    H = sparse(gf.H_rows, gf.H_cols, gf.H_vals) |> Matrix |> Hermitian
    B = Int(√size(H, 1)) # size of each block in `H`
    j_max = (B - 1) ÷ 2  # index for each block in `H` will run in `-j_max:j_max`, giving `B` values in total
    H[diagind(H)] .= [gf.u₀₀ + 4(jx^2 + jy^2) for jx in -j_max:j_max for jy in -j_max:j_max]
    f = eigen(H)
    return f.values, f.vectors
end

"""
Construct wavefunction `wf` on a grid having `nx` points in `x` and `y` direction. Use odd `nx` for nice results.
`v` is one of the vectors output from [`q0_states`](@ref).
Return (`xs`, `wf`), where `xs` is the grid in one dimension.
"""
function make_wavefunction(v::AbstractVector{<:Number}, nx::Integer)
    B = Int(√length(v))
    j_max = (B - 1) ÷ 2
    L = π # spatial period
    xs = range(0, L, nx)
    wf = Matrix{eltype(v)}(undef, nx, nx)
    for (iy, y) in enumerate(xs), (ix, x) in enumerate(xs)
        wf[ix, iy] = sum(v[(j-1)B+i]cis(2jx*x + 2jy*y) for (j, jx) in enumerate(-j_max:j_max)
                                                       for (i, jy) in enumerate(-j_max:j_max)) / L
    end
    return xs, wf
end

"""
Calculate energy dispersion for `nstates` lowest states for all pairs of quasimomenta described by `qxs` and `qys`.
"""
function spectrum(gf::GaugeField{Float}, qxs::AbstractVector{<:Real}, qys::AbstractVector{<:Real}; nsaves::Integer) where {Float<:AbstractFloat}
    E = Array{Float}(undef, nsaves, length(qxs), length(qys))

    H = sparse(gf.H_rows, gf.H_cols, gf.H_vals)
    H_vals = nonzeros(H)
    diagidx = findall(==(0), H_vals) # find indices of diagonal elements -- we saved zeros there (see `constructH`)

    M = Int(√size(H, 1)) # size of each block in `H`
    j_max = (M - 1) ÷ 2  # index for each block in `H` will run in `-j_max:j_max`, giving `M` values in total
    for (iqy, qy) in enumerate(qys), (iqx, qx) in enumerate(qxs)
        for (j, jx) in enumerate(-j_max:j_max), (i, jy) in enumerate(-j_max:j_max)
            H_vals[diagidx[(j-1)M+i]] = gf.u₀₀ + qx^2 + qy^2 + 4(qx*jx + qy*jy) + 4(jx^2 + jy^2)
        end
        # vals, _, _ = eigsolve(H, nsaves, :SR, tol=(Float == Float32 ? 1e-6 : 1e-12))
        vals = eigvals(Hermitian(Matrix(H)))
        E[:, iqx, iqy] = vals[1:nsaves]
    end
    return E
end

struct FloquetGaugeField{Float<:AbstractFloat}
    Q_rows::Vector{Int}
    Q_cols::Vector{Int}
    Q_vals::Vector{Complex{Float}}
    u₀₀::Float # zeroth harmonic (= average) of 𝑈
    blocksize::Int # size of 𝐻ₘ
    M::Integer # Floquet harmonic number (will use temporal harmonics from `-M`th to `M`th.)
end

"Construct a `FloquetGaugeField` object. The cell edges will be reduced `subfactor` times. It is better to take even `n_floquet_harmonics`."
function FloquetGaugeField(ϵ::Float, ϵc::Real, χ::Real; subfactor::Integer, n_floquet_harmonics=10, n_spatial_harmonics::Integer=32, fft_threshold::Real=1e-2) where {Float<:AbstractFloat}
    if isodd(n_spatial_harmonics)
        @warn "`n_spatial_harmonics` must be even. Reducing `n_spatial_harmonics` by one."
        n_spatial_harmonics -= 1
    end
    if isodd(n_floquet_harmonics)
        @warn "`n_floquet_harmonics` must be even. Reducing `n_floquet_harmonics` by one."
        n_floquet_harmonics -= 1
    end
    L = π # periodicity of the potential
    δ = L / subfactor
    gaugefields = [GaugeField(ϵ, ϵc, χ, (δ*i, δ*j); n_harmonics=n_spatial_harmonics, fft_threshold) for i in 0:subfactor-1 for j in 0:subfactor-1]
    Q = constructQ(gaugefields, n_floquet_harmonics)
    return FloquetGaugeField(Q..., gaugefields[1].u₀₀, (n_spatial_harmonics+1)^2, n_floquet_harmonics)
end

"""
Construct the quasienergy operator using temporal harmonics from `-M`th to `M`th."
The resulting Hamiltonian will have M+1 blocks.
"""
function constructQ(gaugefields::Vector{GaugeField{Float}}, M::Integer) where Float<:AbstractFloat
    blocksize = maximum(gaugefields[1].H_rows) # size of 𝐻
    N = length(gaugefields) # number of steps in the driving sequence
    n_elems = length(gaugefields[1].H_vals) * (M+1)^2 # total number of elements in 𝑄: (M+1)^2 blocks each holding `length(gaugefields[1].H_vals)` values
    Q_rows = Vector{Int}(undef, n_elems)
    Q_cols = Vector{Int}(undef, n_elems)
    Q_vals = Vector{Complex{Float}}(undef, n_elems)
    block_vals = similar(gaugefields[1].H_vals) # for storing summed stroboscopic drives
    
    # block-diagonal of 𝑄
    for i in eachindex(block_vals)
        if gaugefields[1].H_vals[i] == 0 # will be true if this is a diagonal value
            block_vals[i] = Inf # mark this element to later find diagonal elements of 𝑄
        else
            block_vals[i] = sum(gaugefields[j].H_vals[i] for j in 1:N) / N
        end
    end
    counter = 1
    fill_blockband!(Q_rows, Q_cols, Q_vals, gaugefields[1].H_rows, gaugefields[1].H_cols, block_vals, 0, blocksize, M+1, counter)
    counter += (M+1) * length(gaugefields[1].H_vals)
    # all remaining block-bands
    for m in 1:M
        for i in eachindex(block_vals)
            block_vals[i] = sum(gaugefields[j].H_vals[i] * cispi(-2m*j/N) for j in 1:N) * (cispi(2m/N)-1) / (2π*im*m)
        end
        fill_blockband!(Q_rows, Q_cols, Q_vals, gaugefields[1].H_rows, gaugefields[1].H_cols, block_vals, m, blocksize, M+1, counter)
        counter += 2(M+1-m) * length(gaugefields[1].H_vals)
    end
    return Q_rows, Q_cols, Q_vals
end

"Fill the `m`th block-off-diagonal of `Q` with matrices `q`. `counter` shows where to start pushing."
function fill_blockband!(Q_rows, Q_cols, Q_vals, q_rows, q_cols, q_vals, m, blocksize, nblockrows, counter)
    if m == 0 # for the block-diagonal can't use `push_vals`
        for i in eachindex(q_vals) # take a value and put it into all required blocks
            for r_b in 0:nblockrows-1 # for each diagonal block whose block-coordinates are (r_b, r_b)
                Q_rows[counter] = r_b*blocksize + q_rows[i]
                Q_cols[counter] = r_b*blocksize + q_cols[i]
                Q_vals[counter] = q_vals[i]
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

"""
Calculate `nsaves` quasienergies closest to `E_target` for quasimomenta `qx` and `qy`, at driving frequency `ω`.
"""
function spectrum(fgf::FloquetGaugeField{Float}, ω::Real, E_target::Real, qxs::AbstractVector{<:Real}, qys::AbstractVector{<:Real}; nsaves::Integer)  where {Float<:AbstractFloat}
    E = Array{Float,3}(undef, nsaves, length(qxs), length(qys))
    
    Q = sparse(fgf.Q_rows, fgf.Q_cols, fgf.Q_vals)
    Q_vals = nonzeros(Q)
    diagidx = findall(==(Inf), Q_vals) # find indices of diagonal elements -- we saved Inf's there (see `constructH`)
    diagonal = Vector{Float}(undef, fgf.blocksize) # 𝑞-dependent diagonal of each diagonal block
    
    M = Int(√fgf.blocksize) # size of each block in `H`
    j_max = (M - 1) ÷ 2  # index for each block in `H` runs in `-j_max:j_max`, giving `M` values in total
    m_max = fgf.M ÷ 2
    for (iqy, qy) in enumerate(qys), (iqx, qx) in enumerate(qxs)
        for (j, jx) in enumerate(-j_max:j_max), (i, jy) in enumerate(-j_max:j_max)
            diagonal[(j-1)M+i] = fgf.u₀₀ + qx^2 + qy^2 + 4(qx*jx + qy*jy) + 4(jx^2 + jy^2)
        end
        for (r_b, m) in enumerate(-m_max:m_max)
            Q_vals[diagidx[(r_b-1)fgf.blocksize+1:r_b*fgf.blocksize]] .= diagonal .+ m * ω
        end
        # LinearAlgebra
        vals = eigvals(Hermitian(Matrix(Q)))
        sort!(vals, by=x -> abs(x - E_target))

        # KrylovKit
        # vals, _, _ = eigsolve(Q, nsaves, EigSorter(x -> abs(x - E_target); rev=false), tol=(Float == Float32 ? 1e-6 : 1e-12))
        
        # Arpack
        # Q_vals[2] += 1e-6
        # vals, _ = eigs(Q, nev=nsaves, sigma=E_target, which=:LM, tol=1e-4)

        E[:, iqx, iqy] .= real.(vals[1:nsaves])
        # writedlm("$iqx.txt", E[:, iqx, iqy])
        println(iqx)
    end
    return E
end

end