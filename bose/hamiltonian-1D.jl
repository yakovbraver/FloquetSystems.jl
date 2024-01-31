using DifferentialEquations, SparseArrays, Combinatorics, SpecialFunctions
using LinearAlgebra: diagind, eigvals, I
using ProgressMeter: @showprogress

import Base.show

"A type representing a quantum lattice. Geometry is square, and can be both 1D and 2D."
struct Lattice
    ncells::Int
    nbozons::Int
    dims::Tuple{Int,Int}
    isperiodic::Bool    # periodicity is allowed only along dimensions whose length is larger than 2
    basis_states::Vector{Vector{Int}}     # all basis states as a vector (index => state)
    index_of_state::Dict{Vector{Int},Int} # a dictionary (state => index)
    neis_of_cell::Vector{Vector{Tuple{Int,Int}}} # neis_of_cell[i][j] = (<cell number of j'th neighbour of i'th cell>, <-1 if neigbour is below or on the right; +1 otherwise>)
end

"Construct a `Lattice` object."
function Lattice(;dims::Tuple{Int,Int}, nbozons::Integer, isperiodic::Bool)
    ncells = prod(dims)
    nstates = binomial(nbozons+ncells-1, nbozons)
    lattice = Lattice(ncells, nbozons, dims, isperiodic, Vector{Vector{Int}}(undef, nstates), Dict{Vector{Int},Int}(), Vector{Vector{Tuple{Int,Int}}}(undef, nstates))
    makebasis!(lattice)
    makeneis!(lattice)
    return lattice
end

"Fill `lattice.neis_of_cell`."
function makeneis!(lattice::Lattice)
    nrows, ncols = lattice.dims
    for cell in 0:lattice.ncells-1 # 0-based index
        row = cell ÷ ncols # 0-based index
        col = cell % ncols # 0-based index
        neis = Tuple{Int,Int,Int}[]
        if row > 0 || (lattice.isperiodic && nrows > 2) # neigbour above
            push!(neis, (rem(row-1, nrows, RoundDown), col, +1))
        end
        if col < ncols - 1 || (lattice.isperiodic && ncols > 2) # neigbour to the right
            push!(neis, (row, (col+1)%ncols, -1))
        end
        if row < nrows - 1 || (lattice.isperiodic && nrows > 2) # neigbour below
            push!(neis, ((row+1)%nrows, col, -1))
        end
        if col > 0 || (lattice.isperiodic && ncols > 2) # neigbour to the left
            push!(neis, (row, rem(col-1, ncols, RoundDown), +1))
        end
        lattice.neis_of_cell[cell+1] = map(x -> (x[2] + x[1]*ncols + 1, x[3]), neis) # convert to cell numbers; `+1` converts to 1-based index
    end
end

"""
A type representing a Bose-Hubbard Hamiltonian,
    H = - ∑ 𝐽ᵢⱼ 𝑎†ᵢ 𝑎ⱼ, + 𝑈/2 ∑ 𝑛ᵢ(𝑛ᵢ - 1).
"""
mutable struct BoseHamiltonian
    lattice::Lattice
    J::Float64
    U::Float64
    f::Float64 # F / ω
    ω::Real
    type::Symbol
    order::Int
    space_of_state::Vector{Tuple{Int,Int}}    # space_of_state[i] stores the subspace number (𝐴, 𝑎) of i'th state, with 𝐴 = 0 assigned to all nondegenerate space
    H::SparseMatrixCSC{Float64, Int} # the Hamiltonian matrix
end

"Construct a `BoseHamiltonian` object defined on `lattice`."
function BoseHamiltonian(lattice::Lattice, J::Real, U::Real, f::Real, ω::Real, space_of_state::Vector{Tuple{Int,Int}}=Vector{Tuple{Int,Int}}(); order::Integer=1, type::Symbol=:smallU)
    bh = BoseHamiltonian(lattice, float(J), float(U), float(f), float(ω), type, order, space_of_state, spzeros(Float64, 1, 1))
    if type == :smallU
        constructH_smallU!(bh, order)
    elseif type == :largeU
        constructH_largeU!(bh, order)
    end
    bh
end

"Construct the Hamiltonian matrix."
function constructH_smallU!(bh::BoseHamiltonian, order::Integer)
    (;J, U, f, ω) = bh
    (;index_of_state, ncells, neis_of_cell) = bh.lattice
    H_rows, H_cols, H_vals = Int[], Int[], Float64[]
    Jeff = J * besselj0(f)

    J_sum = [0.0, 0.0]
    if order == 2
        a_max = 20
        J_sum[1] = (J/ω)^2 * U * sum(         besselj(a, f)^2 / a^2 for a in 1:a_max) # for 𝑗 = k
        J_sum[2] = (J/ω)^2 * U * sum((-1)^a * besselj(a, f)^2 / a^2 for a in 1:a_max) # for 𝑗 ≠ k
    end

    # take each basis state and find which transitions are possible
    for (state, index) in index_of_state
        val_d = 0.0 # diagonal value
        for i = 1:ncells # iterate over the terms of the Hamiltonian
            # 𝑛ᵢ(𝑛ᵢ - 1)
            if (state[i] > 1)
                val_d += U/2 * state[i] * (state[i] - 1)
            end
            # 𝑎†ᵢ 𝑎ⱼ
            for (j, _) in neis_of_cell[i]
                if (state[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    val = -Jeff * sqrt( (state[i]+1) * state[j] )
                    bra = copy(state)
                    bra[j] -= 1
                    bra[i] += 1
                    row = index_of_state[bra]
                    push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                end
            end

            if order == 2
                for (j, _) in neis_of_cell[i], (k, _) in neis_of_cell[i]
                    C₁, C₂ = j == k ? J_sum : (J_sum[2], J_sum[1])
                    # 𝑎†ₖ (2𝑛ᵢ - 𝑛ⱼ) 𝑎ⱼ
                    if (state[j] > 0 && state[i] != state[j]-1)
                        val = C₁ * √( (k == j ? state[k] : state[k]+1) * state[j] ) * (2state[i] - (state[j]-1))
                        bra = copy(state)
                        bra[j] -= 1
                        bra[k] += 1
                        row = index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end
                    # 𝑎†ⱼ (2𝑛ᵢ - 𝑛ⱼ) 𝑎ₖ
                    if (state[k] > 0 && state[i] != (j == k ? state[j]-1 : state[j]))
                        val = C₁ * √( (j == k ? state[j] : state[j]+1) * state[k] ) * (2state[i] - (j == k ? state[j]-1 : state[j]))
                        bra = copy(state)
                        bra[k] -= 1
                        bra[j] += 1
                        row = index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end
                    # 𝑎†ᵢ 𝑎†ᵢ 𝑎ₖ 𝑎ⱼ
                    if ( (k == j && state[j] > 1) || (k != j && state[k] > 0 && state[j] > 0))
                        val = -C₂ * √( (state[i]+2) * (state[i]+1) * (k == j ? (state[j]-1)state[j] : state[j]state[k]))
                        bra = copy(state)
                        bra[j] -= 1
                        bra[k] -= 1
                        bra[i] += 2
                        row = index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end
                    # 𝑎†ₖ 𝑎†ⱼ 𝑎ᵢ 𝑎ᵢ
                    if (state[i] > 1)
                        val = -C₂ * √( (k == j ? (state[j]+2) * (state[j]+1) : (state[k]+1) * (state[j]+1)) * (state[i]-1)state[i])
                        bra = copy(state)
                        bra[i] -= 2
                        bra[j] += 1
                        bra[k] += 1
                        row = index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end
                end
            end
        end
        push_state!(H_rows, H_cols, H_vals, val_d; row=index, col=index)
    end
    bh.H = sparse(H_rows, H_cols, H_vals)
end

"Construct the Hamiltonian matrix for the degenerate case but without DPT. Will work only in 1D, not tested."
function constructH_largeU_diverging!(bh::BoseHamiltonian, order::Integer)
    H_rows, H_cols, H_vals = Int[], Int[], Float64[]
    (;J, U, f, ω) = bh
    (;index_of_state, ncells, nbozons, neis_of_cell) = bh.lattice
    Jeff = J * besselj0(f)

    n_max = nbozons - 1
    n_min = -nbozons - 1
    R1 = Dict{Int64, Real}()
    R2 = Dict{Int64, Real}()
    for n in n_min:n_max
        R1[n] = 𝑅(ω, U*n, f, type=1)
        R2[n] = 𝑅(ω, U*n, f, type=2)
    end

    js = Vector{Int}(undef, 12)
    ks = Vector{Int}(undef, 12)
    ls = Vector{Int}(undef, 12)
    # take each basis state and find which transitions are possible
    for (state, index) in index_of_state
        val_d = 0.0 # diagonal value
        for i = 1:ncells # iterate over the terms of the Hamiltonian
            # 𝑛ᵢ(𝑛ᵢ - 1)
            if (state[i] > 1)
                val_d += U/2 * state[i] * (state[i] - 1)
            end
            # 𝑎†ᵢ 𝑎ⱼ
            for j in (i-1, i+1)
                if j == 0
                    !bh.lattice.isperiodic && continue
                    j = ncells
                elseif j == ncells + 1
                    !bh.lattice.isperiodic && continue
                    j = 1
                end
                if (state[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    val = -Jeff * sqrt( (state[i]+1) * state[j] )
                    bra = copy(state)
                    bra[j] -= 1
                    bra[i] += 1
                    row = index_of_state[bra]
                    push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                end
            end

            js[1:6] .= i-1; js[7:12] .= i+1;
            ks .= [i-2, i-1, i-1, i, i, i+1, i-1, i, i, i+1, i+1, i+2]
            ls .= [i-1, i-2, i, i-1, i+1, i, i, i-1, i+1, i, i+2, i+1]
            if order == 2
                for (j, k, l) in zip(js, ks, ls)
                    if j < 1
                        j = ncells + j
                    elseif j > ncells
                        j = j - ncells
                    end
                    if k < 1
                        k = ncells + k
                    elseif k > ncells
                        k = k - ncells
                    end
                    if l < 1
                        l = ncells + l
                    elseif l > ncells
                        l = l - ncells
                    end

                    # 𝑎†ᵢ 𝑎ⱼ [𝑏𝜔+𝑈(𝑛ₖ-𝑛ₗ-1)]⁻¹ 𝑎†ₖ 𝑎ₗ
                    if ( state[l] > 0 && (j == k || (j == l && state[j] > 1) || (j != l && state[j] > 0)) )
                        R = i-j == k-l ? R1 : R2
                        val = -J^2/2
                        bra = copy(state)
                        val *= √bra[l]
                        bra[l] -= 1
                        bra[k] += 1
                        val *= R[bra[k] - bra[l] - 1] * √bra[k]
                        val *= √bra[j]
                        bra[j] -= 1
                        bra[i] += 1
                        val *= √bra[i]
                        row = index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end

                    # [𝑏𝜔+𝑈(𝑛ₖ-𝑛ₗ-1)]⁻¹ 𝑎†ₖ 𝑎ₗ 𝑎†ᵢ 𝑎ⱼ 
                    if ( state[j] > 0 && (l == i || (l == j && state[l] > 1) || (l != j && state[l] > 0)) )
                        R = i-j == k-l ? R1 : R2
                        val = +J^2/2
                        bra = copy(state)
                        val *= √bra[j]
                        bra[j] -= 1
                        bra[i] += 1
                        val *= √bra[i]
                        val *= √bra[l]
                        bra[l] -= 1
                        bra[k] += 1
                        val *= R[bra[k] - bra[l] - 1] * √bra[k]
                        row = index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end
                end
            end
        end
        push_state!(H_rows, H_cols, H_vals, val_d; row=index, col=index)
    end
    bh.H = sparse(H_rows, H_cols, H_vals)
end

function 𝑅(ω::Real, Un::Real, f::Real; type::Integer)
    N = 20
    a₀ = round(Int, -Un / ω)
    # if `Un / ω` is integer, a₀ should be skipped in the sum
    a_range = isinteger(Un / ω) ? [a₀-N:a₀-1; a₀+1:a₀+N] : collect(a₀-N:a₀+N) # collect for type stability
    r = 0.0
    if type == 1
        for a in a_range
            a == 0 && continue
            r += 1/(a*ω + Un) * besselj(a, f)^2 * (-1)^a
        end
    else
        for a in a_range
            a == 0 && continue
            r += 1/(a*ω + Un) * besselj(a, f)^2
        end
    end
    return r
end

"Construct the Hamiltonian matrix."
function constructH_largeU!(bh::BoseHamiltonian, order::Integer)
    H_rows, H_cols, H_vals = Int[], Int[], Float64[]
    (;index_of_state, ncells, neis_of_cell) = bh.lattice
    (;J, U, f, ω, space_of_state) = bh

    R = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Bool}, Float64}()
    # take each basis state and find which transitions are possible
    for (ket, ket_index) in index_of_state
        A, a = space_of_state[ket_index]
        val_d = 0.0 # diagonal value
        for i = 1:ncells # iterate over the terms of the Hamiltonian

            # 0th order
            if (ket[i] > 1)
                val_d += U/2 * ket[i] * (ket[i] - 1)
            end

            # 1st order
            for (j, i_j) in neis_of_cell[i]
                if (ket[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    bra = copy(ket)
                    bra[j] -= 1
                    bra[i] += 1
                    A′, a′ = space_of_state[index_of_state[bra]]
                    if A′ == A # proceed only if bra is in the same degenerate space
                        val = -J * besselj(a - a′, f*i_j) * sqrt( (ket[i]+1) * ket[j] )
                        row = index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=ket_index)
                    end
                end
            end

            if order >= 2
                for (j, i_j) in neis_of_cell[i]
                    for k = 1:ncells, (l, k_l) in neis_of_cell[k]
                        # 𝑎†ᵢ 𝑎ⱼ 𝑎†ₖ 𝑎ₗ
                        if ( ket[l] > 0 && (j == k || (j == l && ket[j] > 1) || (j != l && ket[j] > 0)) )
                            val = +J^2/2
                            bra = copy(ket)
                            val *= √bra[l]
                            bra[l] -= 1
                            bra[k] += 1
                            val *= √bra[k]
                            B, b = space_of_state[index_of_state[bra]]
                            val *= √bra[j]
                            bra[j] -= 1
                            bra[i] += 1
                            bra_index = index_of_state[bra]
                            A′, a′ = space_of_state[bra_index]
                            if A′ == A # proceed only if bra is in the same degenerate space
                                val *= √bra[i]
                                skipzero = (B == A)
                                val *= (get_R!(R, U, ω, f, bra[i]-bra[j]-1, a′-b, i_j, k_l, a′, a, b, skipzero) +
                                        get_R!(R, U, ω, f, ket[l]-ket[k]-1, a-b, i_j, k_l, a′, a, b, skipzero))
                                push_state!(H_rows, H_cols, H_vals, val; row=bra_index, col=ket_index)
                            end
                        end
                    end
                end
            end
        end
        push_state!(H_rows, H_cols, H_vals, val_d - a*ω; row=ket_index, col=ket_index)
    end
    bh.H = sparse(H_rows, H_cols, H_vals)
end

function push_state!(H_rows, H_cols, H_vals, val; row, col)
    push!(H_vals, val)
    push!(H_cols, col)
    push!(H_rows, row)
end

function get_R!(R, U, ω, f, nα, d, i_j, k_l, a′, a, b, skipzero)
    key = (nα, d, i_j, k_l, a′, a, b, skipzero)
    if !haskey(R, key)
        N = 20
        s = 0.0
        nrange = skipzero ? [-N:-1; 1:N] : collect(-N:N)
        for n in nrange
            s += 1/(U*nα - (d+n)*ω) * besselj(-(a′-b+n), f*i_j) * besselj(a-b+n, f*k_l)
        end
        R[key] = s
    end
    return R[key]
end

"""
Generate all possible combinations of placing the bozons in the lattice.
Populate `lattice.basis_states` and `lattice.index_of_state`.
"""
function makebasis!(lattice::Lattice)
    index = 1 # unique index identifying the state
    (;ncells, nbozons) = lattice
    for partition in integer_partitions(nbozons)
        length(partition) > ncells && continue # Example (nbozons = 3, ncells = 2): partition = [[3,0], [2,1], [1,1,1]] -- skip [1,1,1] as impossible
        append!(partition, zeros(ncells-length(partition)))
        for p in multiset_permutations(partition, ncells)
            lattice.index_of_state[p] = index
            lattice.basis_states[index] = p
            index += 1
        end
    end
end

"Print non-zero elements of the Hamiltonian `bh` in the format ⟨bra| Ĥ |ket⟩."
function Base.show(io::IO, bh::BoseHamiltonian)
    H_rows, H_cols, H_vals = findnz(bh.H)
    for (i, j, val) in zip(H_rows, H_cols, H_vals) # iterate over the terms of the Hamiltonian
        println(io, bh.lattice.basis_states[i], " Ĥ ", bh.lattice.basis_states[j], " = ", round(val, sigdigits=3))
    end
end

"Calculate quasienergy spectrum of `bh` via monodromy matrix for each value of 𝑈 in `Us`."
function quasienergy(bh::BoseHamiltonian, Us::AbstractVector{<:Real})
    H_rows, H_cols, H_vals = Int[], Int[], ComplexF64[]
    H_sign = Int[] # stores the sign of the tunneling phase for each off-diagonal element
    (;J, f, ω) = bh
    (;index_of_state, ncells, neis_of_cell) = bh.lattice

    # Construct the Hamiltonian with `f` = 0 and `U` = 1
    # off-diagonal elements 𝑎†ᵢ 𝑎ⱼ
    for (state, index) in index_of_state
        for i = 1:ncells # iterate over the terms of the Hamiltonian
            for (j, s) in neis_of_cell[i]
                # 𝑎†ᵢ 𝑎ⱼ
                if (state[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    val = -im * -J * sqrt( (state[i]+1) * state[j] ) # multiply by `-im` as in the rhs of ∂ₜ𝜓 = -i𝐻𝜓
                    bra = copy(state)
                    bra[j] -= 1
                    bra[i] += 1
                    row = index_of_state[bra]
                    push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    push!(H_sign, s)
                end
            end
        end
    end
    # diagonal elements 𝑛ᵢ(𝑛ᵢ - 1)
    U = 1
    for (state, index) in index_of_state
        val = 0.0
        for i = 1:ncells
            if (state[i] > 1)
                val += -im * U/2 * state[i] * (state[i] - 1) # multiply by `-im` as in the rhs of ∂ₜ𝜓 = -i𝐻𝜓
            end
        end
        push_state!(H_rows, H_cols, H_vals, val; row=index, col=index)
    end

    nstates = size(bh.H, 1) # change to nstates
    n_U = length(Us)
    ε = Matrix{Float64}(undef, nstates, n_U)
    C₀ = Matrix{ComplexF64}(I, nstates, nstates)
    
    T = 2π / ω
    tspan = (0.0, T)
    H_vals_U = copy(H_vals) # `H_vals_U` will be mutated depending on `U`
    @showprogress for (i, U) in enumerate(Us)
        H_vals_U[end-nstates+1:end] .= U .* H_vals[end-nstates+1:end] # update last `nstates` values in `H_vals_U` -- these are diagonal elements of the Hamiltonian
        params = (H_rows, H_cols, H_vals_U, H_sign, f, ω, nstates)
        H_op = DiffEqArrayOperator(sparse(H_rows, H_cols, H_vals_U), update_func=update_func!)
        prob = ODEProblem(H_op, C₀, tspan, params, save_everystep=false)
        sol = solve(prob, MagnusGauss4(), dt=T/100)
        ε[:, i] = -ω .* angle.(eigvals(sol[end])) ./ 2π
    end

    return ε
end

"Update rhs operator (used for monodromy matrix calculation)."
function update_func!(H, u, p, t)
    H_rows, H_cols, H_vals, H_sign, f, ω, nstates = p
    vals = copy(H_vals)
    vals[1:end-nstates] .*= cis.(f .* sin(ω.*t) .* H_sign) # update off diagonal elements of the Hamiltonian
    H .= sparse(H_rows, H_cols, vals)
end