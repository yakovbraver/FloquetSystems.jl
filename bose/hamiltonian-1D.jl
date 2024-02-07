using DifferentialEquations, SparseArrays, Combinatorics, SpecialFunctions
using LinearAlgebra: diagind, diag, eigvals, I, BLAS, mul!
using ProgressMeter: @showprogress
import ProgressMeter
using FLoops: @floop, @init

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

"Print a textual representation of a `Lattice`."
function Base.show(io::IO, lattice::Lattice)
    println("$(lattice.nbozons) bozons on a $(lattice.isperiodic ? "" : "non")periodic lattice:")
    # print cell numbers
    for i in 1:prod(lattice.dims)
        print(" $i")
        i % lattice.dims[2] == 0 && println()
    end
    println("Total number of states: $(length(lattice.basis_states))")
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
    type::Symbol # `:dpt`, `:dpt_quick`, or anything else for non-dpt
    order::Int
    space_of_state::Vector{Tuple{Int,Int}}    # space_of_state[i] stores the subspace number (𝐴, 𝑎) of i'th state, with 𝐴 = 0 assigned to all nondegenerate space
    E₀::Vector{Int} # zeroth-order spectrum, in units of 𝑈
    H::Matrix{Float64} # the Hamiltonian matrix
end

"Construct a `BoseHamiltonian` object defined on `lattice`."
function BoseHamiltonian(lattice::Lattice, J::Real, U::Real, f::Real, ω::Real, r::Rational=0//1; order::Integer=1, type::Symbol=:basic)
    nstates = length(lattice.basis_states)
    E₀ = zeros(Int, length(lattice.basis_states))
    for (index, state) in enumerate(lattice.basis_states)
        for n_i in state
            if (n_i > 1)
                E₀[index] += n_i * (n_i - 1) ÷ 2 # will always be divisible by 2
            end
        end
    end
    space_of_state::Vector{Tuple{Int,Int}} = if r == 0
        Vector{Tuple{Int,Int}}()
    else
        map(E₀) do E
            a = floor(Int, E * r)
            A = E % denominator(r)
            return (A, a)
        end
    end
    bh = BoseHamiltonian(lattice, float(J), float(U), float(f), float(ω), type, order, space_of_state, E₀, zeros(nstates, nstates))
    if type == :dpt
        constructH_dpt!(bh, order)
    elseif type == :dpt_quick
        constructH_dpt_quick!(bh, order)
    else
        constructH!(bh, order)
    end
    return bh
end

"Update parameters of `bh` and reconstruct `bh.H`."
function update_params!(bh::BoseHamiltonian; J::Real=bh.J, U::Real=bh.U, f::Real=bh.f, ω::Real=bh.ω, order::Integer=bh.order, type::Symbol=bh.type)
    bh.J = J; bh.U = U; bh.f = f; bh.ω = ω; bh.order = order; bh.type = type
    if type == :dpt
        constructH_dpt!(bh, order)
    elseif type == :dpt_quick
        constructH_dpt_quick!(bh, order)
    elseif type == :diverging
        constructH_diverging!(bh, order)
    else
        constructH!(bh, order)
    end
end

"Construct the Hamiltonian matrix."
function constructH!(bh::BoseHamiltonian, order::Integer)
    (;J, U, f, ω, H) = bh
    (;index_of_state, ncells, neis_of_cell) = bh.lattice
    H .= 0
    H[diagind(H)] .= bh.E₀ .* U

    Jeff = J * besselj0(f)

    J_sum = [0.0, 0.0]
    if order == 2
        a_max = 20
        J_sum[1] = (J/ω)^2 * U * sum(         besselj(a, f)^2 / a^2 for a in 1:a_max) # for 𝑗 = k
        J_sum[2] = (J/ω)^2 * U * sum((-1)^a * besselj(a, f)^2 / a^2 for a in 1:a_max) # for 𝑗 ≠ k
    end

    # take each basis state and find which transitions are possible
    for (ket, α) in index_of_state
        for i = 1:ncells # iterate over the terms of the Hamiltonian
            # 𝑎†ᵢ 𝑎ⱼ
            for (j, _) in neis_of_cell[i]
                if (ket[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    val = -Jeff * sqrt( (ket[i]+1) * ket[j] )
                    bra = copy(ket)
                    bra[j] -= 1
                    bra[i] += 1
                    α′ = index_of_state[bra]
                    H[α′, α] += val
                end
            end

            if order == 2
                for (j, _) in neis_of_cell[i], (k, _) in neis_of_cell[i]
                    C₁, C₂ = j == k ? J_sum : (J_sum[2], J_sum[1])
                    # 𝑎†ₖ (2𝑛ᵢ - 𝑛ⱼ) 𝑎ⱼ
                    if (ket[j] > 0 && ket[i] != ket[j]-1)
                        val = C₁ * √( (k == j ? ket[k] : ket[k]+1) * ket[j] ) * (2ket[i] - (ket[j]-1))
                        bra = copy(ket)
                        bra[j] -= 1
                        bra[k] += 1
                        α′ = index_of_state[bra]
                        H[α′, α] += val
                    end
                    # 𝑎†ⱼ (2𝑛ᵢ - 𝑛ⱼ) 𝑎ₖ
                    if (ket[k] > 0 && ket[i] != (j == k ? ket[j]-1 : ket[j]))
                        val = C₁ * √( (j == k ? ket[j] : ket[j]+1) * ket[k] ) * (2ket[i] - (j == k ? ket[j]-1 : ket[j]))
                        bra = copy(ket)
                        bra[k] -= 1
                        bra[j] += 1
                        α′ = index_of_state[bra]
                        H[α′, α] += val
                    end
                    # 𝑎†ᵢ 𝑎†ᵢ 𝑎ₖ 𝑎ⱼ
                    if ( (k == j && ket[j] > 1) || (k != j && ket[k] > 0 && ket[j] > 0))
                        val = -C₂ * √( (ket[i]+2) * (ket[i]+1) * (k == j ? (ket[j]-1)ket[j] : ket[j]ket[k]))
                        bra = copy(ket)
                        bra[j] -= 1
                        bra[k] -= 1
                        bra[i] += 2
                        α′ = index_of_state[bra]
                        H[α′, α] += val
                    end
                    # 𝑎†ₖ 𝑎†ⱼ 𝑎ᵢ 𝑎ᵢ
                    if (ket[i] > 1)
                        val = -C₂ * √( (k == j ? (ket[j]+2) * (ket[j]+1) : (ket[k]+1) * (ket[j]+1)) * (ket[i]-1)ket[i])
                        bra = copy(ket)
                        bra[i] -= 2
                        bra[j] += 1
                        bra[k] += 1
                        α′ = index_of_state[bra]
                        H[α′, α] += val
                    end
                end
            end
        end
    end
end

"""
Construct the Hamiltonian matrix for the degenerate case but without DPT.
We do not assume that 𝑈 ≪ 𝜔, but we do not use DPT either, leading to diverging results.
"""
function constructH_diverging!(bh::BoseHamiltonian, order::Integer)
    (;J, U, f, ω, H) = bh
    (;index_of_state, ncells, nbozons, neis_of_cell) = bh.lattice
    H .= 0
    H[diagind(H)] .= bh.E₀ .* U

    Jeff = J * besselj0(f)

    n_max = nbozons - 1
    n_min = -nbozons - 1
    R1 = Dict{Int64, Real}()
    R2 = Dict{Int64, Real}()
    for n in n_min:n_max
        R1[n] = 𝑅(ω, U*n, f, type=1)
        R2[n] = 𝑅(ω, U*n, f, type=2)
    end

    # take each basis state and find which transitions are possible
    for (ket, α) in index_of_state
        for i = 1:ncells # iterate over the terms of the Hamiltonian
            # 𝑎†ᵢ 𝑎ⱼ
            for (j, _) in neis_of_cell[i]
                if (ket[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    val = -Jeff * sqrt( (ket[i]+1) * ket[j] )
                    bra = copy(ket)
                    bra[j] -= 1
                    bra[i] += 1
                    α′ = index_of_state[bra]
                    H[α′, α] += val
                end
            end

            if order == 2
                for (j, i_j) in neis_of_cell[i], k in 1:ncells, (l, k_l) in neis_of_cell[k]
                    # 𝑎†ᵢ 𝑎ⱼ [𝑏𝜔+𝑈(𝑛ₖ-𝑛ₗ-1)]⁻¹ 𝑎†ₖ 𝑎ₗ
                    if ( ket[l] > 0 && (j == k || (j == l && ket[j] > 1) || (j != l && ket[j] > 0)) )
                        R = i_j == k_l ? R1 : R2
                        val = -J^2/2
                        bra = copy(ket)
                        val *= √bra[l]
                        bra[l] -= 1
                        bra[k] += 1
                        val *= R[bra[k] - bra[l] - 1] * √bra[k]
                        val *= √bra[j]
                        bra[j] -= 1
                        bra[i] += 1
                        val *= √bra[i]
                        α′ = index_of_state[bra]
                        H[α′, α] += val
                    end

                    # [𝑏𝜔+𝑈(𝑛ₖ-𝑛ₗ-1)]⁻¹ 𝑎†ₖ 𝑎ₗ 𝑎†ᵢ 𝑎ⱼ 
                    if ( ket[j] > 0 && (l == i || (l == j && ket[l] > 1) || (l != j && ket[l] > 0)) )
                        R = i_j == k_l ? R1 : R2
                        val = +J^2/2
                        bra = copy(ket)
                        val *= √bra[j]
                        bra[j] -= 1
                        bra[i] += 1
                        val *= √bra[i]
                        val *= √bra[l]
                        bra[l] -= 1
                        bra[k] += 1
                        val *= R[bra[k] - bra[l] - 1] * √bra[k]
                        α′ = index_of_state[bra]
                        H[α′, α] += val
                    end
                end
            end
        end
    end
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
function constructH_dpt!(bh::BoseHamiltonian, order::Integer)
    (;index_of_state, ncells, neis_of_cell) = bh.lattice
    (;J, U, f, ω, E₀, space_of_state, H) = bh
    
    ε = Vector{Float64}(undef, length(E₀)) # energies (including 𝑈 multiplier) reduced to first Floquet zone
    for i in eachindex(E₀)
        ε[i] = E₀[i]*U - space_of_state[i][2]*ω
    end
    H .= 0
    H[diagind(H)] .= ε

    R = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Bool}, Float64}()
    R2 = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Bool}, Float64}()
    R3 = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int}, Float64}()
    # take each basis state and find which transitions are possible
    for (ket, α) in index_of_state
        A, a = space_of_state[α]
        for i = 1:ncells # iterate over the terms of the Hamiltonian
            # 1st order
            for (j, i_j) in neis_of_cell[i]
                if (ket[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    bra = copy(ket)
                    bra[j] -= 1
                    bra[i] += 1
                    α′ = index_of_state[bra]
                    A′, a′ = space_of_state[α′]
                    val = -J * besselj(a - a′, f*i_j) * sqrt( (ket[i]+1) * ket[j] )
                    H[α′, α] += val
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
                            α′ = index_of_state[bra]
                            A′, a′ = space_of_state[α′]
                            val *= √bra[i]
                            val *= (get_R!(R, U, ω, f, bra[i]-bra[j]-1, a′-b, i_j, k_l, a′, a, b, true) +
                                    get_R!(R, U, ω, f, ket[l]-ket[k]-1, a-b, i_j, k_l, a′, a, b, true))
                            H[α′, α] += val
                        end
                    end
                end
            end

            if order >= 3
                for (j, i_j) in neis_of_cell[i]
                    for k = 1:ncells, (l, k_l) in neis_of_cell[k], m = 1:ncells, (n, m_n) in neis_of_cell[m]
                        # 𝑎†ᵢ 𝑎ⱼ 𝑎†ₖ 𝑎ₗ 𝑎†ₘ 𝑎ₙ
                        ket[n] == 0 && continue
                        bra = copy(ket)
                        val = -J^3/2
                        val *= √bra[n]; bra[n] -= 1; bra[m] += 1; val *= √bra[m]
                        bra[l] == 0 && continue
                        γ = index_of_state[bra]
                        C, c = space_of_state[γ]
                        val *= √bra[l]; bra[l] -= 1; bra[k] += 1; val *= √bra[k]
                        bra[j] == 0 && continue
                        β = index_of_state[bra]
                        B, b = space_of_state[β]
                        val *= √bra[j]; bra[j] -= 1; bra[i] += 1; val *= √bra[i]
                        α′ = index_of_state[bra]
                        A′, a′ = space_of_state[α′]

                        s = 0.0 # terms of the sum
                        J_indices = (-a′+b, -b+c, -c+a)
                        J_args = (i_j, k_l, m_n)
                        ΔE1 = E₀[γ] - E₀[α′]
                        ΔE2 = E₀[β] - E₀[γ]
                        s += get_R2!(R2, U, ω, f, ΔE1, ΔE2, c-a′, b-c, J_indices, J_args, true)
                    
                        J_indices = (a-c, b-a′, c-b)
                        J_args = (m_n, i_j, k_l)
                        ΔE1 = E₀[β] - E₀[α]
                        ΔE2 = E₀[γ] - E₀[β]
                        s += get_R2!(R2, U, ω, f, ΔE1, ΔE2, b-a, c-b, J_indices, J_args, true)
                    
                        J_indices = (c-b, b-a′, a-c)
                        J_args = (k_l, i_j, m_n)
                        ΔE1 = E₀[β] - E₀[α′]
                        ΔE2 = E₀[α′] - E₀[γ]
                        s -= get_R2!(R2, U, ω, f, ΔE1, ΔE2, b-a′, a′-c, J_indices, J_args, true)

                        ΔE1 = E₀[β] - E₀[α]
                        ΔE2 = E₀[α] - E₀[γ]
                        s -= get_R2!(R2, U, ω, f, ΔE1, ΔE2, b-a, a-c, J_indices, J_args, true)

                        key = (a, a′, b, c, A, A′, B, C, i_j, k_l, m_n)
                        if !haskey(R3, key)
                            N = 20
                            t = 0.0
                            for p in [-N:-1; 1:N], q in [-N:-1; 1:N]
                                q == p && continue
                                t += besselj(b-a′-p, f*i_j) * besselj(c-b+p-q, f*k_l) * besselj(a-c+q, f*m_n) * (
                                        1 / 2(ε[α′] - ε[γ] - q*ω)     * (1/(ε[γ] - ε[β]  - (p-q)*ω) - 1/(ε[β] - ε[α′] + p*ω)) +
                                        1 / 2(ε[α]  - ε[β] - p*ω)     * (1/(ε[α] - ε[γ]  - q*ω)     - 1/(ε[γ] - ε[β]  - (p-q)*ω)) +
                                        1 / 6(ε[γ]  - ε[β] - (p-q)*ω) * (1/(ε[β] - ε[α′] + p*ω)     + 1/(ε[α] - ε[γ]  - q*ω)) -
                                        1 / 3(ε[α]  - ε[γ] - q*ω) / (ε[β] - ε[α′] + p*ω) )
                            end
                            R3[key] = t
                        end
                        s += R3[key]
                        val *= s
                        H[α′, α] += val
                    end
                end
            end
        end
    end
end

"Construct the Hamiltonian matrix."
function constructH_dpt_quick!(bh::BoseHamiltonian, order::Integer)
    (;index_of_state, ncells, neis_of_cell) = bh.lattice
    (;J, U, f, ω, E₀, space_of_state, H) = bh
    ε = Vector{Float64}(undef, length(E₀)) # energies (including 𝑈 multiplier) reduced to first Floquet zone
    for i in eachindex(E₀)
        ε[i] = E₀[i]*U - space_of_state[i][2]*ω
    end
    H .= 0
    H[diagind(H)] .= ε

    R = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Bool}, Float64}()
    R2 = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Bool}, Float64}()
    R3 = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Int,Int,Int}, Float64}()
    
    # take each basis state and find which transitions are possible
    for (ket, α) in index_of_state
        A, a = space_of_state[α]
        for i = 1:ncells # iterate over the terms of the Hamiltonian
            # 1st order
            for (j, i_j) in neis_of_cell[i]
                if (ket[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    bra = copy(ket)
                    bra[j] -= 1
                    bra[i] += 1
                    α′ = index_of_state[bra]
                    A′, a′ = space_of_state[α′]
                    if A′ == A # proceed only if bra is in the same degenerate space
                        val = -J * besselj(a - a′, f*i_j) * sqrt( (ket[i]+1) * ket[j] )
                        H[α′, α] += val
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
                            α′ = index_of_state[bra]
                            A′, a′ = space_of_state[α′]
                            if A′ == A # proceed only if bra is in the same degenerate space
                                val *= √bra[i]
                                skipzero = (B == A)
                                val *= (get_R!(R, U, ω, f, bra[i]-bra[j]-1, a′-b, i_j, k_l, a′, a, b, skipzero) +
                                        get_R!(R, U, ω, f, ket[l]-ket[k]-1, a-b, i_j, k_l, a′, a, b, skipzero))
                                H[α′, α] += val
                            end
                        end
                    end
                end
            end

            if order >= 3
                for (j, i_j) in neis_of_cell[i]
                    for k = 1:ncells, (l, k_l) in neis_of_cell[k], m = 1:ncells, (n, m_n) in neis_of_cell[m]
                        # 𝑎†ᵢ 𝑎ⱼ 𝑎†ₖ 𝑎ₗ 𝑎†ₘ 𝑎ₙ
                        ket[n] == 0 && continue
                        bra = copy(ket)
                        val = -J^3/2
                        val *= √bra[n]; bra[n] -= 1; bra[m] += 1; val *= √bra[m]
                        bra[l] == 0 && continue
                        γ = index_of_state[bra]
                        C, c = space_of_state[γ]
                        val *= √bra[l]; bra[l] -= 1; bra[k] += 1; val *= √bra[k]
                        bra[j] == 0 && continue
                        β = index_of_state[bra]
                        B, b = space_of_state[β]
                        val *= √bra[j]; bra[j] -= 1; bra[i] += 1; val *= √bra[i]
                        α′ = index_of_state[bra]
                        A′, a′ = space_of_state[α′]
                        if A′ == A
                            s = 0.0 # terms of the sum
                            if A == B
                                J_indices = (-a′+b, -b+c, -c+a)
                                J_args = (i_j, k_l, m_n)
                                ΔE1 = E₀[γ] - E₀[α′]
                                ΔE2 = E₀[β] - E₀[γ]
                                s += get_R2!(R2, U, ω, f, ΔE1, ΔE2, c-a′, b-c, J_indices, J_args, B == C)
                            end
                            if A == C
                                J_indices = (a-c, b-a′, c-b)
                                J_args = (m_n, i_j, k_l)
                                ΔE1 = E₀[β] - E₀[α]
                                ΔE2 = E₀[γ] - E₀[β]
                                s += get_R2!(R2, U, ω, f, ΔE1, ΔE2, b-a, c-b, J_indices, J_args, B == C)
                            end
                            if B == C
                                J_indices = (c-b, b-a′, a-c)
                                J_args = (k_l, i_j, m_n)
                                ΔE1 = E₀[β] - E₀[α′]
                                ΔE2 = E₀[α′] - E₀[γ]
                                s -= get_R2!(R2, U, ω, f, ΔE1, ΔE2, b-a′, a′-c, J_indices, J_args, B == C)

                                ΔE1 = E₀[β] - E₀[α]
                                ΔE2 = E₀[α] - E₀[γ]
                                s -= get_R2!(R2, U, ω, f, ΔE1, ΔE2, b-a, a-c, J_indices, J_args, B == C)
                            end

                            # skip = (A == B && A == C) ? 1 : (A == B && B != C) ? 2 : (A == C && B != C) ? 3 : (A != B && B == C) ? 4 : 5
                            # key = (a, a′, b, c, A, A′, B, C, i_j, k_l, m_n, skip)
                            key = (a, a′, b, c, A, B, C, i_j, k_l, m_n)
                            if !haskey(R3, key)
                                N = 20
                                t = 0.0
                                prange = A == B ? [-N:-1; 1:N] : collect(-N:N)
                                qrange = A == C ? [-N:-1; 1:N] : collect(-N:N)
                                for p in prange, q in qrange
                                    B == C && q == p && continue
                                    t += besselj(b-a′-p, f*i_j) * besselj(c-b+p-q, f*k_l) * besselj(a-c+q, f*m_n) * (
                                         1 / 2(ε[α′] - ε[γ] - q*ω)     * (1/(ε[γ] - ε[β]  - (p-q)*ω) - 1/(ε[β] - ε[α′] + p*ω)) +
                                         1 / 2(ε[α]  - ε[β] - p*ω)     * (1/(ε[α] - ε[γ]  - q*ω)     - 1/(ε[γ] - ε[β]  - (p-q)*ω)) +
                                         1 / 6(ε[γ]  - ε[β] - (p-q)*ω) * (1/(ε[β] - ε[α′] + p*ω)     + 1/(ε[α] - ε[γ]  - q*ω)) -
                                         1 / 3(ε[α]  - ε[γ] - q*ω) / (ε[β] - ε[α′] + p*ω) )
                                end
                                R3[key] = t
                            end
                            s += R3[key]
                            val *= s
                            H[α′, α] += val
                        end
                    end
                end
            end
        end
    end
end

function push_state!(H_rows, H_cols, H_vals, val; row, col)
    push!(H_vals, val)
    push!(H_cols, col)
    push!(H_rows, row)
end

"Return key from the `R` dictionary; required for 2nd order DPT."
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

"Return key from the `R` dictionary; required for 3rd order DPT."
function get_R2!(R, U, ω, f, ΔE1, ΔE2, d1, d2, J_indices, J_args, skipzero)
    i1, i2, i3 = J_indices
    x1, x2, x3 = J_args
    key = (ΔE1, ΔE2, d1, d2, i1, i2, i3, x1, x2, x3, skipzero)
    if !haskey(R, key)
        N = 20
        s = 0.0
        prange = skipzero ? [-N:-1; 1:N] : collect(-N:N)
        for p in prange
            s += 1 / (U*ΔE1 - (d1-p)*ω) / (U*ΔE2 - (d2+p)*ω) * besselj(i1, f*x1) * besselj(i2-p, f*x2) * besselj(i3+p, f*x3)
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
    for C in CartesianIndices(bh.H)
        bh.H[C[1], C[2]] != 0 && println(io, bh.lattice.basis_states[C[1]], " Ĥ ", bh.lattice.basis_states[C[2]], " = ", round(bh.H[C[1], C[2]], sigdigits=3))
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

    nstates = size(bh.H, 1)
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

"Calculate quasienergy spectrum via monodromy matrix for each value of 𝑈 in `Us`. `bh.U` may be arbitrary."
function quasienergy_dense(bh::BoseHamiltonian, Us::AbstractVector{<:Real}; parallelise=true)
    nstates = size(bh.H, 1)
    H = zeros(ComplexF64, nstates, nstates)
    H_sign = zeros(Int, nstates, nstates)
    (;J, f, ω, E₀) = bh
    (;index_of_state, ncells, neis_of_cell) = bh.lattice

    # Construct the Hamiltonian with `f` = 0
    # off-diagonal elements 𝑎†ᵢ 𝑎ⱼ
    for (ket, α) in index_of_state
        for i = 1:ncells # iterate over the terms of the Hamiltonian
            for (j, s) in neis_of_cell[i]
                # 𝑎†ᵢ 𝑎ⱼ
                if (ket[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    val = -im * -J * sqrt( (ket[i]+1) * ket[j] ) # multiply by `-im` as on the rhs of ∂ₜ𝜓 = -i𝐻𝜓
                    bra = copy(ket)
                    bra[j] -= 1
                    bra[i] += 1
                    α′ = index_of_state[bra]
                    H[α′, α] = val
                    H_sign[α′, α] = s
                end
            end
        end
    end

    n_U = length(Us)
    ε = Matrix{Float64}(undef, nstates, n_U)
    C₀ = Matrix{ComplexF64}(I, nstates, nstates)
    
    T = 2π / ω
    tspan = (0.0, T)

    if parallelise
        n_blas = BLAS.get_num_threads() # save original number of threads to restore later
        BLAS.set_num_threads(1)

        # progbar = ProgressMeter.Progress(length(Us)) # makes cores stop periodically when using floop

        @floop for (i, U) in enumerate(Us)
            @init begin
                H_base = copy(H) # diagonal of `H_base` will be mutated depending on `U`
                # diagonal of `H_buff` will remain equal to the diagonal of `H_base` throughout diffeq solving,
                # while off-diagnoal elemnts will be mutated at each step
                H_buff = zeros(ComplexF64, nstates, nstates)
            end
            H_base[diagind(H_base)] .= U .* (-im .* E₀)
            H_buff[diagind(H_base)] .= diag(H_base)
            params = (H_buff, H_base, H_sign, ω, f)
            prob = ODEProblem(schrodinger!, C₀, tspan, params, save_everystep=false)
            sol = solve(prob)
            ε[:, i] = -ω .* angle.(eigvals(sol[end])) ./ 2π

            # ProgressMeter.next!(progbar)
        end
        # ProgressMeter.finish!(progbar)
        BLAS.set_num_threads(n_blas) # restore original number of threads
    else
        @showprogress for (i, U) in enumerate(Us)
            H[diagind(H)] .= U .* (-im .* E₀)
            H_buff = zeros(ComplexF64, nstates, nstates)
            H_buff[diagind(H)] .= diag(H)
            params = (H_buff, H, H_sign, ω, f)
            prob = ODEProblem(schrodinger!, C₀, tspan, params, save_everystep=false)
            sol = solve(prob)
            ε[:, i] = -ω .* angle.(eigvals(sol[end])) ./ 2π
        end
    end

    return ε
end

"Update rhs operator (used for monodromy matrix calculation)."
function schrodinger!(du, u, p, t)
    H_buff, H_base, H_sign, ω, f = p
    p = cis(f * sin(ω*t)); n = cis(-f * sin(ω*t));
    for (i, s) in enumerate(H_sign)
        if s > 0 
            H_buff[i] = H_base[i] * p
        elseif s < 0
            H_buff[i] = H_base[i] * n
        end
    end
    mul!(du, H_buff, u)
end