using OrdinaryDiffEq, SparseArrays, DelimitedFiles
using SpecialFunctions: besselj0, besselj
using LinearAlgebra: diagind, diag, eigvals, mul!, Symmetric, I, BLAS
using ProgressMeter: @showprogress
import ProgressMeter
using FLoops: @floop, @init
import FLoops

import Base.show

"""
A type representing a Bose-Hubbard Hamiltonian,
    H = - ∑ 𝐽ᵢⱼ 𝑎†ᵢ 𝑎ⱼ, + 𝑈/2 ∑ 𝑛ᵢ(𝑛ᵢ - 1).
"""
mutable struct BoseHamiltonian{Float <: AbstractFloat}
    lattice::Lattice
    J::Float
    U::Float
    f::Float # F / ω
    ω::Float
    type::Symbol # `:dpt`, `:dpt_quick`, `:diverging` or anything else for ordinary high-frequency expansion
    order::Int
    space_of_state::Vector{Tuple{Int,Int}}    # space_of_state[i] stores the subspace number (𝐴, 𝑎) of i'th state, with 𝐴 = 0 assigned to all nondegenerate space
    E₀::Vector{Int}    # zeroth-order spectrum, in units of 𝑈
    H::Matrix{Float} # the Hamiltonian matrix
end

"""
Construct a `BoseHamiltonian` object defined on `lattice`.
Type of `J` determines the type of Float used for all fields of the resulting object.
`ωₗ` is the lower bound of the first Floquet zone.
"""
function BoseHamiltonian(lattice::Lattice, J::Float, U::Real, f::Real, ω::Real, r::Rational=0//1, ωₗ::Real=0; order::Integer=1, type::Symbol=:basic) where {Float <: AbstractFloat}
    nstates = length(lattice.basis_states)
    E₀ = zeros(Int, length(lattice.basis_states))
    for (index, state) in enumerate(lattice.basis_states)
        for n_i in state
            if (n_i > 1)
                E₀[index] += n_i * (n_i - 1) ÷ 2 # will always be divisible by 2
            end
        end
    end
    space_of_state = if r == 0
        Vector{Tuple{Int,Int}}()
    else
        map(E₀) do E
            a = (E*r*ω - ωₗ) ÷ ω |> Int
            A = E % denominator(r)
            return (A, a)
        end
    end
    bh = BoseHamiltonian(lattice, Float(J), Float(U), Float(f), Float(ω), type, order, space_of_state, E₀, zeros(Float, nstates, nstates))
    if type == :dpt
        constructH_dpt!(bh, order)
    elseif type == :dpt_quick
        constructH_dpt_quick!(bh, order)
    elseif type == :diverging
        constructH_diverging!(bh, order)
    else
        constructH!(bh, order)
    end
    return bh
end

"Print non-zero elements of the Hamiltonian `bh` in the format ⟨bra| Ĥ |ket⟩."
function Base.show(io::IO, bh::BoseHamiltonian{<:AbstractFloat})
    for C in CartesianIndices(bh.H)
        bh.H[C[1], C[2]] != 0 && println(io, bh.lattice.basis_states[C[1]], " Ĥ ", bh.lattice.basis_states[C[2]], " = ", round(bh.H[C[1], C[2]], sigdigits=3))
    end
end

"Update parameters of `bh` and reconstruct `bh.H`."
function update_params!(bh::BoseHamiltonian{<:AbstractFloat}; J::Real=bh.J, U::Real=bh.U, f::Real=bh.f, ω::Real=bh.ω, order::Integer=bh.order, type::Symbol=bh.type)
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
function constructH!(bh::BoseHamiltonian{Float}, order::Integer) where {Float<:AbstractFloat}
    (;J, U, f, ω, H) = bh
    (;index_of_state, ncells, neis_of_cell) = bh.lattice
    H .= 0
    H[diagind(H)] .= bh.E₀ .* U

    Jeff = J * besselj0(f)

    J_sum = zeros(Float, 2)
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
function constructH_diverging!(bh::BoseHamiltonian{Float}, order::Integer) where {Float<:AbstractFloat}
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
    r = zero(ω)
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
function constructH_dpt!(bh::BoseHamiltonian{Float}, order::Integer) where {Float<:AbstractFloat}
    (;index_of_state, ncells, neis_of_cell) = bh.lattice
    (;J, U, f, ω, E₀, space_of_state, H) = bh
    
    ε = Vector{Float}(undef, length(E₀)) # energies (including 𝑈 multiplier) reduced to first Floquet zone
    for i in eachindex(E₀)
        ε[i] = E₀[i]*U - space_of_state[i][2]*ω
    end
    H .= 0
    H[diagind(H)] .= ε

    R = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Bool}, Float}()
    R2 = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Bool}, Float}()
    R3 = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int}, Float}()
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

                        s = zero(Float) # terms of the sum
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

                        key = (E₀[α], E₀[β], E₀[γ], E₀[α′], i_j, k_l, m_n)
                        if !haskey(R3, key)
                            N = 20
                            t = zero(Float)
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
function constructH_dpt_quick!(bh::BoseHamiltonian{Float}, order::Integer) where {Float<:AbstractFloat}
    (;index_of_state, ncells, neis_of_cell) = bh.lattice
    (;J, U, f, ω, E₀, space_of_state, H) = bh

    ε = Vector{Float}(undef, length(E₀)) # energies (including 𝑈 multiplier) reduced to first Floquet zone
    for i in eachindex(E₀)
        ε[i] = E₀[i]*U - space_of_state[i][2]*ω
    end
    H .= 0
    H[diagind(H)] .= ε

    R = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Bool}, Float}()
    R2 = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Bool}, Float}()
    R3 = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int}, Float}()
    
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
                            s = zero(Float) # terms of the sum
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

                            key = (E₀[α], E₀[β], E₀[γ], E₀[α′], i_j, k_l, m_n)
                            if !haskey(R3, key)
                                N = 20
                                t = zero(Float)
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
        s = zero(U)
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
        s = zero(U)
        prange = skipzero ? [-N:-1; 1:N] : collect(-N:N)
        for p in prange
            s += 1 / (U*ΔE1 - (d1-p)*ω) / (U*ΔE2 - (d2+p)*ω) * besselj(i1, f*x1) * besselj(i2-p, f*x2) * besselj(i3+p, f*x3)
        end
        R[key] = s
    end
    return R[key]
end

"""
Calculate and return the spectrum for the values of 𝑈 in `Us`, using degenerate theory: `type` should be `:dpt` or `:dpt_quick`.
If `type=:dpt_quick`, then `subspace` must contain the subspace number (as in `bh.space_of_state[:][1]`) of interest.
`bh` is used as a parameter holder, but `bh.U`, `bh.type`, and `bh.order` do not matter --- function arguments are used instead.
"""
function scan_U(bh0::BoseHamiltonian{Float}, r::Rational, ωₗ::Real, Us::AbstractVector{<:Real}, subspace::Integer=0; type::Symbol, order::Integer) where {Float<:AbstractFloat}
    (;J, f, ω) = bh0

    n_blas = BLAS.get_num_threads() # save original number of threads to restore later
    BLAS.set_num_threads(1)
    
    progbar = ProgressMeter.Progress(length(Us))
    if type == :dpt
        spectrum = Matrix{Float}(undef, size(bh0.H, 1), length(Us))
        Threads.@threads for iU in eachindex(Us)
            bh = BoseHamiltonian(bh0.lattice, J, Us[iU], f, ω, r, ωₗ; type, order);
            spectrum[:, iU] = eigvals(Symmetric(bh.H))
            ProgressMeter.next!(progbar)
        end
    elseif type == :dpt_quick
        # construct `space_of_state` because `bh0` does not necessarily contain it
        space_of_state = map(bh0.E₀) do E
            a = (E*r*ω - ωₗ) ÷ ω |> Int
            A = E % denominator(r)
            return (A, a)
        end
        As = findall(s -> s[1] == subspace, space_of_state) # `As` stores numbers of state that belong to space `subspace`
        spectrum = Matrix{Float}(undef, length(As), length(Us))
        Threads.@threads for iU in eachindex(Us)
            h = zeros(Float, length(As), length(As)) # reduced matrix of the subspace of interest
            bh = BoseHamiltonian(bh0.lattice, J, Us[iU], f, ω, r; type, order);
            for i in eachindex(As), j in i:length(As)
                h[j, i] = bh.H[As[j], As[i]]
            end
            spectrum[:, iU] = eigvals(Symmetric(h, :L))
            ProgressMeter.next!(progbar)
        end
    else
        @error "Unknown type" type
        spectrum = zeros(1, 1)
    end
    ProgressMeter.finish!(progbar)
    BLAS.set_num_threads(n_blas)
    return spectrum
end

"""
Calculate quasienergy spectrum of `bh` via monodromy matrix for each value of 𝑈 in `Us`.
By default, loop over `Us` is parallelised using all threads that julia was launched with: `nthreads=Threads.nthreads()`.
Setting `nthreads=1` makes the loop over `Us` sequential, but the diffeq solving uses BLAS threading.
For `nthreads > 1`, BLAS threading is turned off, but is restored to the original state upon finishing the calculation.

If `outdir` is passed, a new directory will be created (if it does not exist) where the quasienergy spectrum at each `i`th value of `Us`
will be output immediately after calculation. The files are named as "<i>.txt"; the first value in the files is `Us[i]`, and the following
are the quasienergies.
"""
function quasienergy(bh::BoseHamiltonian{Float}, Us::AbstractVector{<:Real}; nthreads::Int=Threads.nthreads(), outdir::String="") where {Float<:AbstractFloat}
    (;J, f, ω, E₀) = bh
    Cmplx = (Float == Float32 ? ComplexF32 : ComplexF64)
    (;index_of_state, ncells, neis_of_cell) = bh.lattice
    
    H_rows, H_cols, H_vals = Int[], Int[], Cmplx[]
    H_sign = Float[] # stores the sign of the tunneling phase for each off-diagonal element, multiplied by `f`

    # Fill the off-diagonal elemnts of the Hamiltonian for `f` = 0
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
                    push!(H_sign, s*f)
                end
            end
        end
    end

    nstates = size(bh.H, 1)

    # append placeholders for storing diagonal elements
    append!(H_vals, zeros(Float, nstates))
    append!(H_rows, 1:nstates)
    append!(H_cols, 1:nstates)

    n_U = length(Us)
    ε = Matrix{Float}(undef, nstates, n_U)
    C₀ = Matrix{Cmplx}(I, nstates, nstates)
    
    T = 2π / ω
    tspan = (0.0, T)

    if nthreads > 1
        n_blas = BLAS.get_num_threads() # save original number of threads to restore later
        BLAS.set_num_threads(1)
    end
    executor = FLoops.ThreadedEx(basesize=length(Us)÷nthreads)

    # progbar = ProgressMeter.Progress(length(Us); enabled=showprogress) # slows down conputation up to 1.5 times!

    # if `outdir` is given but does not exist, then create it
    (outdir != "" && !isdir(outdir)) && mkdir(outdir)

    @floop executor for (i, U) in enumerate(Us)
        @init begin
            # diagonal of `H_buff` will remain equal to -𝑖𝑈 times the diagonal of `H` throughout diffeq solving,
            # while off-diagnoal elemnts will be mutated at each step
            H_vals_buff = similar(H_vals)
        end
        H_vals_buff[end-nstates+1:end] .= U .* (-im .* E₀) # update last `nstates` values in `H_vals_U` -- these are diagonal elements of the Hamiltonian
        params = (H_rows, H_cols, H_vals_buff, H_vals, H_sign, ω, nstates)
        prob = ODEProblem(schrodinger!, C₀, tspan, params, save_everystep=false)
        sol = solve(prob, Tsit5())
        ε[:, i] = -ω .* angle.(eigvals(sol[end])) ./ 2π
        if outdir != ""
            open(joinpath(outdir, "$(i).txt"), "w") do io
                writedlm(io, vcat([U], ε[:, i])) 
            end
        end
        # ProgressMeter.next!(progbar)
    end
    # ProgressMeter.finish!(progbar)
    nthreads > 1 && BLAS.set_num_threads(n_blas) # restore original number of threads

    return ε
end

"Schrödinger equation used for monodromy matrix calculation."
function schrodinger!(du, u, p, t)
    H_rows, H_cols, H_vals_buff, H_vals_base, H_sign, ω, nstates = p
    H_vals_buff[1:end-nstates] .= @view(H_vals_base[1:end-nstates]) .* cis.(sin(ω.*t) .* H_sign) # update off diagonal elements of the Hamiltonian
    H = sparse(H_rows, H_cols, H_vals_buff)
    mul!(du, H, u)
end

"""
Calculate quasienergy spectrum via monodromy matrix for each value of 𝑈 in `Us`.
`bh` is used as a parameter holder, but `bh.U`, `bh.type`, and `bh.order` do not matter.
By default, loop over `Us` is parallelised using all threads that julia was launched with: `nthreads=Threads.nthreads()`.
Setting `nthreads=1` makes the loop over `Us` sequential, but the diffeq solving uses BLAS threading.
For `nthreads > 1`, BLAS threading is turned off, but is restored to the original state upon finishing the calculation.
"""
function quasienergy_dense(bh::BoseHamiltonian{Float}, Us::AbstractVector{<:Real}; nthreads::Int=Threads.nthreads()) where {Float<:AbstractFloat}
    (;J, f, ω, E₀) = bh
    (;index_of_state, ncells, neis_of_cell) = bh.lattice
    Cmplx = (Float == Float32 ? ComplexF32 : ComplexF64)

    nstates = size(bh.H, 1)
    H = zeros(Cmplx, nstates, nstates)
    H_sign = zeros(Int, nstates, nstates)

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
    ε = Matrix{Float}(undef, nstates, n_U)
    C₀ = Matrix{Cmplx}(I, nstates, nstates)
    
    T = 2π / ω
    tspan = (0.0, T)

    if nthreads > 1
        n_blas = BLAS.get_num_threads() # save original number of threads to restore later
        BLAS.set_num_threads(1)
    end
    executor = FLoops.ThreadedEx(basesize=length(Us)÷nthreads)

    # progbar = ProgressMeter.Progress(length(Us))  # slows down conputation up to 1.5 times!

    @floop executor for (i, U) in enumerate(Us)
        @init begin
            # diagonal of `H_buff` will remain equal to -𝑖𝑈 times the diagonal of `H` throughout diffeq solving,
            # while off-diagnoal elemnts will be mutated at each step
            H_buff = zeros(Cmplx, nstates, nstates)
        end
        H_buff[diagind(H_buff)] .= U .* (-im .* E₀)
        params = (H_buff, H, H_sign, ω, f)
        prob = ODEProblem(schrodinger_dense!, C₀, tspan, params, save_everystep=false)
        sol = solve(prob, Tsit5())
        ε[:, i] = -ω .* angle.(eigvals(sol[end])) ./ 2π

        # ProgressMeter.next!(progbar)
    end
    # ProgressMeter.finish!(progbar)
    nthreads > 1 && BLAS.set_num_threads(n_blas) # restore original number of threads

    return ε
end

"Dense version of the Schrödinger equation used for monodromy matrix calculation."
function schrodinger_dense!(du, u, params, t)
    H_buff, H_base, H_sign, ω, f = params
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