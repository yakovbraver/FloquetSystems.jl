using OrdinaryDiffEq, SparseArrays, DelimitedFiles, FastLapackInterface
using SpecialFunctions: besselj0, besselj
using LinearAlgebra: LAPACK, BLAS, Eigen, Symmetric, I, diagind, diag, eigen, eigvals, mul!
using Distributed, SharedArrays
using ProgressMeter: @showprogress
import ProgressMeter
using FLoops: @floop, @init
import FLoops
import Base.show, Base.copy

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
    ωₗ::Float # lower bound of the Floquet zone; needed for DPT calculations
    r::Rational{Int} # resonance number; needed for quick-DPT calculations
    type::Symbol     # `:dpt`, `:dpt_quick`, `:diverging` or anything else for ordinary high-frequency expansion
    order::Int  # oder of DPT
    space_of_state::Vector{Tuple{Int,Int}}    # space_of_state[i] stores the subspace number (𝐴, 𝑎) of i'th state, with 𝐴 = 0 assigned to all nondegenerate space
    H::Matrix{Float}   # the Hamiltonian matrix
    E₀::Vector{Int}    # zeroth-order spectrum, in units of 𝑈
    ε₀::Vector{Float}  # zeroth-order quasienergy spectrum
    R1::Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Bool}, Float} # required for DPT-2 calculation
    R2::Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Bool}, Float} # required for DPT-3 calculation
    R3::Dict{NTuple{7, Int64}, Float} # required for DPT-3 calculation
end

"""
Construct a `BoseHamiltonian` object defined on `lattice`.
Type of `J` determines the type of Float used for all fields of the resulting object.
`ωₗ` is the lower bound of the first Floquet zone.
"""
function BoseHamiltonian(lattice::Lattice, J::Float, U::Real, f::Real, ω::Real; ωₗ::Real=-ω/2, r::Rational=0//1, order::Integer=1, type::Symbol=:basic) where {Float <: AbstractFloat}
    nstates = length(lattice.basis_states)
    # Calculate zeroth-order spectrum, in units of 𝑈. It only depends on the lattice, so will not change
    E₀ = zeros(Int, nstates)
    for (index, state) in enumerate(lattice.basis_states)
        for n_i in state
            if (n_i > 1)
                E₀[index] += n_i * (n_i - 1) ÷ 2 # will always be divisible by 2
            end
        end
    end
    H = Matrix{Float}(undef, nstates, nstates)
    ε₀ = Vector{Float}(undef, nstates)
    R1 = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Bool}, Float}()
    R2 = Dict{Tuple{Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Bool}, Float}()
    R3 = Dict{NTuple{7, Int64}, Float}()
    space_of_state = (type in (:dpt, :dpt_quick) ? Vector{Tuple{Int,Int}}(undef, nstates) : Vector{Tuple{Int,Int}}())

    bh = BoseHamiltonian(lattice, Float(J), Float(U), Float(f), Float(ω), Float(ωₗ), r, type, order, space_of_state, H, E₀, ε₀, R1, R2, R3)
    update_params!(bh)

    return bh
end

"Print non-zero elements of the Hamiltonian `bh` in the format ⟨bra| Ĥ |ket⟩."
function Base.show(io::IO, bh::BoseHamiltonian{<:AbstractFloat})
    for C in CartesianIndices(bh.H)
        bh.H[C[1], C[2]] != 0 && println(io, bh.lattice.basis_states[C[1]], " Ĥ ", bh.lattice.basis_states[C[2]], " = ", round(bh.H[C[1], C[2]], sigdigits=3))
    end
end

"""
Return a shallow copy of a given `BoseHamiltonian`: copy all scalar fields; reference the same `bh.lattice` and hence copy `bh.E₀`;
make similar `bh.ε₀`; make empty copies of `bh.R*` so that they are not populated in the resulting object but are of the same allocated size (think `sizehint`).
"""
function Base.copy(bh::BoseHamiltonian{Float}) where {Float<:AbstractFloat}
    # since `E₀` depends on the lattice, we are copying `bh.E₀`
    b = BoseHamiltonian(bh.lattice, bh.J, bh.U, bh.f, bh.ω, bh.ωₗ, bh.r, bh.type, bh.order, similar(bh.space_of_state), similar(bh.H),
        copy(bh.E₀), similar(bh.ε₀), empty(bh.R1), empty(bh.R2), empty(bh.R3))
    sizehint!(b.R1, length(bh.R1)) 
    sizehint!(b.R2, length(bh.R2)) 
    sizehint!(b.R3, length(bh.R3))
    return b 
end

"Update parameters of `bh` and reconstruct `bh.H`."
function update_params!(bh::BoseHamiltonian{<:AbstractFloat}; J::Real=bh.J, U::Real=bh.U, f::Real=bh.f, ω::Real=bh.ω, ωₗ::Real=bh.ωₗ, r::Rational=bh.r, order::Integer=bh.order, type::Symbol=bh.type)
    bh.J = J; bh.U = U; bh.f = f; bh.ω = ω; bh.ωₗ = ωₗ; bh.r = r; bh.order = order; bh.type = type
    if type == :dpt
        map!(bh.space_of_state, bh.E₀) do E
            # rounding helps in cases such as when E*U - ωₗ = 29.9...96 and ÷10 gives 2 instead of 3
            a = round(E*U - ωₗ, sigdigits=6) ÷ ω |> Int
            A = E % denominator(r) # not used in DPT!
            return (A, a)
        end
        for i in eachindex(bh.E₀)
            bh.ε₀[i] = bh.E₀[i]*U - bh.space_of_state[i][2]*ω
        end
        constructH_dpt!(bh, order)
    elseif type == :dpt_quick
        map!(bh.space_of_state, bh.E₀) do E
            a = round(E*r*ω - ωₗ, sigdigits=6) ÷ ω |> Int
            A = clamp(E % denominator(r), 0, 1) # if division result is 0, then A = 0, otherwise A = 1
            return (A, a)
        end
        for i in eachindex(bh.E₀)
            bh.ε₀[i] = bh.E₀[i]*U - bh.space_of_state[i][2]*ω
        end
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
    (;J, U, f, ω, E₀, ε₀, space_of_state, H, R1, R2, R3) = bh

    H .= 0
    H[diagind(H)] .= ε₀

    empty!(R1); empty!(R2); empty!(R3);

    bra = similar(bh.lattice.basis_states[1])
    # take each basis state and find which transitions are possible
    for (ket, α) in index_of_state
        A, a = space_of_state[α]
        for i = 1:ncells # iterate over the terms of the Hamiltonian
            # 1st order
            for (j, i_j) in neis_of_cell[i]
                if (ket[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    copy!(bra, ket)
                    bra[j] -= 1
                    bra[i] += 1
                    α′ = index_of_state[bra]
                    _, a′ = space_of_state[α′]
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
                            copy!(bra, ket)
                            val *= √bra[l]
                            bra[l] -= 1
                            bra[k] += 1
                            val *= √bra[k]
                            _, b = space_of_state[index_of_state[bra]]
                            val *= √bra[j]
                            bra[j] -= 1
                            bra[i] += 1
                            α′ = index_of_state[bra]
                            _, a′ = space_of_state[α′]
                            val *= √bra[i]
                            val *= (get_R!(R1, U, ω, f, bra[i]-bra[j]-1, a′-b, i_j, k_l, a′, a, b, true) +
                                    get_R!(R1, U, ω, f, ket[l]-ket[k]-1, a-b, i_j, k_l, a′, a, b, true))
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
                        copy!(bra, ket)
                        val = -J^3/2
                        val *= √bra[n]; bra[n] -= 1; bra[m] += 1; val *= √bra[m]
                        bra[l] == 0 && continue
                        γ = index_of_state[bra]
                        _, c = space_of_state[γ]
                        val *= √bra[l]; bra[l] -= 1; bra[k] += 1; val *= √bra[k]
                        bra[j] == 0 && continue
                        β = index_of_state[bra]
                        _, b = space_of_state[β]
                        val *= √bra[j]; bra[j] -= 1; bra[i] += 1; val *= √bra[i]
                        α′ = index_of_state[bra]
                        _, a′ = space_of_state[α′]

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
                        N = 20
                        s += get!(R3, key) do
                            t = zero(Float)
                            for p in -N:N
                                p == 0 && continue
                                for q in -N:N
                                    (q == 0 || q == p) && continue
                                    # cast to Int32 as otherwise multiplication of three `besselj`s leads to allocations if `typeof(f) == Float32`
                                    t += besselj(Int32(b-a′-p), f*i_j) * besselj(Int32(c-b+p-q), f*k_l) * besselj(Int32(a-c+q), f*m_n) * (
                                            1 / 2(ε₀[α′] - ε₀[γ] - q*ω)     * (1/(ε₀[γ] - ε₀[β]  - (p-q)*ω) - 1/(ε₀[β] - ε₀[α′] + p*ω)) +
                                            1 / 2(ε₀[α]  - ε₀[β] - p*ω)     * (1/(ε₀[α] - ε₀[γ]  - q*ω)     - 1/(ε₀[γ] - ε₀[β]  - (p-q)*ω)) +
                                            1 / 6(ε₀[γ]  - ε₀[β] - (p-q)*ω) * (1/(ε₀[β] - ε₀[α′] + p*ω)     + 1/(ε₀[α] - ε₀[γ]  - q*ω)) -
                                            1 / 3(ε₀[α]  - ε₀[γ] - q*ω) / (ε₀[β] - ε₀[α′] + p*ω) )
                                end
                            end
                            t
                        end
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
    (;J, U, f, ω, E₀, ε₀, space_of_state, H, R1, R2, R3) = bh

    H .= 0
    H[diagind(H)] .= ε₀

    empty!(R1); empty!(R2); empty!(R3);
    
    bra = similar(bh.lattice.basis_states[1])
    # take each basis state and find which transitions are possible
    for (ket, α) in index_of_state
        A, a = space_of_state[α]
        for i = 1:ncells # iterate over the terms of the Hamiltonian
            # 1st order
            for (j, i_j) in neis_of_cell[i]
                if (ket[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    copy!(bra, ket)
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
                            copy!(bra, ket)
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
                                val *= (get_R!(R1, U, ω, f, bra[i]-bra[j]-1, a′-b, i_j, k_l, a′, a, b, skipzero) +
                                        get_R!(R1, U, ω, f, ket[l]-ket[k]-1, a-b, i_j, k_l, a′, a, b, skipzero))
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
                        copy!(bra, ket)
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
                            N = 20
                            s += get!(R3, key) do
                                t = zero(Float)
                                for p in -N:N
                                    (A == B && p == 0) && continue
                                    for q in -N:N
                                        ((A == C && q == 0) || (B == C && q == p)) && continue
                                        t += besselj(Int32(b-a′-p), f*i_j) * besselj(Int32(c-b+p-q), f*k_l) * besselj(Int32(a-c+q), f*m_n) * (
                                            1 / 2(ε₀[α′] - ε₀[γ] - q*ω)     * (1/(ε₀[γ] - ε₀[β]  - (p-q)*ω) - 1/(ε₀[β] - ε₀[α′] + p*ω)) +
                                            1 / 2(ε₀[α]  - ε₀[β] - p*ω)     * (1/(ε₀[α] - ε₀[γ]  - q*ω)     - 1/(ε₀[γ] - ε₀[β]  - (p-q)*ω)) +
                                            1 / 6(ε₀[γ]  - ε₀[β] - (p-q)*ω) * (1/(ε₀[β] - ε₀[α′] + p*ω)     + 1/(ε₀[α] - ε₀[γ]  - q*ω)) -
                                            1 / 3(ε₀[α]  - ε₀[γ] - q*ω) / (ε₀[β] - ε₀[α′] + p*ω) )
                                    end
                                end
                                t
                            end
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
    N = 20
    get!(R, key) do 
        s = zero(U)
        for n in -N:N
            skipzero && n == 0 && continue
            s += 1/(U*nα - (d+n)*ω) * besselj(Int32(-(a′-b+n)), f*i_j) * besselj(Int32(a-b+n), f*k_l)
        end
        s
    end
end

"Return key from the `R` dictionary; required for 3rd order DPT."
function get_R2!(R, U, ω, f, ΔE1, ΔE2, d1, d2, J_indices, J_args, skipzero)
    i1, i2, i3 = J_indices
    x1, x2, x3 = J_args
    key = (ΔE1, ΔE2, d1, d2, i1, i2, i3, x1, x2, x3, skipzero)
    N = 20
    get!(R, key) do
        s = zero(U)
        for p in -N:N
            skipzero && p == 0 && continue
            s += 1 / (U*ΔE1 - (d1-p)*ω) / (U*ΔE2 - (d2+p)*ω) * 
                 besselj(Int32(i1), f*x1) * besselj(Int32(i2)-p, f*x2) * besselj(Int32(i3)+p, f*x3)
        end
        s
    end
end

"""
Calculate and return the spectrum for the values of 𝑈 in `Us`, using degenerate theory.
Calculation is based on the parameters in `bh`, including `bh.order`.
`bh.type` will be checked and set to `:dpt` if not set already.

If `sort=true`, the second returned argument, which is the permutation matrix, will be populated.
This allows one to isolate the quasienergies of states having the largest overlap with the ground state.
"""
function dpt(bh0::BoseHamiltonian{Float}, Us::AbstractVector{<:Real}; sort::Bool=false, showprogress=true) where {Float<:AbstractFloat}
    bh0.type != :dpt && update_params!(bh0; type=:dpt)
    
    nthreads = Threads.nthreads()
    if nthreads > 1
        nblas = BLAS.get_num_threads() # save original number of threads to restore later
        BLAS.set_num_threads(1)
    end
        
    nstates = size(bh0.H, 1)
    
    nU = length(Us)
    spectrum = Matrix{Float}(undef, nstates, nU)
    sp = Matrix{Int}(undef, nstates, nU) # sorting matrix

    bh_chnl = Channel{BoseHamiltonian{Float}}(nthreads)
    worskpace_chnl = Channel{HermitianEigenWs{Float, Matrix{Float}, Float}}(nthreads)
    for _ in 1:nthreads
        put!(bh_chnl, copy(bh0))
        put!(worskpace_chnl, HermitianEigenWs(bh0.H, vecs=sort))
    end

    progbar = ProgressMeter.Progress(nU; enabled=showprogress)
    Threads.@threads for i in eachindex(Us)
        bh = take!(bh_chnl)
        worskpace = take!(worskpace_chnl)
        update_params!(bh; U=Us[i])
        if sort
            spectrum[:, i], v = LAPACK.syevr!(worskpace, 'V', 'A', 'U', bh.H, 0.0, 0.0, 0, 0, 1e-6)
            sortperm!(@view(sp[:, i]), @view(v[1, :]), rev=true, by=abs2)
        else
            spectrum[:, i] = LAPACK.syevr!(worskpace, 'N', 'A', 'U', bh.H, 0.0, 0.0, 0, 0, 1e-6)[1]
        end
        put!(bh_chnl, bh)
        put!(worskpace_chnl, worskpace)
        ProgressMeter.next!(progbar)
    end
    ProgressMeter.finish!(progbar)
    nthreads > 1 && BLAS.set_num_threads(nblas) # restore original number of threads
    return spectrum, sp
end

"""
Calculate and return the spectrum for the values of 𝑈 in `Us`, using DPT.
`subspace` must contain the subspace number (as in `bh.space_of_state[:][1]`) of interest.
Calculation is based on the parameters in `bh`, including `bh.order`.
`bh.type` will be checked and set to `:dpt_quick` if not set already.

If `sort=true`, the second returned argument, which is the permutation matrix, will be populated.
This allows one to isolate the quasienergies of states having the largest overlap with the ground state.
"""
function dpt_quick(bh0::BoseHamiltonian{Float}, Us::AbstractVector{<:Real}; sort::Bool=false, showprogress=true, subspace::Integer=0) where {Float<:AbstractFloat}
    bh0.type != :dpt_quick && update_params!(bh0; type=:dpt_quick)

    nthreads = Threads.nthreads()
    if nthreads > 1
        nblas = BLAS.get_num_threads() # save original number of threads to restore later
        BLAS.set_num_threads(1)
    end
        
    As = findall(s -> s[1] == subspace, bh0.space_of_state) # `As` stores numbers of state that belong to space `subspace`
    nstates = length(As)
    nU = length(Us)
    spectrum = Matrix{Float}(undef, nstates, nU)
    sp = Matrix{Int}(undef, nstates, nU)

    bh_chnl = Channel{BoseHamiltonian{Float}}(nthreads)
    worskpace_chnl = Channel{HermitianEigenWs{Float, Matrix{Float}, Float}}(nthreads)
    H_chnl = Channel{Matrix{Float}}(nthreads)
    for _ in 1:nthreads
        put!(bh_chnl, copy(bh0))
        put!(worskpace_chnl, HermitianEigenWs(Matrix{Float}(undef, nstates, nstates), vecs=sort))
        put!(H_chnl, Matrix{Float}(undef, nstates, nstates))
    end

    progbar = ProgressMeter.Progress(nU; enabled=showprogress)
    Threads.@threads for i in eachindex(Us)
        nanfound = false
        bh = take!(bh_chnl)
        worskpace = take!(worskpace_chnl)
        H = take!(H_chnl)
        
        update_params!(bh; U=Us[i])

        for i in eachindex(As), j in i:length(As)
            h = bh.H[As[j], As[i]]
            isnan(h) && (nanfound = true; break)
            H[j, i] = bh.H[As[j], As[i]]
        end

        # if `Us[i]` is such that DPT is invalid, then `bh.H` is (or is close to being) singular, so that Inf's will appear during diagonalisation.
        if nanfound
            spectrum[:, i] .= NaN
        else
            if sort
                spectrum[:, i], v = LAPACK.syevr!(worskpace, 'V', 'A', 'L', H, 0.0, 0.0, 0, 0, 1e-6)
                !isnan(spectrum[1, i]) && sortperm!(@view(sp[:, i]), @view(v[1, :]), rev=true, by=abs2)
            else
                spectrum[:, i] = LAPACK.syevr!(worskpace, 'N', 'A', 'L', H, 0.0, 0.0, 0, 0, 1e-6)[1]
            end
        end
        put!(bh_chnl, bh)
        put!(worskpace_chnl, worskpace)
        put!(H_chnl, H)
        ProgressMeter.next!(progbar)
    end
    ProgressMeter.finish!(progbar)
    nthreads > 1 && BLAS.set_num_threads(nblas) # restore original number of threads
    return spectrum, sp
end

"""
Analyse residual couplings with states outside the FZ.
The passed `bh` has to be initialised with the required ωₗ defining the FZ.
Take a state |𝛼0⟩⟩ together with the states |𝛼′𝑛⟩⟩ which it is coupled to in first order. Scan 𝑛 to find the strongest coupling ratio defined as
<coupling strength> / <energy distance>. Save this ratio to the correponding elements `bh.H[α′, α]`. Repeat for all (α, α′) pairs.
Additionally, return a matrix `W` with 𝑛's: `W[α′, α] = n`.
"""
function residuals!(bh::BoseHamiltonian{Float}) where {Float<:AbstractFloat}
    (;index_of_state, ncells, neis_of_cell) = bh.lattice
    (;J, f, ω, ε₀, space_of_state, H) = bh

    H .= 0
    W = zeros(Int, size(H))

    bra = similar(bh.lattice.basis_states[1])
    # take each basis state and find which transitions are possible
    for (ket, α) in index_of_state
        _, a = space_of_state[α]
        for i = 1:ncells # iterate over the terms of the Hamiltonian
            for (j, i_j) in neis_of_cell[i]
                if (ket[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    copy!(bra, ket)
                    bra[j] -= 1
                    bra[i] += 1
                    α′ = index_of_state[bra]
                    _, a′ = space_of_state[α′]
                    r_max, n_max = 0.0, 0
                    for n in -5:5 # large values of `n` are likely to lead to low ratios because of large energy distance
                        n == 0 && continue # skip levels inside the FZ
                        r = besselj(a - (a′ + n), f*i_j) / (ε₀[α] - (ε₀[α′] - n*ω)) |> abs # `n`s are with different signs because adding `n` to subspace number means subtracting `nω` from the energy
                        r > r_max && (r_max = r; n_max = -n) # -n will show the number of Floquet zone
                    end
                    H[α′, α] = J * r_max * sqrt( (ket[i]+1) * ket[j] )
                    W[α′, α] = n_max
                end
            end
        end
    end
    return W
end

function quasienergy(bh::BoseHamiltonian{Float}, Us::AbstractVector{<:Real}; outdir::String="", sort::Bool=false, showprogress::Bool=true, reltol::Real=1e-3) where {Float<:AbstractFloat}
    nstates = length(bh.lattice.basis_states)
    nU = length(Us)
    if nprocs() == 1
        ε = Matrix{Float}(undef, nstates, nU)
        sp = Matrix{Int}(undef, nstates, nU)
    else
        ε = SharedMatrix{Float}(nstates, nU)
        sp = SharedMatrix{Int}(nstates, nU)
    end
    quasienergy!(ε, sp, bh, Us; outdir, sort, showprogress, reltol)
    return ε, sp
end

"""
Calculate quasienergy spectrum of `bh` via monodromy matrix for each value of 𝑈 in `Us`.
Loop over `Us` is parallelised: using processes if Julia is launched with more than one process, and using threads otherwise.
BLAS threading is turned off, but is restored to the original state upon finishing the calculation.

If `outdir` is passed, a new directory will be created (if it does not exist) where the quasienergy spectrum at each `i`th value of `Us`
will be output immediately after calculation. The files are named as "<i>.txt"; the first value in the file is `Us[i]`, and the following
are the quasienergies.

If `sort=true`, the second argument (`sp`), which is the permutation matrix, will be populated.
This allows one to isolate the quasienergies of states having the largest overlap with the ground state.
If output is enabled using `outdir`, the files with have a second column where the first value is `Us[i]`, and the following are the permutation integers.

`reltol` is passed to the ODE solver.
"""
function quasienergy!(ε::AbstractMatrix, sp::AbstractMatrix, bh::BoseHamiltonian{Float}, Us::AbstractVector{<:Real}; outdir::String="", sort::Bool=false,
                      showprogress::Bool=true, reltol::Real=1e-3) where {Float<:AbstractFloat}
    (;J, f, ω, E₀) = bh
    Cmplx = (Float == Float32 ? ComplexF32 : ComplexF64)
    (;index_of_state, ncells, neis_of_cell) = bh.lattice
    
    H_rows, H_cols, H_vals = Int[], Int[], Cmplx[]
    H_sign = Float[] # stores the sign of the tunneling phase for each off-diagonal element

    # Fill the off-diagonal elemnts of the Hamiltonian for `f` = 0
    bra = similar(bh.lattice.basis_states[1])
    for (ket, index) in index_of_state
        for i = 1:ncells # iterate over the terms of the Hamiltonian
            for (j, s) in neis_of_cell[i]
                # 𝑎†ᵢ 𝑎ⱼ
                if (ket[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    val = -im * -J * sqrt( (ket[i]+1) * ket[j] ) # multiply by `-im` as on the rhs of ∂ₜ𝜓 = -i𝐻𝜓
                    copy!(bra, ket)
                    bra[j] -= 1
                    bra[i] += 1
                    row = index_of_state[bra]
                    push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    push!(H_sign, s)
                end
            end
        end
    end

    nstates = size(bh.H, 1)

    # append placeholders for storing diagonal elements
    append!(H_vals, zeros(Float, nstates))
    append!(H_sign, zeros(Float, nstates))
    append!(H_rows, 1:nstates)
    append!(H_cols, 1:nstates)

    nthreads = Threads.nthreads()
    np = nprocs()
    noblas = nthreads > 1 || np > 1
    if noblas
        nblas = BLAS.get_num_threads() # save original number of threads to restore later
        BLAS.set_num_threads(1)
    end

    C₀ = Matrix{Cmplx}(I, nstates, nstates)
    
    T = 2π / ω
    tspan = (0.0, T)
    
    S = sparse(H_rows, H_cols, H_sign) # construct the "sign" matrix
    H_sign_vals = nonzeros(S) # get the values so that they are in the same order as `H_base_vals`

    # construct the base matrix, which will not be mutated
    H_base = sparse(H_rows, H_cols, H_vals)
    H_rows, H_cols, H_base_vals = findnz(H_base)
    dind = (H_rows .== H_cols)

    outdir != "" && mkpath(outdir)

    params = (H_base, H_base_vals, H_base_vals, H_sign_vals, ω, f) # the contents of `params` do not matter as they will be replaced
    prob = ODEProblem(schrodinger!, C₀, tspan, params, save_everystep=false, save_start=false)

    if np > 1 # distributed mode
        nw = nworkers()
        pmap(1:nw) do pid
            # diagonal of `H_buff` will remain equal to the diagonal of `H_base` throughout diffeq solving, while off-diagnoal elements will be mutated at each step
            H_buff = H_base # `H_base` will be copied to each process, so we only set `H_buff` to reference it
            integrator = OrdinaryDiffEq.init(prob, Tsit5(); reltol)
            workspace = EigenWs(C₀, rvecs=sort)
            for i in pid:nw:length(Us)
                quasienergy_step!(ε, sp, i, Us[i], dind, E₀, H_buff, integrator, workspace, C₀, H_base_vals, H_sign_vals, ω, f, sort, outdir)
            end
        end
    else # threaded mode
        H_buff_chnl = Channel{typeof(H_base)}(nthreads)
        worskpace_chnl = Channel{EigenWs{Cmplx, Matrix{Cmplx}, Float}}(nthreads)
        integrator_chnl = Channel{typeof(OrdinaryDiffEq.init(prob, Tsit5(); reltol))}(nthreads)
        for _ in 1:nthreads
            put!(H_buff_chnl, copy(H_base))
            put!(worskpace_chnl, EigenWs(C₀, rvecs=sort))
            put!(integrator_chnl, OrdinaryDiffEq.init(prob, Tsit5(); reltol))
        end

        progbar = ProgressMeter.Progress(length(Us); enabled=showprogress)
        Threads.@threads for i in eachindex(Us)
            H_buff = take!(H_buff_chnl)
            integrator = take!(integrator_chnl)
            workspace = take!(worskpace_chnl)
            
            quasienergy_step!(ε, sp, i, Us[i], dind, E₀, H_buff, integrator, workspace, C₀, H_base_vals, H_sign_vals, ω, f, sort, outdir)

            put!(H_buff_chnl, H_buff)
            put!(integrator_chnl, integrator)
            put!(worskpace_chnl, workspace)

            ProgressMeter.next!(progbar)
        end
        ProgressMeter.finish!(progbar)
    end
    noblas && BLAS.set_num_threads(nblas) # restore original number of threads
end

function quasienergy_step!(ε, sp, i, U, dind, E₀, H_buff, integrator, workspace, C₀, H_base_vals, H_sign_vals, ω, f, sort, outdir)
    # diagonal of `H_buff` will remain equal to the diagonal of `H_base` throughout diffeq solving, while off-diagnoal elements will be mutated at each step
    H_buff_vals = nonzeros(H_buff) # a view to non-zero elements
    H_buff_vals[dind] .= U .* (-im .* E₀) # update diagonal of the Hamiltonian
    
    reinit!(integrator, C₀)
    integrator.p = (H_buff, H_buff_vals, H_base_vals, H_sign_vals, ω, f)
    sol = solve!(integrator)

    if sort
        t = LAPACK.geevx!(workspace, 'N', 'N', 'V', 'N', sol[end])
        e, v = t[2], t[4] # convenience views
        sortperm!(@view(sp[:, i]), @view(v[1, :]), rev=true, by=abs2)
        @. ε[:, i] = -ω * angle(e) / 2π
    else
        LAPACK.geevx!(workspace, 'N', 'N', 'N', 'N', sol[end])
        @. ε[:, i] = -ω * angle(workspace.W) / 2π
    end

    if outdir != ""
        open(joinpath(outdir, "$(i).txt"), "w") do io
            if sort
                writedlm(io, vcat([U U], [ε[:, i] sp[:, i]]))
            else
                writedlm(io, vcat([U], ε[:, i]))
            end
        end
    end
end

"Schrödinger equation used for monodromy matrix calculation."
function schrodinger!(du, u, params, t)
    H_buff, H_buff_vals, H_base_vals, H_sign_vals, ω, f = params
    # update off diagonal elements of the Hamiltonian
    p = cis(f * sin(ω*t)); n = cis(-f * sin(ω*t));
    for (i, s) in enumerate(H_sign_vals)
        if s > 0 
            H_buff_vals[i] = H_base_vals[i] * p
        elseif s < 0
            H_buff_vals[i] = H_base_vals[i] * n
        end
    end
    mul!(du, H_buff, u)
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