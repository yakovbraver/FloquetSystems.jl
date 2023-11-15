using SparseArrays, Combinatorics, SpecialFunctions
using DifferentialEquations, FLoops
using LinearAlgebra: diagind, eigvals, I

import Base.show

"""
A type representing a Bose-Hubbard Hamiltonian,
    H = - ∑ 𝐽ᵢⱼ 𝑎†ᵢ 𝑎ⱼ, + 𝑈/2 ∑ 𝑛ᵢ(𝑛ᵢ - 1).
"""
mutable struct BoseHamiltonian
    J::Float64
    U::Float64
    f::Float64 # F / ω
    ncells::Int
    nbozons::Int
    H::SparseMatrixCSC{ComplexF64, Int} # the Hamiltonian matrix
    basis_states::Vector{Vector{Int}}     # all basis states as a vector (index => state)
    index_of_state::Dict{Vector{Int},Int} # a dictionary (state => index)
end

"Construct a `BoseHamiltonian` object defined on `lattice`."
function BoseHamiltonian(J::Real, U::Real, f::Real, ncells::Integer, nbozons::Integer; isperiodic::Bool)
    nstates = binomial(nbozons+ncells-1, nbozons)
    bh = BoseHamiltonian(float(J), float(U), float(f), ncells, nbozons, spzeros(Float64, 1, 1), Vector{Vector{Int}}(undef, nstates), Dict{Vector{Int},Int}())
    makebasis!(bh)
    constructH!(bh, isperiodic)
    bh
end

"Construct the Hamiltonian matrix."
function constructH!(bh::BoseHamiltonian, isperiodic::Bool)
    H_rows, H_cols, H_vals = Int[], Int[], Float64[]
    J = bh.J * besselj0(bh.f)
    # take each basis state and find which transitions are possible
    for (state, index) in bh.index_of_state
        val_d = 0.0 # diagonal value
        for i = 1:bh.ncells # iterate over the terms of the Hamiltonian
            # 𝑛ᵢ(𝑛ᵢ - 1)
            if (state[i] > 1) # check that at least two particles are present at site `i` so that destruction 𝑎ⱼ𝑎ⱼ is possible
                val_d += bh.U/2 * state[i] * (state[i] - 1)
            end
            # 𝑎†ᵢ 𝑎ⱼ
            for j in (i-1, i+1)
                if j == 0
                    !isperiodic && continue
                    j = bh.ncells
                elseif j == bh.ncells + 1
                    !isperiodic && continue
                    j = 1
                end
                if (state[j] > 0) # check that a particle is present at site `j` so that destruction 𝑎ⱼ is possible
                    val = -J * sqrt( (state[i]+1) * state[j] )
                    bra = copy(state)
                    bra[j] -= 1
                    bra[i] += 1
                    row = bh.index_of_state[bra]
                    push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                end
            end
        end
        push_state!(H_rows, H_cols, H_vals, val_d; row=index, col=index)
    end
    bh.H = sparse(H_rows, H_cols, H_vals)
end

function push_state!(H_rows, H_cols, H_vals, val; row, col)
    push!(H_vals, val)
    push!(H_cols, col)
    push!(H_rows, row)
end

"""
Generate all possible combinations of placing the bozons in the lattice.
Populate `bh.basis_states` and `bh.index_of_state`.
"""
function makebasis!(bh::BoseHamiltonian)
    index = 1 # unique index identifying the state
    (;ncells, nbozons) = bh
    for partition in integer_partitions(nbozons)
        length(partition) > ncells && continue # Example (nbozons = 3, ncells = 2): partition = [[3,0], [2,1], [1,1,1]] -- skip [1,1,1] as impossible
        append!(partition, zeros(ncells-length(partition)))
        for p in multiset_permutations(partition, ncells)
            bh.index_of_state[p] = index
            bh.basis_states[index] = p
            index += 1
        end
    end
end

"Print non-zero elements of the Hamiltonian `bh` in the format ⟨bra| Ĥ |ket⟩."
function Base.show(io::IO, bh::BoseHamiltonian)
    H_rows, H_cols, H_vals = findnz(bh.H)
    for (i, j, val) in zip(H_rows, H_cols, H_vals) # iterate over the terms of the Hamiltonian
        println(io, bh.basis_states[i], " Ĥ ", bh.basis_states[j], " = ", round(val, sigdigits=3))
    end
end

"""
Calculate quasienergy spectrum of `bh` via monodromy matrix for each value of 𝑈 in `Us`.
Passed `bh` should correrspond to 𝑈 = 1 and 𝐹 = 0.
"""
function quasienergy(bh::BoseHamiltonian, F::Real, ω::Real, Us::AbstractVector{<:Real})
    n_levels = size(bh.H, 1)
    n_U = length(Us)

    T = 2π / ω
    tspan = (0.0, T)
    
    ε = Matrix{Float64}(undef, n_levels, n_U)
    C₀ = Matrix{ComplexF64}(I, n_levels, n_levels)

    H = copy(bh.H)
    di = diagind(H)
    inter_term = H[di] # interaction term 𝑈/2 ∑ 𝑛ᵢ(𝑛ᵢ - 1) for 𝑈 = 1

    drive_term = similar(inter_term)
    for (state, index) in bh.index_of_state
        drive_term[index] = sum(F * j * state[j] for j in eachindex(state)) # ⟨s| ∑ 𝐹𝑗𝑛ⱼ |s⟩
    end

    H .*= -im # as in the lhs of the Schrödinger equation
    for (i, U) in enumerate(Us)
        params = (di, inter_term, U, drive_term, ω)
        H_op = DiffEqArrayOperator(H, update_func=update_func!)
        prob = ODEProblem(H_op, C₀, tspan, params, save_everystep=false)
        sol = solve(prob, MagnusGauss4(), dt=T/100)
        ε[:, i] = -ω .* angle.(eigvals(sol[end])) ./ 2π
    end

    return ε
end

"Update rhs operator (used for monodromy matrix calculation)."
function update_func!(H, u, p, t)
    di, inter_term, U, drive_term, ω = p
    @. H[di] .= -im * (inter_term * U + drive_term * cos(ω*t))
end