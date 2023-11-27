using SparseArrays, Combinatorics, SpecialFunctions
using DifferentialEquations, FLoops
using LinearAlgebra: diagind, eigvals, I
using ProgressMeter: @showprogress

import Base.show

"""
A type representing a Bose-Hubbard Hamiltonian,
    H = - ∑ 𝐽ᵢⱼ 𝑎†ᵢ 𝑎ⱼ, + 𝑈/2 ∑ 𝑛ᵢ(𝑛ᵢ - 1).
"""
mutable struct BoseHamiltonian
    J::Float64
    U::Float64
    f::Float64 # F / ω
    ω::Real
    ncells::Int
    nbozons::Int
    H::SparseMatrixCSC{ComplexF64, Int} # the Hamiltonian matrix
    basis_states::Vector{Vector{Int}}     # all basis states as a vector (index => state)
    index_of_state::Dict{Vector{Int},Int} # a dictionary (state => index)
end

"Construct a `BoseHamiltonian` object defined on `lattice`."
function BoseHamiltonian(J::Real, U::Real, f::Real, ω::Real, ncells::Integer, nbozons::Integer; isperiodic::Bool, order::Integer=1)
    nstates = binomial(nbozons+ncells-1, nbozons)
    bh = BoseHamiltonian(float(J), float(U), float(f), float(ω), ncells, nbozons, spzeros(Float64, 1, 1), Vector{Vector{Int}}(undef, nstates), Dict{Vector{Int},Int}())
    makebasis!(bh)
    # constructH!(bh, isperiodic, order)
    constructH_U!(bh, isperiodic, order)
    bh
end

"Construct the Hamiltonian matrix."
function constructH!(bh::BoseHamiltonian, isperiodic::Bool, order::Integer)
    H_rows, H_cols, H_vals = Int[], Int[], Float64[]
    (;J, U, f, ω) = bh
    Jeff = J * besselj0(f)

    J_sum = [0.0, 0.0]
    if order == 3
        a_max = 20
        J_sum[1] = (J/ω)^2 * U * sum(         besselj(a, f)^2 / a^2 for a in 1:a_max) # for 𝑗 = k
        J_sum[2] = (J/ω)^2 * U * sum((-1)^a * besselj(a, f)^2 / a^2 for a in 1:a_max) # for 𝑗 ≠ k
    end

    # take each basis state and find which transitions are possible
    for (state, index) in bh.index_of_state
        val_d = 0.0 # diagonal value
        for i = 1:bh.ncells # iterate over the terms of the Hamiltonian
            # 𝑛ᵢ(𝑛ᵢ - 1)
            if (state[i] > 1)
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
                    val = -Jeff * sqrt( (state[i]+1) * state[j] )
                    bra = copy(state)
                    bra[j] -= 1
                    bra[i] += 1
                    row = bh.index_of_state[bra]
                    push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                end
            end

            if order == 3
                for j in (i-1, i+1), k in (i-1, i+1)
                    if j == 0
                        !isperiodic && continue
                        j = bh.ncells
                    elseif j == bh.ncells + 1
                        !isperiodic && continue
                        j = 1
                    end
                    if k == 0
                        !isperiodic && continue
                        k = bh.ncells
                    elseif k == bh.ncells + 1
                        !isperiodic && continue
                        k = 1
                    end
                    C₁, C₂ = j == k ? J_sum : (J_sum[2], J_sum[1])
                    # 𝑎†ₖ (2𝑛ᵢ - 𝑛ⱼ) 𝑎ⱼ
                    if (state[j] > 0 && state[i] != state[j]-1)
                        val = C₁ * √( (k == j ? state[k] : state[k]+1) * state[j] ) * (2state[i] - (state[j]-1))
                        bra = copy(state)
                        bra[j] -= 1
                        bra[k] += 1
                        row = bh.index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end
                    # 𝑎†ⱼ (2𝑛ᵢ - 𝑛ⱼ) 𝑎ₖ
                    if (state[k] > 0 && state[i] != (j == k ? state[j]-1 : state[j]))
                        val = C₁ * √( (j == k ? state[j] : state[j]+1) * state[k] ) * (2state[i] - (j == k ? state[j]-1 : state[j]))
                        bra = copy(state)
                        bra[k] -= 1
                        bra[j] += 1
                        row = bh.index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end
                    # 𝑎†ᵢ 𝑎†ᵢ 𝑎ₖ 𝑎ⱼ
                    if ( (k == j && state[j] > 1) || (k != j && state[k] > 0 && state[j] > 0))
                        val = -C₂ * √( (state[i]+2) * (state[i]+1) * (k == j ? (state[j]-1)state[j] : state[j]state[k]))
                        bra = copy(state)
                        bra[j] -= 1
                        bra[k] -= 1
                        bra[i] += 2
                        row = bh.index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end
                    # 𝑎†ₖ 𝑎†ⱼ 𝑎ᵢ 𝑎ᵢ
                    if (state[i] > 1)
                        val = -C₂ * √( (k == j ? (state[j]+2) * (state[j]+1) : (state[k]+1) * (state[j]+1)) * (state[i]-1)state[i])
                        bra = copy(state)
                        bra[i] -= 2
                        bra[j] += 1
                        bra[k] += 1
                        row = bh.index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end
                end
            end
        end
        push_state!(H_rows, H_cols, H_vals, val_d; row=index, col=index)
    end
    bh.H = sparse(H_rows, H_cols, H_vals)
end

"Construct the Hamiltonian matrix."
function constructH_U!(bh::BoseHamiltonian, isperiodic::Bool, order::Integer)
    H_rows, H_cols, H_vals = Int[], Int[], Float64[]
    (;J, U, f, ω) = bh
    Jeff = J * besselj0(f)

    n_max = bh.nbozons + 1
    n_min = -bh.nbozons - 3
    R1 = Dict{Int64, Real}()
    R2 = Dict{Int64, Real}()
    for n in n_min:n_max
        R1[n] = 𝑅(ω, U*n, type=1)
        R2[n] = 𝑅(ω, U*n, type=2)
    end

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
                    val = -Jeff * sqrt( (state[i]+1) * state[j] )
                    bra = copy(state)
                    bra[j] -= 1
                    bra[i] += 1
                    row = bh.index_of_state[bra]
                    push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                end
            end

            if order == 2
                for (j, j₂, k) in zip((i-1, i+1), (i-2, i+2), (i+1, i-1))
                    if j == 0
                        j = bh.ncells
                    elseif j == bh.ncells + 1
                        j = 1
                    elseif k == 0
                        k = bh.ncells
                    elseif k == bh.ncells + 1
                        k = 1
                    end
                    if j₂ < 1
                        j₂ = bh.ncells + j₂
                    elseif j₂ > bh.ncells
                        j₂ = j₂ - bh.ncells
                    end

                    if (state[j] > 1)
                        # 𝑎†ᵢ 𝑎†ᵢ 𝑎ⱼ 𝑎ⱼ
                        n = state[i]+2 - (state[j]-2)
                        val = -J/2 * (R1[n - 3] - R1[n - 1]) * √((state[i]+1) * (state[i]+2) * state[j] * (state[j]-1))
                        bra = copy(state)
                        bra[j] -= 2
                        bra[i] += 2
                        row = bh.index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)

                        # 𝑎†ᵢ 𝑎†ⱼ₂ 𝑎ⱼ 𝑎ⱼ
                        n = state[i]+1 - (state[j]-2)
                        val = -J/2 * (R2[n - 2] - R2[n - 1]) * √((state[j₂]+1) * (state[i]+1) * state[j] * (state[j]-1))
                        bra = copy(state)
                        bra[j] -= 2
                        bra[i] += 1
                        bra[j₂] += 1
                        row = bh.index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end
                    # 𝑎†ᵢ 𝑎†ᵢ 𝑎ⱼ 𝑎ₖ
                    if (state[j] > 0 && state[k] > 0)
                        n = state[i]+2 - (state[j]-1)
                        val = -J/2 * (R2[n - 2] - R2[n - 1]) * √((state[i]+1) * (state[i]+2) * state[j] * state[k])
                        bra = copy(state)
                        bra[j] -= 1
                        bra[k] -= 1
                        bra[i] += 2
                        row = bh.index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end
                    # 𝑎†ₖ (𝑛ᵢ + 1) 𝑎ⱼ - 𝑎†ₖ 𝑛ᵢ 𝑎ⱼ
                    if (state[j] > 0)
                        n = state[i] - (state[j]-1)
                        val = -J/2 * R1[n] * (state[i] + 1) * √((state[k]+1) * state[j])
                              +J/2 * R1[n - 1] * state[i] * √((state[k]+1) * state[j])
                        bra = copy(state)
                        bra[j] -= 1
                        bra[k] += 1
                        row = bh.index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end
                    # 𝑎†ᵢ 𝑛ⱼ 𝑎ⱼ₂ - 𝑎†ᵢ (𝑛ⱼ + 1) 𝑎ⱼ₂
                    if (state[j₂] > 0)
                        n = state[i]+1 - state[j]
                        val = -J/2 * R1[n] * state[j] * √((state[i]+1) * state[j₂])
                              +J/2 * R1[n - 1] * (state[j] + 1) * √((state[i]+1) * state[j₂])
                        bra = copy(state)
                        bra[j₂] -= 1
                        bra[i] += 1
                        row = bh.index_of_state[bra]
                        push_state!(H_rows, H_cols, H_vals, val; row, col=index)
                    end
                    # 𝑛ⱼ (𝑛ᵢ + 1) - (𝑛ⱼ + 1) 𝑛ᵢ
                    if state[i] > 0 && state[j] > 0
                        n = state[i] - state[j]
                        val = -J/2 * R2[n + 1] * state[j] * (state[i]+1)
                            +J/2 * R2[n - 1] * (state[j] + 1) * state[i]
                        push_state!(H_rows, H_cols, H_vals, val; row=index, col=index)
                    end
                end
            end
        end
        push_state!(H_rows, H_cols, H_vals, val_d; row=index, col=index)
    end
    bh.H = sparse(H_rows, H_cols, H_vals)
end

function 𝑅(ω::Real, Un::Real; type::Integer)
    N = 5
    a₀ = round(Int, -Un / ω)
    # if `Un / ω` is integer, a₀ should be skipped in the sum
    a_range = isinteger(Un / ω) ? [a₀-N:a₀-1; a₀+1:a₀+N] : collect(a₀-N:a₀+N) # collect for type stability
    r = 0.0
    if type == 1
        for a in a_range
            a == 0 && continue
            r += 1/(a*ω + Un) * besselj(a, 1)^2 * (-1)^a
        end
    else
        for a in a_range
            a == 0 && continue
            r += 1/(a*ω + Un) * besselj(a, 1)^2
        end
    end
    return r
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

    H .*= -im # as on the lhs of the Schrödinger equation
    @showprogress for (i, U) in enumerate(Us)
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