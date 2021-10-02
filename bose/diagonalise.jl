# using Arpack
# A = spzeros(4, 4)
# A[2, 3] = 5
# A[3, 2] = 3
# e = eigs(A; nev=1, which=:SR)

using SparseArrays, Combinatorics, KrylovKit

import Base.show

"A type representing a lattice."
struct Lattice{T}
    J::SparseMatrixCSC{T} # tunneling strengths
    ncells::Int           # number of cells
    nbozons::Int          # number of bozons 
    nstates::Int          # number of possible states (configurations)
end

"""
Construct a `Lattice` object with `nbozons` bozons from the tunneling strengths matrix `J`.
`J` is assumed to be hermitian, hence only upper triangular part will be used.
Number of cells is determined automatically as the number of columns in `J`.
"""
function Lattice(J::AbstractSparseMatrix, nbozons::Integer)
    ncells = size(J)[2]
    Lattice(J, ncells, nbozons, binomial(nbozons+ncells-1, nbozons))
end


"A type representing a Bosonic Hamiltonian, Ĥ = ∑ 𝐽ᵢⱼ â†ᵢ âⱼ"
mutable struct BoseHamiltonian{T}
    lattice::Lattice{T}
    H::SparseMatrixCSC{T} # the Hamiltonian matrix
    states::Vector{Vector{Int}} # all possible states as a vector (index => state)
    index_of_state::Dict{Vector{Int},Int} # a dictionary (state => index)
end

"Construct a `BoseHamiltonian` object defined on `lattice`."
function BoseHamiltonian(lattice::Lattice)
    bh = BoseHamiltonian(lattice, spzeros(eltype(lattice.J), 1, 1), Vector{Vector{Int}}(undef, lattice.nstates), Dict{Vector{Int},Int}())
    makebasis!(bh)
    constructH!(bh)
    bh
end

"Construct the Hamiltonian matrix."
function constructH!(bh::BoseHamiltonian)
    J_rows, J_cols, J_vals = findnz(bh.lattice.J)
    H_rows, H_cols, H_vals = Int[], Int[], eltype(J_vals)[]
    for (state, index) in bh.index_of_state
        for (i, j, c) in zip(J_rows, J_cols, J_vals) # iterate over the terms of the Hamiltonian
            if (state[j] > 0) # check that a particle is present at site `j` so that destruction âⱼ is possible
                H_val = c * sqrt( (state[i]+1) * state[j] )
                push!(H_vals, H_val)
                H_col = index
                push!(H_cols, H_col)
                bra = copy(state)
                bra[j] -= 1 
                bra[i] += 1
                H_row = bh.index_of_state[bra]
                push!(H_rows, H_row)
                # place the conjugate element into the transposed position
                push!(H_vals, conj(H_val))
                push!(H_cols, H_row) 
                push!(H_rows, H_col)
            end
        end
    end
    bh.H = sparse(H_rows, H_cols, H_vals)
end

"""
Generate all possible combinations of placing the bozons in the lattice.
Populate `bh.states` and `bh.index_of_state`.
"""
function makebasis!(bh::BoseHamiltonian)
    index = 1;
    nb, nc = bh.lattice.nbozons, bh.lattice.ncells
    for partition in integer_partitions(nb)
        length(partition) > nc && continue # Example (nbozons = 3, ncells = 2): partition = [[3,0], [2,1], [1,1,1]] -- skip [1,1,1] as impossible
        append!(partition, zeros(nc-length(partition)))
        for p in multiset_permutations(partition, nc)
            bh.index_of_state[p] = index
            bh.states[index] = p
            index += 1
        end
    end
end

"Print non-zero elements of the Hamiltonian `bh` in the format ⟨bra| Ĥ |ket⟩"
function Base.show(io::IO, bh::BoseHamiltonian)
    H_rows, H_cols, H_vals = findnz(bh.H)
    for (i, j, val) in zip(H_rows, H_cols, H_vals) # iterate over the terms of the Hamiltonian
        println(io, bh.states[i], " Ĥ ", bh.states[j], " = ", round(val, sigdigits=3))
    end
end

J = spzeros(4, 4)
J[1, 2] = J[1, 3] = J[2, 4] = J[3, 4] = 1
nbozons = 2
lattice = Lattice(J, nbozons)
bh = BoseHamiltonian(lattice)

vals, vecs, info = eigsolve(bh.H, 1, :SR)

findall(x -> abs(x) >= 1e-3, vecs[1])