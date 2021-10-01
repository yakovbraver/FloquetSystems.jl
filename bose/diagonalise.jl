# using Arpack
# A = spzeros(4, 4)
# A[2, 3] = 5
# A[3, 2] = 3
# e = eigs(A; nev=1, which=:SR)

using SparseArrays, Combinatorics, KrylovKit

"
Return Hamiltonian
    Ĥ = ∑ Cᵢⱼ â†ᵢ âⱼ,
where Cᵢⱼ (i ≤ j) is the (i ↔ j) tunneling strength. Only the upper triangle of Ĉ is used.
Also, return a vector of basis states on which Ĥ is defined.
Number of cells in the lattice is determined automatically as max(j).
"
function constructH(C::AbstractSparseMatrix; nbozons::Integer)
    C_rows, C_cols, C_vals = findnz(C)
    ncells = maximum(C_cols)
    H_rows, H_cols, H_vals = Int[], Int[], eltype(C_vals)[]
    index_of_state, state_at_index = makebasis(ncells, nbozons)
    for (state, index) in index_of_state
        for (i, j, c) in zip(C_rows, C_cols, C_vals) # iterate over the terms of the Hamiltonian
            if (state[j] > 0) # check that a particle is present at site `j` so that destruction âⱼ is possible
                H_val = c * sqrt( (state[i]+1) * state[j] )
                push!(H_vals, H_val)
                H_col = index
                push!(H_cols, H_col)
                bra = copy(state)
                bra[j] -= 1 
                bra[i] += 1
                H_row = index_of_state[bra]
                push!(H_rows, H_row)
                # place the conjugate element into the transposed position
                push!(H_vals, conj(H_val))
                push!(H_cols, H_row) 
                push!(H_rows, H_col)
            end
        end
    end
    sparse(H_rows, H_cols, H_vals), state_at_index
end
    

"""
Generate all possible combinations of placing `nbozons` bozons to `ncells` cells.
Return a dictionary (state => index) and a vector (index => state).
"""
function makebasis(ncells::Integer, nbozons::Integer)
    index_of_state = Dict{Vector{Int}, Int}()
    state_at_index = Vector{Vector{Int}}(undef, binomial(nbozons+ncells-1, nbozons))
    index = 1;
    for partition in integer_partitions(nbozons)
        length(partition) > ncells && continue # Example (nbozons = 3, ncells = 2): partition = [[3,0], [2,1], [1,1,1]] -- skip [1,1,1] as impossible
        append!(partition, zeros(ncells-length(partition)))
        for p in multiset_permutations(partition, ncells)
            index_of_state[p] = index
            state_at_index[index] = p
            index += 1
        end
    end
    index_of_state, state_at_index
end

C = spzeros(4, 4)
C[1, 2] = C[1, 3] = C[2, 4] = C[3, 4] = 1
H, index = constructH(C, nbozons=2)

A = spzeros(2, 2)
A[1, 2] = 1
H, index = constructH(A, nbozons=3)

vals, vecs, info = eigsolve(H, 1, :SR)

findall(x -> abs(x) >= 1e-3, vecs[5])
println(vecs[5])