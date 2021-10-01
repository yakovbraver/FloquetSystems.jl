# using Arpack

using SparseArrays, Combinatorics

# A = spzeros(4, 4)
# A[2, 3] = 5
# A[3, 2] = 3
# e = eigs(A; nev=1, which=:SR)

using KrylovKit

"
Return Hamiltonian
    Ĥ = ∑ Cᵢⱼ â†ᵢ âⱼ,
where Cᵢⱼ (i ≤ j) is the (i ↔ j) tunneling strength. Only the upper triangle of Ĉ is used.
Number of cells in the lattice is determined as max(j).
"
function constructH(C::AbstractSparseMatrix; nbozons::Integer)
    C_rows, C_cols, C_vals = findnz(C)
    ncells = maximum(C_cols)
    H_rows, H_cols, H_vals = Int[], Int[], eltype(C_vals)[]
    for partition in makebasis(ncells, nbozons)
        for basis_vec in partition
            for (i, j, c) in zip(C_rows, C_cols, C_vals) # iterate over the terms of the Hamiltonian
                if (basis_vec[j] > 0) # check that a particle is present at site `j` so that destruction âⱼ is possible
                    H_val = c * sqrt( (basis_vec[i]+1) * basis_vec[j] )
                    push!(H_vals, H_val)
                    H_col = vectoint(basis_vec, base=nbozons+1)
                    push!(H_cols, H_col) 
                    basis_vec[j] -= 1 
                    basis_vec[i] += 1
                    H_row = vectoint(basis_vec, base=nbozons+1)
                    push!(H_rows, H_row)
                    # place the conjugate element into the transposed position
                    push!(H_vals, conj(H_val))
                    push!(H_cols, H_row) 
                    push!(H_rows, H_col)
                end
            end
        end
    end
    sparse(H_rows, H_cols, H_vals)
end
    

"""
Generate all possible combinations of placing `nbozons` bozons to `ncells` cells.
A vector `basis` of iterable objects is returned. Iterating over `basis[n]` yields all combinations where `n` cells are occupied.
"""
function makebasis(ncells::Integer, nbozons::Integer)
    basis = Vector{Combinatorics.MultiSetPermutations{Vector{Int}}}()
    # iterate partitions from smallest number of occupied cells to largest,
    # and break if we reach the number of occupied cells that exceeds the total number of cells in the lattice.
    # This is only possible if nbozons > ncells
    # Example (nbozons = 3, ncells = 2): partition = [[3,0], [2,1], [1,1,1]] -- break at [1,1,1]
    for partition in Iterators.reverse(integer_partitions(nbozons))
        length(partition) > ncells && break
        append!(partition, zeros(ncells-length(partition)))
        push!(basis, multiset_permutations(partition, ncells))
    end
    basis
end

"Return a number of base `base` constructed using the digits in `vec`"
function vectoint(vec::Vector{<:Integer}; base::Integer)
    num = zero(eltype(vec))
    for val in vec
        num = num * base + val
    end
    num
end

C = spzeros(4, 4)
C[1, 2] = C[1, 3] = C[2, 4] = C[3, 4] = 1
H = constructH(C, nbozons=2)

A = spzeros(2, 2)
A[1, 2] = 1
H = constructH(A, nbozons=3)

vals, vecs, info = eigsolve(H, 1, :SR)

findall(x -> abs(x) >= 1e-3, vecs[5])
println(vecs[5])