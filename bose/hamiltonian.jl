using SparseArrays, Combinatorics

import Base.show

"A type representing a lattice."
mutable struct Lattice
    dims::Tuple{Int,Int}    # lattice dimensions
    J::SparseMatrixCSC{ComplexF64, Int}  # tunnelling strengths
    J_default::ComplexF64   # default tunnelling value (between the cells of the same type)
    phases::Vector{Float64} # phases induced at each cell
    nbozons::Int            # number of bozons 
    nstates::Int            # number of possible states (configurations)
    is_defect::BitVector    # indicates whether a cell is a defect or not
end

"""
Construct a `Lattice` object with `nbozons` bozons from the tunnelling strengths matrix `J`
and using the distribution of phases in `phases`. The default tunnelling strength between the cells of the same type
is specified by `J_default`. It is required when moving defects since it may be needed to restore the default value.
`J` is assumed to be hermitian, hence only the upper triangular part may be specified.
Number of cells is determined automatically as the number of rows in `J`.
"""
function Lattice(dims::Tuple{Integer,Integer}, J::SparseMatrixCSC{ComplexF64,Int64}, J_default::Number, phases::Vector{<:Real}, nbozons::Integer)
    ncells = prod(dims)
    Lattice(dims, J, ComplexF64(J_default), phases, nbozons, binomial(nbozons+ncells-1, nbozons), falses(ncells))
end

"""
Construct a rectangular `nrows`x`ncols` lattice with `nbozons` bozons.
The tunneling strengths are initialised to `J_default`.
𝐽ᵢⱼ describes the transition of a particle form cell 𝑗 to cell 𝑖, and only the lower triangular part of 𝐽̂ is populated.
Boundary conditions are controlled by `periodic`. If lattice is periodic, the phases are claculated automatically
to respect periodicity. If lattice is not periodic, a vector `Δϕ` of phases difference in the 𝑥 and 𝑦 directions may be given.
"""
function Lattice(;dims::Tuple{Integer,Integer}, J_default::Number, nbozons::Integer, Δϕ=[0.0, 0.0], periodic=true)
    nrows, ncols = dims;
    J_size = (ncols-1) * nrows + (nrows-1) * ncols
    if periodic
        Δϕ .= [2π / ncols, 2π / nrows]
        nrows > 2 && (J_size += ncols)
        ncols > 2 && (J_size += nrows)
    end
    J_rows = Vector{Int}(undef, J_size)
    J_cols = Vector{Int}(undef, J_size)
    J_vals = Vector{ComplexF64}(undef, J_size)
    phases = Vector{Float64}(undef, nrows*ncols)
    counter = 1
    cell = 1
    for col in 1:ncols, row in 1:nrows # iterate in column-major order
        if row < nrows # if there is a row below
            neighbour = cell + 1
            J_rows[counter] = neighbour
            J_cols[counter] = cell
            J_vals[counter] = J_default
            counter += 1
        end
        if col < ncols  # if there is a column to the right
            neighbour = cell + nrows
            J_rows[counter] = neighbour
            J_cols[counter] = cell
            J_vals[counter] = J_default
            counter += 1
        end
        phases[cell] = Δϕ[1] * col + Δϕ[2] * row
        cell += 1
    end
    if periodic
        ncols > 2 && for row in 1:nrows
            neighbour = row + (ncols-1) * nrows
            J_rows[counter] = neighbour
            J_cols[counter] = row
            J_vals[counter] = J_default
            counter += 1
        end
        nrows > 2 && for col in 1:ncols
            cell = (col-1)*nrows + 1
            neighbour = cell + nrows - 1
            J_rows[counter] = neighbour
            J_cols[counter] = cell
            J_vals[counter] = J_default
            counter += 1
        end
    end
    Lattice(dims, sparse(J_rows, J_cols, J_vals), J_default, phases, nbozons)
end


"""
A type representing a Bosonic Hamiltonian,
    Ĥ = − ∑ 𝐽ᵢⱼ â†ᵢ âⱼ,
note the minus sign.
"""
mutable struct BoseHamiltonian
    lattice::Lattice
    H::SparseMatrixCSC{ComplexF64, Int64} # the Hamiltonian matrix
    basis_states::Vector{Vector{Int}}     # all basis states as a vector (index => state)
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
    # take each basis state and find which transitions are possible
    for (state, index) in bh.index_of_state
        for (i, j, J) in zip(J_rows, J_cols, J_vals) # iterate over the terms of the Hamiltonian
            if (state[j] > 0) # check that a particle is present at site `j` so that destruction âⱼ is possible
                H_val = -J * sqrt( (state[i]+1) * state[j] )
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
Populate `bh.basis_states` and `bh.index_of_state`.
"""
function makebasis!(bh::BoseHamiltonian)
    index = 1 # unique index identifying the state
    ncells = prod(bh.lattice.dims)
    for partition in integer_partitions(bh.lattice.nbozons)
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
Return the flux piercing a square plaquette whose upper left corner is cell number `cell`.
The lattice is assumed not to be periodic.
"""
function get_flux(lattice::Lattice, cell::Integer)
    flux = 0.0
    froms = [cell, cell+1, cell+1 + lattice.dims[1], cell + lattice.dims[1]]
    for (i, from) in enumerate(froms)
        to = froms[i%4 + 1]
        # if `to < from`, then `lattice.J[to, from]` is not stored, so we use conjugate of `lattice.J[from, to]`
        flux += to > from ? angle(lattice.J[to, from]) : -angle(lattice.J[from, to])
    end
    flux % 2π
end

"Mark the cells with numbers given in `old_defects` as ordinary cells, recalculating the tunnelling matrix and the Hamiltonian."
function remove_defects!(bh::BoseHamiltonian, old_defects::Vector{<:Integer})
    move_defects!(bh::BoseHamiltonian, old_defects::Vector{<:Integer}, Int[])
end

"Mark the cells with numbers given in `new_defects` as defects, recalculating the tunnelling matrix and the Hamiltonian."
function add_defects!(bh::BoseHamiltonian, new_defects::Vector{<:Integer})
    move_defects!(bh::BoseHamiltonian, Int[], new_defects::Vector{<:Integer})
end

"""
Mark the cells with numbers given in `old_defects` as ordinary cells, and the cells with numbers given in `new_defects` as defects.
Rrecalculate the tunnelling matrix and the Hamiltonian.
"""
function move_defects!(bh::BoseHamiltonian, old_defects::Vector{<:Integer}, new_defects::Vector{<:Integer})
    # update positions of defects in the lattice
    bh.lattice.is_defect[old_defects] .= false
    bh.lattice.is_defect[new_defects] .= true
    # update elements of `bh.lattice.J` that 
    updated = Int[] # will hold indices of elements of `bh.lattice.J` that have been updated because of relocation of defects
    J_rows, J_cols, J_vals = findnz(bh.lattice.J)
    J_default = bh.lattice.J_default # for convenience
    for cells in (old_defects, new_defects) # first iterate over all cells that previously were defects, then over all cells that have become defects
        for cell in cells
            for cell_index in findall(==(cell), J_rows)
                push!(updated, cell_index)
                neighbour = J_cols[cell_index]
                cell_is_defect = bh.lattice.is_defect[cell]
                neighbour_is_defect = bh.lattice.is_defect[neighbour]
                if cell_is_defect == neighbour_is_defect
                    J_vals[cell_index] = J_default # restore the default tunneling strength
                else
                    # note that necessarily `cell` > `neighbour`, so `J_vals[cell_index]` describes the transition `cell` ← `neighbour`
                    ϕ = bh.lattice.phases[cell] + bh.lattice.phases[neighbour]
                    J_vals[cell_index] = cell_is_defect ? J_default * cis(ϕ/2) : -J_default * cis(-ϕ/2)
                end
            end
            for cell_index in findall(==(cell), J_cols)
                push!(updated, cell_index)
                neighbour = J_rows[cell_index]
                cell_is_defect = bh.lattice.is_defect[cell]
                neighbour_is_defect = bh.lattice.is_defect[neighbour]
                if cell_is_defect == neighbour_is_defect
                    J_vals[cell_index] = J_default # restore the default tunneling strength
                else
                    # note that necessarily `cell` < `neighbour`, so `J_vals[cell_index]` describes the transition `cell` → `neighbour`
                    ϕ = bh.lattice.phases[cell] + bh.lattice.phases[neighbour]
                    J_vals[cell_index] = cell_is_defect ? -J_default * cis(-ϕ/2) : J_default * cis(ϕ/2)
                end
            end
        end
    end
    bh.lattice.J = sparse(J_rows, J_cols, J_vals)

    # update the Hamiltonian matrix
    for (state, index) in bh.index_of_state
        for (i, j, J) in zip(J_rows[updated], J_cols[updated], J_vals[updated]) # iterate over the terms of the Hamiltonian
            if (state[j] > 0) # check that a particle is present at site `j` so that destruction âⱼ is possible
                H_val = -J * sqrt( (state[i]+1) * state[j] )
                bra = copy(state)
                bra[j] -= 1 
                bra[i] += 1
                bra_index = bh.index_of_state[bra]
                bh.H[bra_index, index] = H_val
                bh.H[index, bra_index] = conj(H_val)
            end
        end
    end
end