using Combinatorics: binomial, integer_partitions, multiset_permutations
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
function Lattice(;dims::Tuple{Int,Int}, nbozons::Integer=prod(dims), isperiodic::Bool)
    ncells = prod(dims)
    nstates = binomial(nbozons+ncells-1, nbozons)
    lattice = Lattice(ncells, nbozons, dims, isperiodic, Vector{Vector{Int}}(undef, nstates), Dict{Vector{Int},Int}(), Vector{Vector{Tuple{Int,Int}}}(undef, nstates))
    makebasis!(lattice)
    makeneis!(lattice)
    return lattice
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
        print(io, " $i")
        i % lattice.dims[2] == 0 && println(io)
    end
    println(io, "Total number of states: $(length(lattice.basis_states))")
end