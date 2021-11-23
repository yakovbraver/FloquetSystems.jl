using KrylovKit

include("hamiltonian.jl")
include("de.jl")
include("sa.jl")

"Neighbour function for Simulated Annealing."
function make_neighbour!(x_new::Vector{<:Integer}, x_current::Vector{<:Integer}, bh::BoseHamiltonian)
    ncells = prod(bh.lattice.dims)
    x_new .= x_current
    mutant_index = rand(1:length(x_current))    # index of the defect that will be moved
    # generate a new position for the defect, repeat until it does not coincide with existing defects
    while x_new[mutant_index] in x_current
        x_new[mutant_index] = rand(1:ncells)
    end
    old_defects = findall(bh.lattice.is_defect)
    dublicates = intersect(old_defects, x_new)
    move_defects!(bh, setdiff(old_defects, dublicates), setdiff(x_new, dublicates))
end

"Goal function for Simulated Annealing."
function goal_func_sa(bh::BoseHamiltonian)
    vals, _, _ = eigsolve(bh.H, 1, :SR)
    vals[1]
end

"Goal function for Differential Evolution."
function goal_func_de!(bh::BoseHamiltonian, x)
    old_defects = findall(bh.lattice.is_defect)
    new_defects = ceil.(Int, x)
    # check for defects occupying the same cell
    l = length(new_defects)
    for i in 1:l, j in i+1:l
        new_defects[i] == new_defects[j] && return Inf
    end
    dublicates = intersect(old_defects, new_defects)
    move_defects!(bh, setdiff(old_defects, dublicates), setdiff(new_defects, dublicates))

    vals, _, _ = eigsolve(bh.H, 1, :SR)
    vals[1]
end

"""
Find the optimal configuration of defects using `method`, and return a tuple `(value, defects)`.
The Hamiltonian object `bh` should not contain any defects, they will be added and optimised during execution of the function.
When the function returns, the defects are set to the optimal configuration on `bh.lattice`. 
"""
function optimise_defects!(bh::BoseHamiltonian, ndefects::Integer; method::Symbol)
    # initialisation: randomly place the defects on `bh.lattice`
    remove_defects!(bh, findall(bh.lattice.is_defect)) # clean the lattice as it may contain defects
    ncells = prod(bh.lattice.dims)
    defects = Vector{Int}(undef, ndefects)
    new_defect = rand(1:ncells)
    for i in eachindex(defects)
        while new_defect in defects
            new_defect = rand(1:ncells)
        end
        defects[i] = new_defect
    end
    add_defects!(bh, defects)

    # optimise using the chosen method
    best_val = 0
    if method == :sa
        best_val, defects = anneal!(bh, defects, make_neighbour!, goal_func_sa, niterations=500, show_every=50)
    else
        best_val, defects_index, _, _ = diff_evolution(x -> goal_func_de!(bh, x); lower=ones(ndefects), upper=ones(ndefects)*ncells, algorithm=:rand1bin, npoints=10,
                                            CR=0.1, F=0.9, maxiter=400, trace_step=50, constrain_bounds=true)
        defects = ceil.(Int, defects_index)
    end

    # set the defect to the optimal configuration on `bh.lattice`
    old_defects = findall(bh.lattice.is_defect)
    dublicates = intersect(old_defects, defects)
    move_defects!(bh, setdiff(old_defects, dublicates), setdiff(defects, dublicates))
    
    (best_val, defects)
end