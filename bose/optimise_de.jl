using KrylovKit

include("de.jl")
include("hamiltonian.jl")

function goal_func(bh::BoseHamiltonian, x)
    old_defects = findall(bh.lattice.is_defect)
    new_defects = ceil.(Int, x)
    # check for defect occupying the same cell
    l = length(new_defects)
    for i in 1:l, j in i+1:l
        new_defects[i] == new_defects[j] && return Inf
    end
    dublicates = intersect(old_defects, new_defects)
    move_defects!(bh, setdiff(old_defects, dublicates), setdiff(new_defects, dublicates))

    vals, _, _ = eigsolve(bh.H, 1, :SR)
    vals[1]
end

function optimise_defects(bh::BoseHamiltonian, ndefects::Integer)
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

    diff_evolution(x -> goal_func(bh, x); lower=ones(ndefects), upper=ones(ndefects)*ncells, algorithm=:rand1bin, npoints=10,
                   CR=0.1, F=0.9, maxiter=200, trace_step=50, constrain_bounds=true)
end