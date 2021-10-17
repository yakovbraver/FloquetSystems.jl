using KrylovKit
import Optim

include("hamiltonian.jl")

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
    x_new
end

super_val = 100;
super_x = [0, 0, 0, 0]

function goal_func(x, bh::BoseHamiltonian)
    vals, _, _ = eigsolve(bh.H, 1, :SR)
    if vals[1] < super_val
        global super_val = vals[1]
        super_x .= findall(bh.lattice.is_defect)
    end
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

    result = Optim.optimize(x -> goal_func(x, bh), Float64.(defects),
                Optim.SimulatedAnnealing(neighbor = (x_curr, x_new) -> make_neighbour!(Int.(x_new), Int.(x_curr), bh)),
                Optim.Options(iterations=500, show_trace=true, show_every=50))
    (Optim.minimizer(result), Optim.minimum(result))
end
