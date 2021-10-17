using KrylovKit

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
end

function goal_func(bh::BoseHamiltonian)
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

    anneal!(bh, defects, niterations=500, show_every=50)
end

"Temperature as a function of iteration number, i ≥ 1"
temperature(i) = 1 / log(i)

"Simulated annealing, adapted from Optim.jl"
function anneal!(bh::BoseHamiltonian, current_state::Vector{<:Integer}; niterations::Integer, show_every=1000)
    proposal_state = similar(current_state)
    f_current = goal_func(bh)
    best_state = similar(current_state)
    f_best::Float64 = f_current
    for i in 1:niterations            
        # Determine the temperature for current iteration
        T = temperature(i)
        # Randomly generate a neighbor of our current state
        make_neighbour!(proposal_state, current_state, bh)
        # Evaluate the cost function at the proposed state
        f_proposal = goal_func(bh)

        if f_proposal <= f_current
            # If proposal is superior, we always move to it
            current_state .= proposal_state
            f_current = f_proposal

            # If the new state is the best state yet, keep a record of it
            if f_proposal < f_best
                f_best = f_proposal
                best_state .= proposal_state
            end
        else
            # If proposal is inferior, we move to it with probability p
            p = exp(-(f_proposal - f_current) / T)
            if rand() <= p
                current_state .= proposal_state
                f_current = f_proposal
            end
        end
        if i % show_every == 0
            println("iter $i: f_best = $f_best")
        end
    end
    (best_state, f_best)
end