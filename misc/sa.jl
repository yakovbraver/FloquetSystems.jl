include("hamiltonian.jl")

"Temperature as a function of iteration number, i ≥ 1"
temperature(i) = 1 / log(i)

"Simulated annealing, adapted from Optim.jl"
function anneal!(bh::BoseHamiltonian, current_state::Vector{<:Integer}, make_neighbour!::Function, goal_func::Function;
                 niterations::Integer, show_every=1000)
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
    (f_best, best_state)
end