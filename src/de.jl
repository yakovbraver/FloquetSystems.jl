function diff_evolution(fn::Function; lower::Array{Float64}, upper::Array{Float64}, algorithm::Symbol=:rand1bin, npoints::Int=10,
                        CR::Float64=0.9, F::Float64=0.8, maxiter::Int=200, trace_step::Int=maxiter+1,
                        constrain_bounds::Bool=false, reltol::Float64=1e-6, abstol::Float64=1e-6, vtr::Float64=-Inf)
    if !all(upper .> lower)
        @error "Not all elements of `upper` are greater than the elements of `lower`. Terminating."
        return
    end
    if !in(algorithm, [:rand1bin, :localtobest1bin])
        @warn "Unknown algoritm '$algorithm' received. 'rand1bin' will be used instead."
        algorithm = :rand1bin
    end
    if npoints < 5
        @warn "Number of points should be at least 5 (received $npoints). 5 points will be used."
        npoints = 5
    end
    if CR < 0 || CR > 1
        @warn "CR must be in the range [0, 1] (received $CR). Setting CR=0.9"
        CR = 0.9
    end
    if F ≤ 0 || F > 1
        @warn "F must be in the range (0, 1] (received $F). Setting F=0.8"
        F = 0.8
    end

    D = length(lower) # dimension
    x = [(upper[i] - lower[i])rand() + lower[i] for i = 1:D, j = 1:npoints] # generate initial points
    u = Array{Float64}(undef, D, npoints)      # for storing mutated points
    f_values = [fn(x[:, i]) for i = 1:npoints] # make a list of function values at every original point

    best_value = minimum(f_values)
    best_point_index = argmin(f_values)
    termination_reason = :maxiter

    iteration = 1
    @views @inbounds for outer iteration = 1:maxiter
        for i = 1:npoints
            # generate indices of points used for mutation
            a = rand(1:npoints); while a == i   a = rand(1:npoints); end
            b = rand(1:npoints); while b == i || b == a   b = rand(1:npoints); end
            j_m = rand(1:D) # an index of the parameter that will be certainly mutated

            if algorithm == :rand1bin
                c = rand(1:npoints); while c == i || c == b || c == a   c = rand(1:npoints); end
                for j = 1:D     # for each parameter
                    if rand() ≤ CR || j == j_m   # mutate the parameter
                        u[j, i] = x[j, c] + F * (x[j, b] - x[j, a])
                        if constrain_bounds == true
                            if u[j, i] < lower[j]
                                u[j, i] = (x[j, c] + lower[j]) / 2   # bounce back
                            elseif u[j, i] > upper[j]
                                u[j, i] = (upper[j] + x[j, c]) / 2   # bounce back
                            end
                        end
                    else        # no mutation -- just copy the value
                        u[j, i] = x[j, i]
                    end
                end
            elseif algorithm == :localtobest1bin
                for j = 1:D     # for each parameter
                    if rand() ≤ CR || j == j_m   # mutate the parameter
                        u[j, i] = x[j, i] + F * (x[j, b] - x[j, a] + x[j, best_point_index] - x[j, i])
                        if constrain_bounds == true
                            if u[j, i] < lower[j]
                                u[j, i] = (x[j, i] + lower[j]) / 2   # bounce back
                            elseif u[j, i] > upper[j]
                                u[j, i] = (upper[j] + x[j, i]) / 2   # bounce back
                            end
                        end
                    else        # no mutation -- just copy the value
                        u[j, i] = x[j, i]
                    end
                end
            end
        end # loop through ensemble

        # select the next generation
        for i = 1:npoints
            f_mutated = fn(u[:, i])       # evaluate the function at the mutated point
            if f_mutated ≤ f_values[i]    # if the mutated point is better than the original
                x[:, i] .= u[:, i]        # select the mutated point for the new generation
                f_values[i] = f_mutated   # update the list of all function values
                if f_mutated < best_value     # check whether the mutated point gives the best function value
                    best_value = f_mutated    # update the best value
                    best_point_index = i      # update the index of the best point
                end
            end
        end

        if iteration % trace_step == 0
            println("Generation ", iteration, ": best value = ", best_value)
        end

        if best_value ≤ vtr
            termination_reason = :vtr
            break
        end

        if maximum(f_values) - best_value < reltol * abs(best_value) ||
           maximum(f_values) - best_value < abstol
            termination_reason = :tol
            break
        end 
    end

    (best_value, x[:, best_point_index], iteration, termination_reason)
end