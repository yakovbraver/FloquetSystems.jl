include("hamiltonian.jl")
include("optimise.jl")

using SparseArrays, KrylovKit
using Plots, LaTeXStrings
pyplot()
theme(:dark, size=(600, 600))

"""
Plot occupations of each lattice cell in a state `state`, which is a superposition of the basis states of `bh`.
A rectangular lattice is assumed.
"""
function plotstate(bh::BoseHamiltonian, state::Vector{<:Number}, ε::Float64)
    ncells = prod(bh.lattice.dims)
    final_state = zeros(ncells)
    for i in eachindex(state)
        final_state .+= abs2(state[i]) .* bh.basis_states[i]
    end
    final_state ./= bh.lattice.nbozons # normalise to unity (otherwise normalised to `nbozons`)
    x, y = 1:bh.lattice.dims[2], 1:bh.lattice.dims[1]
    state_matrix = reshape(final_state, bh.lattice.dims[1], bh.lattice.dims[2])
    fig = heatmap(x, y, state_matrix, xticks=x, yticks=y, yflip=true, color=:viridis)
    title!(L"\varepsilon = %$(round(ε, sigdigits=6))" * "; fluxes given in units of " * L"\pi")
    
    defects = findall(bh.lattice.is_defect)
    defects_rows = [(cell-1) % bh.lattice.dims[1] + 1 for cell in defects]
    defects_cols = [(cell-1) ÷ bh.lattice.dims[1] + 1 for cell in defects]
    scatter!(defects_cols, defects_rows, color=:white, markersize=5, label="defect")

    for row in 1:bh.lattice.dims[1]
        for col in 1:bh.lattice.dims[2]
            flux = get_flux(bh.lattice, (row, col)) / π
            if abs(flux) > 1e-3
                f = rationalize(flux, tol=1e-5)
                n, d = numerator(f), denominator(f)
                annotate!([(col + 0.5, row + 0.5, (L"\frac{%$n}{%$d}", :white, 8))]) # default size 16
            end
        end
    end
    display(fig)

    state_matrix
end

#-------
nbozons = 1
lattice3 = Lattice(dims=(6, 6), J_default=1, periodic=true, nϕ=2, driving_type=:linear; nbozons)
bh = BoseHamiltonian(lattice3)
add_defects!(bh, [11,16,21,26,31])
bh.lattice.J[6, 1]

vals, vecs, info = eigsolve(bh.H, 1, :SR)
plotstate(bh, vecs[1], vals[1])

#-------

nbozons = 1
lattice35 = Lattice(dims=(35, 35), J_default=1, periodic=true; nbozons)
bh = BoseHamiltonian(lattice35)
add_defects!(bh, collect(range(103, length=4, step=34)))

vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1], vals[1])

#------

ndefects = 6
nbozons = 1
lattice6 = Lattice(dims=(8, 8), J_default=1, periodic=true, nϕ=2; nbozons)
bh = BoseHamiltonian(lattice6)
add_defects!(bh, [13,12,19,20,27,29])
# move_defects!(bh, [15], [21])
best_val, best_defects = optimise_defects!(bh, ndefects, method=:sa)
move_defects!(bh, findall(bh.lattice.is_defect), best_defects)
vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1], vals[1])
savefig("$(ndefects).pdf")

#------

"Repeat optimisation `niter` times and return the best result as a tuple `(value, defects)`."
function search(lattice, ndefects; niter=20, method::Symbol)
    nthreads = Threads.nthreads()
    best_defects = Matrix{Int}(undef, ndefects, nthreads)
    best_vals = ones(nthreads)*100
    Threads.@threads for _ in 1:niter
        lat = deepcopy(lattice)
        bh = BoseHamiltonian(lat)
        val, defects = optimise_defects!(bh, ndefects; method)
        tid = Threads.threadid()
        if val < best_vals[tid]
            best_vals[tid] = val
            best_defects[:, tid] .= defects
        end
    end
    best_index = argmin(best_vals)
    (best_vals[best_index], best_defects[:, best_index])
end

ndefects = 7
nbozons = 1
lattice = Lattice(dims=(20, 20), J_default=1, periodic=true, nϕ=1, driving_type=:linear; nbozons)
best_val, best_defects = search(lattice, ndefects, niter=8, method=:de)
bh = BoseHamiltonian(lattice)
# add_defects!(bh, [9,10,30,28,48,49])
add_defects!(bh, best_defects)

vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1], vals[1])
savefig("$(ndefects)_optimal.pdf")

function param_sweep(;n_range, ndefects_range)
    for n in n_range
        foldername = "n = $n"
        !isdir(foldername) && mkdir(foldername)
        for ndefects in ndefects_range
            lattice = Lattice(dims=(20, 20), J_default=1, periodic=true, nϕ=n, driving_type=:linear; nbozons)
            best_val, best_defects = search(lattice, ndefects, niter=8, method=:de)
            # writedlm("$foldername/$(ndefects)_optimal.txt", best_defects)
            bh = BoseHamiltonian(lattice)
            add_defects!(bh, best_defects)

            vals, vecs, info = eigsolve(bh.H, 1, :SR)

            fs = plotstate(bh, vecs[1], vals[1])
            savefig("$foldername/$(ndefects)_optimal.pdf")
        end
    end
end

param_sweep(n_range=1:10, ndefects_range=3:7)