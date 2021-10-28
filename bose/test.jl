include("hamiltonian.jl")

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
    title!(L"\varepsilon = %$(round(ε, sigdigits=6))" * "; fluxes are in units of pi")
    
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
                annotate!([(col + 0.5, row + 0.5, (L"\frac{%$n}{%$d}", :white, 16))])
            end
        end
    end
    display(fig)

    state_matrix
end

#-------
nbozons = 1
lattice3 = Lattice(dims=(3, 3), J_default=1, periodic=true; nbozons)
bh = BoseHamiltonian(lattice3)
add_defects!(bh, [5])
get_flux(bh.lattice, (1, 1))
get_flux(bh.lattice, (2, 1))
get_flux(bh.lattice, (1, 2))
get_flux(bh.lattice, (2, 2))

vals, vecs, info = eigsolve(bh.H, 1, :SR)
plotstate(bh, vecs[1], vals[1])

#-------
nbozons = 1
# lattice35 = Lattice(dims=(3, 5), J_default=1, periodic=false, Δϕ=[π/2, π/2]; nbozons)
lattice35 = Lattice(dims=(3, 3), J_default=1, periodic=false, Δϕ=[π/3, π/3]; nbozons)
bh = BoseHamiltonian(lattice35)
add_defects!(bh, [5, 8])
get_flux(bh.lattice, (1, 1))
get_flux(bh.lattice, (2, 1))
get_flux(bh.lattice, (1, 2))
get_flux(bh.lattice, (2, 2))

#-------
nbozons = 1
lattice6 = Lattice(dims=(6, 6), J_default=1, periodic=true; nbozons)
bh = BoseHamiltonian(lattice6)

energies = [-4.0]
for pos in 1:6
    add_defects!(bh, [pos])
    vals, vecs, info = eigsolve(bh.H, 1, :SR)
    push!(energies, vals[1])
end

fs = plotstate(bh, vecs[1], vals[1])
en = copy(energies)
plot(0:6, en, marker=:circle, label="diagonal")
plot!(0:6, energies, marker=:circle, label="row", legend=:topleft)
energies_optimal = [-4, energies[2], energies[3], -3.920, -3.930, -3.912, -3.911]
plot!(0:6, energies_optimal, marker=:circle, label="optimised", xlabel="no. of defects")
title!("6x6 periodic, " * L"\Delta\phi=\pi/3")
savefig("6x6 periodic.pdf") 
#-------
nbozons = 1
lattice35 = Lattice(dims=(35, 35), J_default=1, periodic=true; nbozons)
bh = BoseHamiltonian(lattice35)
add_defects!(bh, collect(range(103, length=4, step=34)))

vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1], vals[1])
#------
include("optimise.jl")
ndefects = 7
nbozons = 1
lattice6 =  Lattice(dims=(6, 6), J_default=1, periodic=true, nϕ=1; nbozons)
bh = BoseHamiltonian(lattice6)
add_defects!(bh, [9, 10, 11, 16, 17, 22, 23])
move_defects!(bh, [15], [21])
best_defects, best_val = optimise_defects(bh, ndefects)
move_defects!(bh, findall(bh.lattice.is_defect), best_defects)
vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1], vals[1])
savefig("$(ndefects).pdf")