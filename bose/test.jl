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
    heatmap(x, y, state_matrix, xticks=x, yticks=y, yflip=true, color=:viridis)
    title!(L"\varepsilon = %$(round(ε, sigdigits=4))")
    
    defects = findall(bh.lattice.is_defect)
    defects_rows = [(cell-1) % bh.lattice.dims[1] + 1 for cell in defects]
    defects_cols = [(cell-1) ÷ bh.lattice.dims[1] + 1 for cell in defects]
    scatter!(defects_cols, defects_rows, color=:white, markersize=5, label="defect") |> display

    state_matrix
end

#-------
nbozons = 1
lattice3 = Lattice(dims=(3, 3), J_default=1, periodic=true; nbozons)
bh = BoseHamiltonian(lattice3)
add_defects!(bh, [5])
get_flux(bh.lattice, 1)
get_flux(bh.lattice, 2)
get_flux(bh.lattice, 5)
get_flux(bh.lattice, 4)

vals, vecs, info = eigsolve(bh.H, 1, :SR)
plotstate(bh, vecs[1], vals[1])

#-------
nbozons = 1
# lattice35 = Lattice(dims=(3, 5), J_default=1, periodic=false, Δϕ=[π/2, π/2]; nbozons)
lattice35 = Lattice(dims=(3, 3), J_default=1, periodic=false, Δϕ=[π/3, π/3]; nbozons)
bh = BoseHamiltonian(lattice35)
add_defects!(bh, [5, 8])
get_flux(bh.lattice, 1)
get_flux(bh.lattice, 2)
get_flux(bh.lattice, 4)
get_flux(bh.lattice, 5)

#-------
nbozons = 1
lattice6 = Lattice(dims=(6, 6), J_default=1, periodic=true; nbozons)
bh = BoseHamiltonian(lattice6)

energies = Float64[]
for pos in 6:5:31
    add_defects!(bh, [pos])
    vals, vecs, info = eigsolve(bh.H, 1, :SR)
    push!(energies, vals[1])
end

fs = plotstate(bh, vecs[1], vals[1])

plot(0:6, energies, marker=:circle, label="row 3-9-15...")
plot!(0:6, energies, marker=:circle, label="diag 6-11-16...", legend=:topleft)
title!("6x6 periodic, " * L"\Delta\phi=\pi/3")
savefig("6x6 periodic.pdf") 
#-------
nbozons = 1
lattice35 = Lattice(dims=(35, 35), J_default=1, periodic=true; nbozons)
bh = BoseHamiltonian(lattice35)
add_defects!(bh, collect(range(103, length=4, step=34)))
get_flux(bh.lattice, 5)
get_flux(bh.lattice, 10)
get_flux(bh.lattice, 11)

vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1], vals[1])
#------
include("optimise.jl")
ndefects = 3
nbozons = 1
lattice6 =  Lattice(dims=(6, 6), J_default=1, periodic=true; nbozons)
bh = BoseHamiltonian(lattice6)
best_defects, best_val = optimise_defects(bh, ndefects)
move_defects!(bh, findall(bh.lattice.is_defect), best_defects)
vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1], vals[1])
savefig("$ndefects.pdf")