include("hamiltonian.jl")

using SparseArrays, KrylovKit
using Plots
pyplot()
theme(:dark, size=(600, 600))

"""
Plot a `state`, which is a superposition of the basis states of `bh`.
A rectangular lattice is assumed.
"""
function plotstate(bh::BoseHamiltonian, state::Vector{<:Number})
    ncells = prod(bh.lattice.dims)
    final_state = zeros(ncells)
    for i in eachindex(state)
        final_state .+= abs2(state[i]) .* bh.basis_states[i]
    end
    final_state ./= bh.lattice.nbozons # normalise to unity (otherwise normalised to `nbozons`)
    x, y = 1:bh.lattice.dims[2], 1:bh.lattice.dims[1]
    state_matrix = reshape(final_state, bh.lattice.dims[1], bh.lattice.dims[2])
    heatmap(x, y, state_matrix, xticks=x, yticks=y, yflip=true, color=:viridis) |> display
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
plotstate(bh, vecs[1])

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
add_defects!(bh, [11, 16, 21, 26])
get_flux(bh.lattice, 5)
get_flux(bh.lattice, 10)
get_flux(bh.lattice, 11)

vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1])