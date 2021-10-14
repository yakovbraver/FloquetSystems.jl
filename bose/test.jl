include("hamiltonian.jl")

using SparseArrays, KrylovKit
using Plots
pyplot()
theme(:dark, size=(600, 600))

"""
Plot a `state`, which is a superposition of the basis states of `bh`.
A rectangular lattice with dimensions `dims` is assumed.
"""
function plotstate(bh::BoseHamiltonian, state::Vector{<:Number}, dims::Tuple{Int,Int})
    ncells = bh.lattice.ncells
    final_state = zeros(ncells)
    for i in eachindex(state)
        final_state .+= abs2(state[i]) .* bh.basis_states[i]
    end
    final_state ./= bh.lattice.nbozons # normalise to unity (otherwise normalised to `nbozons`)
    x, y = 1:dims[2], 1:dims[1]
    state_matrix = reshape(final_state, dims[1], dims[2])
    heatmap(x, y, state_matrix, xticks=x, yticks=y, yflip=true, color=:viridis) |> display
    state_matrix
end

#-------
nbozons = 1
lattice3 = Lattice(nrows=3, ncols=3, J_default=1, periodic=true; nbozons)
bh = BoseHamiltonian(lattice3)
move_defects!(bh, Int[], Int[5])
vals, vecs, info = eigsolve(bh.H, 1, :SR)
(angle(bh.lattice.J[2, 1])+angle(bh.lattice.J[5, 2])-angle(bh.lattice.J[5, 4])-angle(bh.lattice.J[4, 1])) % 2π
(angle(bh.lattice.J[6, 3])-angle(bh.lattice.J[6, 5])-angle(bh.lattice.J[5, 2])+angle(bh.lattice.J[3, 2])) % 2π
(angle(bh.lattice.J[9, 6])-angle(bh.lattice.J[9, 8])-angle(bh.lattice.J[8, 5])+angle(bh.lattice.J[6, 5])) % 2π
(angle(bh.lattice.J[8, 5])-angle(bh.lattice.J[8, 7])-angle(bh.lattice.J[7, 4])+angle(bh.lattice.J[5, 4])) % 2π

plotstate(bh, vecs[1], (1, 6))

#-------
nbozons = 1
lattice1 = Lattice(nrows=1, ncols=6, J_default=1, periodic=true; nbozons)
bh = BoseHamiltonian(lattice1)

vals, vecs, info = eigsolve(bh.H, 1, :SR)

plotstate(bh, vecs[1], (1, 6))

#-------
nbozons = 1
lattice6 = Lattice(nrows=6, ncols=6, J_default=1, periodic=true; nbozons)
Φ = π/3

bh = BoseHamiltonian(lattice6)

vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1], (6, 6))