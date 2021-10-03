include("diagonalise.jl")

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
    x, y = 1:dims[1], 1:dims[2]
    state_matrix = transpose(reshape(final_state, dims[1], dims[2]))
    heatmap(x, y, state_matrix, xticks=x, yticks=y, yflip=true, color=:viridis) |> display
    state_matrix
end

J = spzeros(4, 4)
J[1, 2] = J[1, 3] = 1; J[2, 4] = J[3, 4] = 1
nbozons = 2
lattice = Lattice(J, nbozons)
bh = BoseHamiltonian(lattice)

vals, vecs, info = eigsolve(bh.H, 1, :SR)

plotstate(bh, vecs[1], (2, 2))

#-------
nbozons = 1
lattice3 = Lattice(3, 3, ComplexF64(1), nbozons)
bh = BoseHamiltonian(lattice3)

vals, vecs, info = eigsolve(bh.H, 1, :SR)

plotstate(bh, vecs[1], (3, 3))

#-------
nbozons = 1
lattice6 = Lattice(6, 6, ComplexF64(1), nbozons)
# Φ = π/3
# lattice6.J[1, 2] *= cispi(Φ)
# lattice6.J[5, 6] *= cispi(-Φ)
bh = BoseHamiltonian(lattice6)

vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1], (6, 6))