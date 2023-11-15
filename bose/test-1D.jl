includet("hamiltonian-1D.jl")

using SparseArrays, KrylovKit, LinearAlgebra
using Plots, LaTeXStrings
using BenchmarkTools
plotlyjs()
theme(:dark, size=(800, 600))

"""
Plot occupations of each lattice cell in a state `state`, which is a superposition of the basis states of `bh`.
A rectangular lattice is assumed.
"""
function plotstate(bh::BoseHamiltonian, state::Vector{<:Number}, ε::Float64)
    final_state = zeros(bh.ncells)
    for i in eachindex(state)
        final_state .+= abs2(state[i]) .* bh.basis_states[i]
    end
    final_state ./= bh.nbozons # normalise to unity (otherwise normalised to `nbozons`)
    
    fig = bar(1:ncells, final_state, title=L"\varepsilon = %$(round(ε, sigdigits=3))")
    display(fig)
end

nbozons = 5
ncells = 5
binomial(nbozons+ncells-1, nbozons)
J = 1 # setting to 1 so that `U` is measured in units of `J`
U = 3
f = 2
ω = 20
sqrt(2)J * besselj0(f) * nbozons
-2J * besselj0(f)
@time bh = BoseHamiltonian(J, U, f, ω, ncells, nbozons, isperiodic=true, order=3);
# @time vals, vecs, info = eigsolve(bh.H, 100, krylovdim=126, :SR)
vals, vecs = eigen(Hermitian(Matrix(bh.H)))
plotstate(bh, vecs[:, 1], vals[1])
plot(abs2.(vecs[:, 1]))

M = copy(bh.H)
M[diagind(M)] .= 0
heatmap(real.(M), yaxis=:flip)
ishermitian(bh.H)
nvals = binomial(nbozons+ncells-1, nbozons)
nU = 500
spectrum = Matrix{Float64}(undef, nvals, nU)
Us = range(0, 10, nU)
for (iU, U) in enumerate(Us)
    bh = BoseHamiltonian(J, U, f, ω, ncells, nbozons, isperiodic=true, order=3)
    # vals, vecs, info = eigsolve(bh.H, nvals, krylovdim=nvals, :SR)
    # spectrum[:, iU] = vals[1:nvals]
    spectrum[:, iU] = eigvals(Hermitian(Matrix(bh.H)))
end

spectrum .%= ω
spectrum[spectrum .< 0] .+= ω
scatter(Us, spectrum', xlabel="U/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, title="f = $f");
scatter!(Us, spectrum' .- ω, xlabel="U/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native);
title!("order = 3");
yaxis!((-10, 10))
yaxis!((-1, 2))

# Exact quasienergy spectrum
nbozons = 5; ncells = 5
binomial(nbozons+ncells-1, nbozons)
J = 1 # setting to 1 so that `U` is measured in units of `J`
ω = 20
U = 1; F = 2ω
bh = BoseHamiltonian(J, U, 0, ω, ncells, nbozons, isperiodic=true)
Us = range(0, 20, 10)

quasienergy(bh, F, ω, Us)
@btime ε = quasienergy(bh, F, ω, Us)
scatter(Us, ε', xlabel="U/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, title="F = $F")
yaxis!((-1, 2)); yticks!(-1:1:2)

# using DelimitedFiles
# open("spectrum_F40.txt", "w") do io
#     writedlm(io, ε)
# end