includet("hamiltonian-1D.jl")

using SparseArrays, LinearAlgebra
using Plots, LaTeXStrings
using BenchmarkTools, SpecialFunctions
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

nbozons = 5; ncells = 5
binomial(nbozons+ncells-1, nbozons)
J = 1 # setting to 1 so that `U` is measured in units of `J`
U = 30sqrt(1.01)
f = 2
ω = 20
sqrt(2)J * besselj0(f) * nbozons
-2J * besselj0(f)
@time bh = BoseHamiltonian(J, U, f, ω, ncells, nbozons, isperiodic=true, type=:largeU, order=1)
# @time vals, vecs, info = eigsolve(bh.H, 100, krylovdim=126, :SR)
vals, vecs = eigen(Hermitian(Matrix(bh.H)))
plotstate(bh, vecs[:, 1], vals[1])
plot(abs2.(vecs[:, 1]))

M = copy(bh.H)
M[diagind(M)] .= 0
heatmap(M, yaxis=:flip, c=:coolwarm)
issymmetric(bh.H)
for i in axes(bh.H, 1)
    for j in axes(bh.H, 2)
        if !(bh.H[i, j] ≈ bh.H[j, i])
            println((i, j))
        end
    end
end

nvals = binomial(nbozons+ncells-1, nbozons)
nU = 1000
spectrum = Matrix{Float64}(undef, nvals, nU)
Us = range(1, 45, nU) .* √1.01
for (iU, U) in enumerate(Us)
    bh = BoseHamiltonian(J, U, f, ω, ncells, nbozons, isperiodic=true, type=:largeU, order=2)
    # vals, vecs, info = eigsolve(bh.H, nvals, krylovdim=nvals, :SR)
    # spectrum[:, iU] = vals[1:nvals]
    spectrum[:, iU] = eigvals(Symmetric(Matrix(bh.H)))
end

gr()
plotlyjs()
spectrum .%= ω
# spectrum .+= 2ω
spectrum[spectrum .< 0] .+= ω
scatter(Us, spectrum', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native);
scatter!(Us, spectrum' .- ω, xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native)
title!(L"\omega/J=%$ω, F/\omega=%$f"*", order=1")
savefig("spectrum.png")
savefig("exact.html")
yaxis!((-20, 20))
yaxis!((-10.5, 10.5))

# Exact quasienergy spectrum
nbozons = 5; ncells = 5
binomial(nbozons+ncells-1, nbozons)
J = 1 # setting to 1 so that `U` is measured in units of `J`
ω = 20
U = 1; F = 2ω
bh = BoseHamiltonian(J, U, 0, ω, ncells, nbozons, isperiodic=true)
Us = range(0, 20, 100)

ε = quasienergy(bh, F, ω, Us)
scatter(Us, ε', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native)
yaxis!((-1, 2)); yticks!(-1:1:2)

using DelimitedFiles
ε = readdlm("spectrum_F40.txt")
spectrum = ε
# open("spectrum_F40.txt", "w") do io
#     writedlm(io, ε)
# end

open("H.txt", "w") do io
    show(io, bh)
end