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
U = 1#sqrt(1.01)
f = 2
ω = 20
-2J * besselj0(f) * nbozons * cos(π/(ncells+1)) # non-periodic: exact ground state energy of 𝑊¹ at 𝑈 = 0; spectrum of 𝑊¹ at 𝑈 = 0 is NOT the exact quasienergy spectrum
-2J * besselj0(f) * nbozons # periodic: exact ground state energy of 𝑊¹ at 𝑈 = 0; spectrum of 𝑊¹ at 𝑈 = 0 IS the exact quasienergy spectrum
@time bh = BoseHamiltonian(J, U, f, ω, ncells, nbozons, isperiodic=false, type=:smallU, order=1)
# @time vals, vecs, info = eigsolve(bh.H, 100, krylovdim=126, :SR)
vals, vecs = eigen(Symmetric(Matrix(bh.H)))
scatter(vals, xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"f=%$f", markersize=2, markerstrokewidth=0, c=2, legend=false, ticks=:native)
scatter!([-2J * cos(π*i/(ncells+1)) for i in 1:ncells], xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"f=%$f", markersize=2, markerstrokewidth=0, c=2, legend=false, ticks=:native)

M = copy(bh.H)
M[diagind(M)] .= 0
heatmap(M, yaxis=:flip, c=:coolwarm)
heatmap(bh.H, yaxis=:flip, c=:coolwarm)

nvals = binomial(nbozons+ncells-1, nbozons)
nU = 1000
spectrum = Matrix{Float64}(undef, nvals, nU)
Us = range(0, 5, nU) #.* √1.01
for (iU, U) in enumerate(Us)
    bh = BoseHamiltonian(J, U, f, ω, ncells, nbozons, isperiodic=false, type=:smallU, order=2)
    # vals, vecs, info = eigsolve(bh.H, nvals, krylovdim=nvals, :SR)
    # spectrum[:, iU] = vals[1:nvals]
    spectrum[:, iU] = eigvals(Symmetric(Matrix(bh.H)))
end

gr()
plotlyjs()
spectrum .%= ω
spectrum[spectrum .< 0] .+= ω
scatter(Us, spectrum', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native);
scatter!(Us, spectrum' .- ω, xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native);
hline!([-2J * besselj0(f) *  nbozons * cos(2pi/ncells * i) for i in 1:ncells])
title!("order=$(bh.order)")
savefig("order=2-zoom.png")
yaxis!((-2, 2))
yaxis!((-10.5, 10.5))

# Exact quasienergy spectrum
nbozons = 5; ncells = 5
J = 1 # setting to 1 so that `U` is measured in units of `J`
ω = 20
U = 1
f = 2
bh = BoseHamiltonian(J, U, f, ω, ncells, nbozons, isperiodic=false, type=:smallU)

Us = range(0, 1, 2)
ε = quasienergy(bh, Us)
minimum(ε[:, 1])

gr()
scatter(Us, ε', xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"F/\omega=%$f"*", exact", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native)
scatter!(Us, ε' .+ 20, xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"F/\omega=%$(F/ω)", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native)
scatter!(Us, ε' .- 20, xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"F/\omega=%$(F/ω)", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native)
ylims!(-20, 20)
title!(L"F/\omega=%$(F/ω)"*", exact")
savefig("exact-zoom2.png")
ylims!(-2, 2)
vline!([40/3], c=:white)
plot!(minorgird=true, minorticks=5, minorgridalpha=1)

using DelimitedFiles
ε = readdlm("spectrum_F10.txt")

open("f2_U12-15.txt", "w") do io
    writedlm(io, ε)
end

# degenerate theory

# the degenerate subspace is composed of the ground state with energy Eₗ and excited states Eₕ = Eₗ + Mω
# We are interested in trasition from Eₗ = 0 to Eₕ = U*3(3-1)/2 = 3U; so Mω = 3U. We know that there is no contribution
# from M = 1, so we take M = 2.
nbozons = 5; ncells = 5
nstates = binomial(nbozons+ncells-1, nbozons)
J = 1 # setting to 1 so that `U` is measured in units of `J`
f = 2
ω = 20

M = 2
U₀ = ω * M/3
Eₗ = 0
Eₕ = 3U₀

# construct a "blank" BH and calculate the zeroth-order spectrum
bh = BoseHamiltonian(J, U₀, f, ω, ncells, nbozons, isperiodic=true, type=:none, order=1)
E₀ = zeros(nstates)
for (index, state) in enumerate(bh.basis_states)
    for i = 1:bh.ncells # iterate over the terms of the Hamiltonian
        if (state[i] > 1)
            E₀[index] += bh.U/2 * state[i] * (state[i] - 1)
        end
    end
end
scatter(E₀)

# space_of_state[i] stores the subspace number of i'th state, with -1 indicating nondegenerate space
space_of_state = map(E₀) do E
    M = (E - Eₗ) / ω
    M_int = round(Int, M)
    if isapprox(M, M_int, atol=0.01)
        return M_int
    else
        return -1
    end
end

scatter!(space_of_state)
plot!(legend=false)
bh = BoseHamiltonian(J, U₀, f, ω, ncells, nbozons, space_of_state, isperiodic=true, type=:largeU, order=2)

M = copy(bh.H)
M[diagind(M)] .= 0
heatmap(M, yaxis=:flip, c=:coolwarm)
heatmap(bh.H, yaxis=:flip, c=:coolwarm)

nU = 1001
spectrum = Matrix{Float64}(undef, size(bh.H, 1), nU)
Us = range(U₀-1, U₀+2, nU)
@time for (iU, U) in enumerate(Us)
    bh = BoseHamiltonian(J, U, f, ω, ncells, nbozons, space_of_state, isperiodic=true, type=:largeU, order=2)
    spectrum[:, iU] = eigvals(Symmetric(Matrix(bh.H)))
end

gr()
plotlyjs()
scatter(Us, spectrum', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native);
ylims!(-2, 2);
xlims!(12, 15);
vline!([U₀], c=:white)