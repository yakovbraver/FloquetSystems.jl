using FloquetSystems

using LinearAlgebra, BenchmarkTools, SpecialFunctions
using Plots, LaTeXStrings
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
lattice = Lattice(;dims=(1, ncells), isperiodic=true)
lattice = Lattice(;dims=(3, 3), isperiodic=true)
nstates = length(lattice.basis_states)
J = 1 # setting to 1 so that `U` is measured in units of `J`
U = 1#sqrt(1.01)
f = 2
ω = 20
-2J * besselj0(f) * nbozons * cos(π/(ncells+1)) # non-periodic: exact ground state energy of 𝑊¹ at 𝑈 = 0; spectrum of 𝑊¹ at 𝑈 = 0 is NOT the exact quasienergy spectrum
-2J * besselj0(f) * nbozons # periodic: exact ground state energy of 𝑊¹ at 𝑈 = 0; spectrum of 𝑊¹ at 𝑈 = 0 IS the exact quasienergy spectrum
@time bh = BoseHamiltonian(lattice, J, U, f, ω, order=1);
# @time vals, vecs, info = eigsolve(bh.H, 100, krylovdim=126, :SR)
vals, vecs = eigen(Symmetric(Matrix(bh.H)))
scatter(vals, xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"f=%$f", markersize=2, markerstrokewidth=0, c=2, legend=false, ticks=:native)
scatter!([-2J * cos(π*i/(ncells+1)) for i in 1:ncells], xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"f=%$f", markersize=2, markerstrokewidth=0, c=2, legend=false, ticks=:native)

M = copy(bh.H)
M[diagind(M)] .= 0
heatmap(M, yaxis=:flip, c=:coolwarm)
heatmap(bh.H, yaxis=:flip, c=:coolwarm)

nU = 1000
spectrum = Matrix{Float64}(undef, length(lattice.basis_states), nU)
Us = range(0, 2ω, nU) #.* √1.01
bh = BoseHamiltonian(lattice, J, U, f, ω, order=2, type=:diverging)
@time for (iU, U) in enumerate(Us)
    update_params!(bh; U)
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
savefig("test.png")
yaxis!((-2, 2))
yaxis!((-10.5, 10.5))

# Exact quasienergy spectrum
lattice = Lattice(;dims=(1, 5), isperiodic=true)
lattice = Lattice(;dims=(2, 3), isperiodic=true)
J = 1 # setting to 1 so that `U` is measured in units of `J`
ω = 20
U = 1
f = 2
bh = BoseHamiltonian(lattice, J, U, f, ω)

Us = range(0, ω, 80)
Us = range(6, 8, 300)
@time ε = quasienergy_dense(bh, Us, parallelise=true);
# e = copy(ε)
sum(abs.(e .- ε))

gr()
fig = scatter(Us, ε', xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"F/\omega=%$f"*", exact", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, widen=false);
scatter!(Us, ε' .+ ω, markersize=0.5, markerstrokewidth=0, c=1);
scatter!(Us, ε' .- ω, markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native);
ylims!(-ω/2, ω/2)
ylims!(-2, 2)
title!(fig, L"F/\omega=%$f, \omega=%$ω"*", $(lattice.dims[1])x$(lattice.dims[2]) lattice, exact")
savefig("f$(f)_w$(ω)_2d3_$(lattice.dims[1])x$(lattice.dims[2])-exact.png")
ylims!(1, 2)
xlims!(Us[1], Us[end])
for k in [1, 2, 3, 4, 6, 7, 10, 15]
    plot!(fig, [0, 10], [0, 10k], c=:white)
end
fig
vline!([10/6], c=:white)

u = 100
fig1 = scatter(sort(ε[:, u]), xlabel=L"U/J", ylabel=L"\varepsilon/J", title="exact", markersize=1, markerstrokewidth=0, c=1, legend=false, ticks=:native)
scatter!(spectrum[:, u], xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=1, markerstrokewidth=0, c=3, legend=false, ticks=:native)
sp2 = copy(spectrum)

using DelimitedFiles
ε_old = readdlm("f2_w20_U12.3-14.3_2x3-exact_6min.txt")
fig = scatter(ε_old[1, :], ε_old[2:end, :]', xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"F/\omega=%$f, \omega=%$ω"*", exact", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, widen=false)
scatter!(ε_old[1, :], ε_old[2:end, :]' .+ ω, markersize=0.5, markerstrokewidth=0, c=1);
scatter!(ε_old[1, :], ε_old[2:end, :]' .- ω, markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native);
ylims!(-0.6, -0.3)
ylims!(0, 3)
xlims!(1.57, 1.7)
ylims!(-ω/2, ω/2)
title!(fig, L"F/\omega=%$f, \omega=%$ω"*", $(lattice.dims[1])x$(lattice.dims[2]) lattice, exact")
vline!([40/3], c=:white)
savefig("f$(f)_w$(ω)_2d3_$(lattice.dims[1])x$(lattice.dims[2])-exact.png")

open("f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-exact.txt", "w") do io
    writedlm(io, vcat(Us', ε))
end

# degenerate theory

J = 1 # setting to 1 so that `U` is measured in units of `J`
f = 5
ω = 10

r = 2//3

U₀ = float(ω) * r

lattice = Lattice(;dims=(1, 5), isperiodic=true)
lattice = Lattice(;dims=(2, 3), isperiodic=true)
@time bh = BoseHamiltonian(lattice, J, U₀, f, ω, r, type=:dpt, order=1);
scatter(bh.H[1,:], markersize=1, markerstrokewidth=0)
bh.H[1, 1]

scatter(bh.E₀, markersize=0.5, markerstrokewidth=0)
range6U = (findfirst(==(6), bh.E₀), findlast(==(6), bh.E₀)) # range of states of energy 6U
range6U = (findfirst(==(4), bh.E₀), findlast(==(4), bh.E₀)) # range of states of energy 6U

issymmetric(bh.H)
M = copy(bh.H);
M[diagind(M)] .= 0
sum(abs.(M - M'))

M = copy(bh.H);
M[diagind(M)] .= 0;
f2 = heatmap(abs.(M), yaxis=:flip, c=:viridis)
heatmap(abs.(bh.H), yaxis=:flip, c=:viridis)
plot(bh.H[diagind(bh.H)])

nU = 300
Us = range(U₀-1, U₀+1, nU)
Us = range(6.01, 7.99, nU)

spectrum = scan_U(bh, r, Us; type=:dpt_quick, order=3)

gr()
plotlyjs()
spectrum .%= ω
spectrum[spectrum .< 0] .+= ω
figD2 = scatter(Us, spectrum', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, widen=false);
scatter!(Us, (spectrum .- ω)', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, widen=false);
ylims!(-5, 5)
ylims!(0, 3)
ylims!(figD2, (-0.6, -0.3))
ylims!(figD2, (-ω/2, ω/2))
xlims!(6, 8)
plot(fig, figD2)
title!("order = 2")
plot!(xlims=(U₀-1, U₀+1), ylims=(-2, 2), title="isolated")
vline!([U₀], c=:white)

savefig("f$(f)_w$(ω)_$(numerator(r))d$(denominator(r))_$(lattice.dims[1])x$(lattice.dims[2])-dpt2.png")

using DelimitedFiles
spectrum_file = readdlm("f5_w10_U6-8_2x3-dpt3_10min.txt")
figF = scatter(spectrum_file[1, :], spectrum_file[2:end, :]', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native)
scatter!(spectrum_file[1, :], spectrum_file[2:end, :]' .+ ω, markersize=0.5, markerstrokewidth=0, c=1);
scatter!(spectrum_file[1, :], spectrum_file[2:end, :]' .- ω, markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native);
plot(figD2, figF, link=:y)
ylims!(-0.6, -0.4)

open("f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-dpt3.txt", "w") do io
    writedlm(io, vcat(Us', spectrum))
end