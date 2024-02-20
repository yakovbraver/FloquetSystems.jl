using FloquetSystems

using LinearAlgebra, BenchmarkTools, SpecialFunctions
using Plots, LaTeXStrings
plotlyjs()
theme(:dark, size=(800, 600))
theme(:default, size=(800, 600))
colour = :black

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
nstates = length(lattice.basis_states)
J = 1.0f0 # setting to 1 so that `U` is measured in units of `J`
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

nU = 500
spectrum = Matrix{typeof(J)}(undef, length(lattice.basis_states), nU)
Us = range(0, ω, nU) #.* √1.01
bh = BoseHamiltonian(lattice, J, U, f, ω, order=2, type=:basic)
@time for (iU, U) in enumerate(Us)
    update_params!(bh; U)
    spectrum[:, iU] = eigvals(Symmetric(bh.H))
end

gr()
spectrum .%= ω
spectrum[spectrum .< 0] .+= ω
scatter(Us, spectrum', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native);
scatter!(Us, spectrum' .- ω, xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native);
hline!([-2J * besselj0(f) *  nbozons * cos(2pi/ncells * i) for i in 1:ncells])
title!("order=$(bh.order)")
savefig("test.png")
yaxis!((-2, 2))
yaxis!((-10.5, 10.5))

# Exact quasienergy spectrum
lattice = Lattice(;dims=(2, 4), isperiodic=true)
J = 1.0f0 # setting to 1 so that `U` is measured in units of `J`
ω = 20
U = 1
f = 2
bh = BoseHamiltonian(lattice, J, U, f, ω)

Us = range(13, 13.6, 8)
Us = range(12, 15, 24)
@time ε, sp = quasienergy(bh, Us, nthreads=1);

# isolating levels of interest
sp = readdlm("f2_w20_U12-15_2x4-exact-perm.txt")
n_isol = 1
E = Matrix{eltype(ε)}(undef, n_isol, length(Us))
for i in eachindex(Us)
    E[:, i] = ε[Int.(sp[1:n_isol, i]), i]
end

gr()
fig = scatter(Us, ε', xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"F/\omega=%$f, \omega=%$ω"*", $(lattice.dims[1])x$(lattice.dims[2]) lattice, exact", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
scatter!(Us, ε' .+ ω, markersize=0.5, markerstrokewidth=0, c=colour);
scatter!(Us, ε' .- ω, markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native);
ylims!(-ω/2, ω/2)
ylims!(-2, 2)
vline!([U₀], c=:red)
title!(fig, L"F/\omega=%$f, \omega=%$ω"*", $(lattice.dims[1])x$(lattice.dims[2]) lattice, exact")
savefig("f$(f)_w$(ω)_2d3_$(lattice.dims[1])x$(lattice.dims[2])-exact-$(n_isol).png")
ylims!(1, 2)
xlims!(Us[1], Us[end])
for k in [1, 2, 3, 4, 6, 7, 10, 15]
    plot!(fig, [0, 10], [0, 10k], c=:white)
end
fig

u = 100
fig1 = scatter(sort(ε[:, u]), xlabel=L"U/J", ylabel=L"\varepsilon/J", title="exact", markersize=1, markerstrokewidth=0, c=colour, legend=false, ticks=:native)
scatter!(spectrum[:, u], xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=1, markerstrokewidth=0, c=3, legend=false, ticks=:native)
sp2 = copy(spectrum)

using DelimitedFiles
ε_old = readdlm("f2_w20_U12-15_2x4-exact.txt")
ε = ε_old[2:end, :]
Us = ε_old[1, :]
fig = scatter(Us, E', xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"F/\omega=%$f, \omega=%$ω"*", exact", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
scatter!(Us, E' .+ ω, markersize=0.5, markerstrokewidth=0, c=colour);
scatter!(Us, E' .- ω, markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native);
vline!([U₀], c=:red);
ylims!(-3, 3)
title!(fig, L"F/\omega=%$f, \omega=%$ω"*", $(lattice.dims[1])x$(lattice.dims[2]) lattice, exact");
xlims!(12.33, 14.33);
xlims!(6, 8)
ylims!(-ω/2, ω/2);

savefig("f$(f)_w$(ω)_2d3_$(lattice.dims[1])x$(lattice.dims[2])-exact.png")

open("calcs/f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-exact.txt", "w") do io
    writedlm(io, vcat(Us', ε))
end

# degenerate theory

J = 1.0f0 # setting to 1 so that `U` is measured in units of `J`
f = 2
ω = 20

r = 2//3

ωₗ = 0
U₀ = float(ω) * r

lattice = Lattice(;dims=(1, 7), isperiodic=true)
@time bh = BoseHamiltonian(lattice, J, U₀, f, ω, r, ωₗ, type=:dpt, order=3);
scatter!(1:length(lattice.basis_states), i -> bh.space_of_state[i][2], markersize=1, markerstrokewidth=0, legend=false)
scatter!(abs.(bh.H[1, :]), markersize=1, markerstrokewidth=0, legend=false)
scatter(diag(bh.H), markersize=1, markerstrokewidth=0, legend=false)
plot!(legend=false)
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

nU = 64
Us = range(U₀-1, U₀+1, nU)
Us = range(1.57, 1.71, nU)

spectrum = scan_U(bh, r, ωₗ, Us; type=:dpt_quick, order=3);

gr()
plotlyjs()
spectrum .%= ω
spectrum = map(x -> !ismissing(x) && x < 0 ? x + ω : x, spectrum)
figD2 = scatter(Us, spectrum', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
scatter!(Us, (spectrum .- ω)', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
vline!([U₀], c=:red);
ylims!(0, 3);
title!("order = 3")
ylims!(-5, 5)
ylims!(figD2, (-0.6, -0.3));
ylims!(figD2, (-ω/2, ω/2))
xlims!(12.33, 14.33);
plot(fig, figD2)
plot!(xlims=(U₀-1, U₀+1), ylims=(-2, 2), title="isolated")

savefig("f$(f)_w$(ω)_$(numerator(r))d$(denominator(r))_$(lattice.dims[1])x$(lattice.dims[2])-dpt2.png")

using DelimitedFiles
spectrum_file = readdlm("calcs/f2_w20_U12.3-14.3_2x3-dpt3-rezoned.txt")
figF = scatter(spectrum_file[1, :], spectrum_file[2:end, :]', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native);
scatter!(spectrum_file[1, :], spectrum_file[2:end, :]' .+ ω, markersize=0.5, markerstrokewidth=0, c=colour);
scatter!(spectrum_file[1, :], spectrum_file[2:end, :]' .- ω, markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native);
plot(figD2, figF, link=:y)
ylims!(0, 2)

open("calcs/f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-dpt3-rezoned.txt", "w") do io
    mask = spectrum[1, :] .!== missing
    writedlm(io, vcat(Us[mask]', spectrum[:, mask]))
end

####### Levels in the Floquet zone
gr()
r = 2//3
U₀ = float(ω) * r
bh = BoseHamiltonian(lattice, J, U₀, f, ω, r, type=:dpt, order=1);
fig0 = scatter(bh.E₀*U₀, markersize=2, markerstrokewidth=0, minorgrid=true, ylabel=L"\varepsilon/J", xlabel="level number");
scatter!(bh.E₀*U₀ .+ ω, markersize=2, markerstrokewidth=0, minorgrid=true, ylabel=L"\varepsilon/J", xlabel="level number");
for i in 1:20
    scatter!(fig0, bh.E₀*U₀ .- i*ω, markersize=2, markerstrokewidth=0, legend=false);
end
title!(L"\omega=%$(ω),\ U = 2\omega/3"*", $(lattice.dims[1])x$(lattice.dims[2]) lattice")
hline!([-ω/2, ω/2], c=:white)
ylims!(-1.5ω, 1.5ω)
savefig("levels_w$(ω)_$(numerator(r))d$(denominator(r))_$(lattice.dims[1])x$(lattice.dims[2]).pdf")