using FloquetSystems

using LinearAlgebra, BenchmarkTools, SpecialFunctions
using Plots, LaTeXStrings
plotlyjs()
theme(:dark, size=(800, 600))
colour = 1
# theme(:default, size=(800, 600))
# colour = :black

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

"Return `n_isol` levels of `ε` using the permutation matrix `sp`."
function isolate(ε, sp; n_isol)
    E = Matrix{eltype(ε)}(undef, n_isol, size(ε, 2))
    for i in axes(E, 2)
        E[:, i] = ε[Int.(sp[1:n_isol, i]), i]
    end
    return E
end

################ Basic high-frequency theory

nbozons = 5; ncells = 5
lattice = Lattice(;dims=(1, ncells), isperiodic=true)
nstates = length(lattice.basis_states)
J = 1.0f0 # setting to 1 so that `U` is measured in units of `J`
U = 1
f = 2
ω = 20
-2J * besselj0(f) * nbozons * cos(π/(ncells+1)) # non-periodic: exact ground state energy of 𝑊¹ at 𝑈 = 0; spectrum of 𝑊¹ at 𝑈 = 0 is NOT the exact quasienergy spectrum
-2J * besselj0(f) * nbozons # periodic: exact ground state energy of 𝑊¹ at 𝑈 = 0; spectrum of 𝑊¹ at 𝑈 = 0 IS the exact quasienergy spectrum
bh = BoseHamiltonian(lattice, J, U, f, ω, order=1);
vals, vecs = eigen(Symmetric(Matrix(bh.H)))
scatter(vals, xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"f=%$f", markersize=2, markerstrokewidth=0, c=2, legend=false, ticks=:native)

M = copy(bh.H)
M[diagind(M)] .= 0
heatmap(M, yaxis=:flip, c=:coolwarm)

nU = 500
spectrum = Matrix{typeof(J)}(undef, length(lattice.basis_states), nU)
Us = range(0, ω, nU)
bh = BoseHamiltonian(lattice, J, U, f, ω, order=2, type=:basic)
@time for (iU, U) in enumerate(Us)
    update_params!(bh; U)
    spectrum[:, iU] = eigvals(Symmetric(bh.H))
end

gr()
spectrum .%= ω
spectrum[spectrum .< 0] .+= ω
scatter(Us, spectrum', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native);
scatter!(Us, spectrum' .- ω, xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native)
title!("order=$(bh.order)")

################ Exact theory

lattice = Lattice(;dims=(1, 6), isperiodic=true)
J = 1.0f0 # setting to 1 so that `U` is measured in units of `J`
f = 2
ω = 20.0
U = 1
bh = BoseHamiltonian(lattice, J, U, f, ω)

Us = range(12, 15, 320)
GC.gc()
ε, sp = quasienergy(bh, Us, sort=false, showprogress=true);

gr()
fig = scatter(Us, ε', xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"F/\omega=%$f, \omega=%$ω"*", $(lattice.dims[1])x$(lattice.dims[2]) lattice, exact", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
ylims!(-ω/2, ω/2)
title!(fig, L"F/\omega=%$f, \omega=%$ω"*", $(lattice.dims[1])x$(lattice.dims[2]) lattice, exact")
savefig("f$(f)_w$(ω)_2d3_$(lattice.dims[1])x$(lattice.dims[2])-exact.png")
ylims!(0, 2)
xlims!(Us[1], Us[end])

open("calcs/2D/f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-exact.txt", "w") do io
    writedlm(io, vcat(Us', ε))
end

open("calcs/2D/f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-exact-perm.txt", "w") do io
    writedlm(io, sp)
end

using DelimitedFiles
ε_old = readdlm("calcs/2x3/f5_w10_U6-8_2x3-exact.txt")
sp = readdlm("calcs/2x3/f5_w10_U6-8_2x3-exact-perm.txt")
ε = ε_old[2:end, :]
Us = ε_old[1, :]
# isolating levels of interest
n_isol = 10
E = isolate(ε, sp; n_isol)

fig = scatter(Us, E', xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"F/\omega=%$f, \omega=%$ω"*", $(lattice.dims[1])x$(lattice.dims[2]) lattice, exact", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
scatter!(Us, E' .+ ω, markersize=0.5, markerstrokewidth=0, c=colour);
scatter!(Us, E' .- ω, markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native);
vline!([U₀], c=:red);
ylims!(-3, 3)
xlims!(1.5, 1.8)

savefig("calcs/2x4/f$(f)_w$(ω)_$(numerator(r))d$(denominator(r))_$(lattice.dims[1])x$(lattice.dims[2])-exact-$n_isol.png")

open("calcs/f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-exact.txt", "w") do io
    writedlm(io, vcat(Us', ε))
end

################ Degenerate theory

J = 1.0f0 # setting to 1 so that `U` is measured in units of `J`
f = 2
ω = 20

r = 2//3

U₀ = float(ω) * r

lattice = Lattice(;dims=(1, 5), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U₀, f, ω; r, type=:dpt, order=2);
scatter(1:length(lattice.basis_states), i -> bh.space_of_state[i][1], markersize=1, markerstrokewidth=0, legend=false)
scatter(abs.(bh.H[:, 1]), markersize=1, markerstrokewidth=0, legend=false)
scatter(bh.E₀, markersize=1, markerstrokewidth=0, legend=false)


M = copy(bh.H);
M[diagind(M)] .= 0;
f2 = heatmap(abs.(M), yaxis=:flip, c=:viridis)
heatmap(abs.(bh.H), yaxis=:flip, c=:viridis)
plot(bh.H[diagind(bh.H)])

nU = 320
Us = range(U₀-1, U₀+1, nU)
Us = range(12, 15, nU)

lattice = Lattice(;dims=(1, 6), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U₀, f, ω; r, type=:edpt, order=2);
spectrum, sp = edpt(bh, Us; sort=false, showprogress=true);
spectrum, sp = dpt(bh, Us; sort=false, showprogress=true);

# remove Inf's
mask = @. !isnan(spectrum[1, :])
spectrum = spectrum[:, mask]
sp = sp[:, mask]
Us = Us[mask]

spec = isolate(spectrum, sp, n_isol=10) # use if calculated using sorting

gr()
fig = scatter(Us, spec', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
scatter!(Us, (spec .+ ω)', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
scatter!(Us, (spec .- ω)', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
ylims!(fig, (-ω/2, ω/2))
vline!([U₀], c=:white);
ylims!(0, 2)
title!("order = 3")

savefig("f$(f)_w$(ω)_$(numerator(r))d$(denominator(r))_$(lattice.dims[1])x$(lattice.dims[2])-edpt3-$n_isol.png")

# plotting from files

using DelimitedFiles
f = 2.0
ω = 30.0
r = 2//3
lattice = Lattice(;dims=(2, 4), isperiodic=true)
spectrum_file = readdlm("f$(f)_w$(ω)_U18.0-22.0_1x6-edpt3.txt")
sp = readdlm("f$(f)_w$(ω)_U18.0-22.0_1x6-edpt3-perm.txt")

Us = spectrum_file[1, :]
spectrum = spectrum_file[2:end, :]

n_isol = 10
spec = isolate(spectrum, sp; n_isol)

fig = scatter(Us, spec' ./ ω, xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
scatter!(Us, spec' ./ ω .+ 1, markersize=0.5, markerstrokewidth=0, c=colour);
scatter!(Us, spec' ./ ω .- 1, markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native);
title!(fig, L"F/\omega=%$f, \omega=%$ω"*", $(lattice.dims[1])x$(lattice.dims[2]) lattice, order = 3");
ylims!(-1/2, 1/2)
ylims!(1, 1.05)

savefig("calcs/2x4/f$(f)_w$(ω)_$(numerator(r))d$(denominator(r))_$(lattice.dims[1])x$(lattice.dims[2])-exact-$n_isol.png")

open("calcs/2x4/f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-dpt3.txt", "w") do io
    writedlm(io, vcat(Us', spectrum))
end

open("calcs/2x4/f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-dpt3-perm.txt", "w") do io
    writedlm(io, sp)
end

################ Analyse residual couplings
J = 1.0f0
f = 2
ω = 20
r = 2//3
U₀ = float(ω) * r
lattice = Lattice(;dims=(1, 8), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U₀, f, ω; r, type=:edpt, order=1);

print_state(lattice, 545)

scatter(bh.ε₀, markerstrokewidth=0, markersize=1, legend=false)
scatter(bh.E₀, markerstrokewidth=0, markersize=1, legend=false)

W = residuals!(bh)
m, i = findmax(bh.H)
W[i]

ra = [1]
ra = [478:1037; 1108:1275]
ra = [3236:3655; 3824:4103]
ra = 5112:5475

maximum(bh.H[:, ra])
ar = argmax(bh.H[:, ra])
W[ar[1], ra[ar[2]]]

theme(:dark, size=(1600, 600))
f1 = heatmap(bh.H, yaxis=:flip, c=:viridis);
f2 = heatmap(W, yaxis=:flip, c=:viridis);
plot(f1, f2, link=:both)