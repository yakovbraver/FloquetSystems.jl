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

"Return `n_isol` levels of `ε` using the permutation matrix `sp`."
function isolate(ε, sp; n_isol)
    E = Matrix{eltype(ε)}(undef, n_isol, size(ε, 2))
    for i in eachindex(Us)
        E[:, i] = ε[Int.(sp[1:n_isol, i]), i]
    end
    return E
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
lattice = Lattice(;dims=(1, 5), isperiodic=true)
J = 1.0f0 # setting to 1 so that `U` is measured in units of `J`
ω = 20
U = 1
f = 2
bh = BoseHamiltonian(lattice, J, U, f, ω)

Us = range(13, 13.6, 8)
Us = range(12, 15, 300)
ε, sp = quasienergy(bh, Us, nthreads=8);
c, p = quasienergy(bh, Us, nthreads=8)
@code_warntype FloquetSystems.schrodinger!(c, c, p, 1) 

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
ε_old = readdlm("calcs/2x4/f5_w10_U6-8_2x4-exact.txt")
sp = readdlm("calcs/2x4/f5_w10_U6-8_2x4-exact-perm.txt")
ε = ε_old[2:end, :]
Us = ε_old[1, :]
# isolating levels of interest
n_isol = 200
E = isolate(ε, sp; n_isol)

fig = scatter(Us, E', xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"F/\omega=%$f, \omega=%$ω"*", exact", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
scatter!(Us, E' .+ ω, markersize=0.5, markerstrokewidth=0, c=colour);
scatter!(Us, E' .- ω, markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native);
vline!([U₀], c=:red);
title!(fig, L"F/\omega=%$f, \omega=%$ω"*", $(lattice.dims[1])x$(lattice.dims[2]) lattice, exact");
ylims!(-3, 3)
xlims!(12.33, 14.33);
xlims!(1.5, 1.8)
ylims!(-ω/2, ω/2)
ylims!(-1.3, -0.8)

savefig("calcs/2x4/f$(f)_w$(ω)_$(numerator(r))d$(denominator(r))_$(lattice.dims[1])x$(lattice.dims[2])-exact-$n_isol.png")

open("calcs/f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-exact.txt", "w") do io
    writedlm(io, vcat(Us', ε))
end

# degenerate theory

J = 1.0f0 # setting to 1 so that `U` is measured in units of `J`
f = 5
ω = 10

r = 1//6

ωₗ = -ω/2
U₀ = float(ω) * r

lattice = Lattice(;dims=(2, 4), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U₀, f, ω, r, ωₗ, type=:dpt, order=2);
@time update_params!(bh; f)
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

nU = 300
Us = range(U₀-1, U₀+1, nU)
Us = range(1.57, 1.71, nU)
Us = range(12, 15, nU)
Us = range(0, ω, nU)

spectrum, sp = dpt(bh, r, ωₗ, Us; order=2, sort=false);

# remove Inf's
mask = spectrum[1, :] .!= Inf
spectrum = spectrum[:, mask]
sp = sp[:, mask]
Us = Us[mask]

spec = isolate(spectrum, sp, n_isol=1)

gr()
spectrum .%= ω
spectrum[spectrum .< 0] .+= ω 
figD2 = scatter(Us, spectrum', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
scatter!(Us, (spectrum .- ω)', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
vline!([U₀], c=:red);
ylims!(0, 3)
title!("order = 3")
ylims!(-5, 5)
ylims!(figD2, (-0.6, -0.3));
ylims!(figD2, (-ω/2, ω/2))
xlims!(12.33, 14.33);
plot(fig, figD2)
plot!(xlims=(U₀-1, U₀+1), ylims=(-2, 2), title="isolated")

savefig("f$(f)_w$(ω)_$(numerator(r))d$(denominator(r))_$(lattice.dims[1])x$(lattice.dims[2])-dpt3-$n_isol.png")

using DelimitedFiles
f = 2
ω = 20
r = 1//6
spectrum_file = readdlm("calcs/2x4/f$(f)_w$(ω)_U0-30_2x4-dpt3.txt")
sp = readdlm("calcs/2x4/f$(f)_w$(ω)_U0-30_2x4-dpt3-perm.txt")

Us = spectrum_file[1, :]
spectrum = spectrum_file[2:end, :]

# remove Inf's
mask = spectrum[1, :] .!= Inf
spectrum = spectrum[:, mask]
sp = sp[:, mask]
Us = Us[mask]

n_isol = 200
spec = isolate(spectrum, sp; n_isol)

figD = scatter(Us, spec', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native, widen=false);
scatter!(Us, spec' .+ ω, markersize=0.5, markerstrokewidth=0, c=colour);
scatter!(Us, spec' .- ω, markersize=0.5, markerstrokewidth=0, c=colour, legend=false, ticks=:native);
title!(figD, L"F/\omega=%$f, \omega=%$ω"*", $(lattice.dims[1])x$(lattice.dims[2]) lattice, order = 3, rz");
xlims!(9, 11);
ylims!(-ω/2, ω/2)
ylims!(-1.3, -0.8)

savefig("calcs/2x4/f$(f)_w$(ω)_$(numerator(r))d$(denominator(r))_$(lattice.dims[1])x$(lattice.dims[2])-rz-dpt3-$n_isol.png")

open("calcs/f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-dpt3.txt", "w") do io
    writedlm(io, vcat(Us', spectrum))
end

open("calcs/f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-dpt3-perm.txt", "w") do io
    writedlm(io, sp)
end

####### Levels in the Floquet zone
gr()
f = 5
ω = 10
lattice = Lattice(;dims=(2, 4), isperiodic=true)
r = 1//6
U₀ = float(ω) * r
ωₗ = -ω/2
ωₗ = 0
bh = BoseHamiltonian(lattice, J, U₀, f, ω, r, ωₗ; type=:dpt, order=2);
e = sort(bh.E₀*U₀)
e = bh.E₀*U₀
fig0 = scatter(e, markersize=2, markerstrokewidth=0, minorgrid=true, ylabel=L"\varepsilon/J", xlabel="level number", legend=false)
scatter!(e .+ ω, markersize=2, markerstrokewidth=0, minorgrid=true, ylabel=L"\varepsilon/J", xlabel="level number");
for i in 1:20
    scatter!(fig0, e .- i*ω, markersize=2, markerstrokewidth=0, legend=false);
end
title!(L"\omega=%$(ω),\ U = 2\omega/3"*", $(lattice.dims[1])x$(lattice.dims[2]) lattice")
hline!([-ω/2, ω/2], c=:white);
ylims!(-1.5ω, 1.5ω)
savefig("levels_w$(ω)_$(numerator(r))d$(denominator(r))_$(lattice.dims[1])x$(lattice.dims[2]).pdf")

crit(bh.H)

M = copy(bh.H)
M[diagind(M)] .= 0
heatmap(M, yaxis=:flip, c=:viridis)
eigvals(bh.H)

findall(isnan, bh.H)
count(isnan, bh.H)

scatter!(abs.(bh.H[1, 2:end]), markersize=2, markerstrokewidth=0, label="$ωₗ")
scatter!(1:length(lattice.basis_states), i -> bh.space_of_state[i][2], markersize=1, markerstrokewidth=0, legend=false)
bh.space_of_state[6205][2]
bh.space_of_state[6383][2]

function crit(h)
    cr = 0.0
    for c in 1:size(h, 1) - 1
        for r in c+1:size(h, 1)
            d = abs(h[r, r] - h[c, c])
            if d > 1e-7
                cr += abs(h[r, c]) / d
            end
        end
    end
    cr
end



ε = Vector{Float64}(undef, length(bh.E₀)) # energies (including 𝑈 multiplier) reduced to first Floquet zone
for i in eachindex(bh.E₀)
    ε[i] = bh.E₀[i]*U₀ - bh.space_of_state[i][2]*ω
end
fig0 = scatter(ε, markersize=1, markerstrokewidth=0, minorgrid=true, ylabel=L"\varepsilon/J", xlabel="level number", legend=false)
ylims!(-5, 5)