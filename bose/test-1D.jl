includet("hamiltonian-1D.jl")

using SparseArrays, LinearAlgebra, FLoops
using Plots, LaTeXStrings
using BenchmarkTools, SpecialFunctions
using ProgressMeter
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
lattice = Lattice(;dims=(1, ncells), nbozons, isperiodic=true)
lattice = Lattice(;dims=(3, 3), nbozons=9, isperiodic=true)
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
bh = BoseHamiltonian(lattice, J, U, f, ω, order=2)
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
lattice = Lattice(;dims=(1, 5), nbozons=5, isperiodic=true)
lattice = Lattice(;dims=(2, 3), nbozons=6, isperiodic=true)
J = 1 # setting to 1 so that `U` is measured in units of `J`
ω = 20
U = 1
f = 2
bh = BoseHamiltonian(lattice, J, U, f, ω)

Us = range(0, ω, 300) # 6 bozons, nU = 300 => 9:12
Us = range(19, 21, 2) # 6 bozons, nU = 300 => 9:12
@time ε = quasienergy_dense(bh, Us, parallelise=true);
# e = copy(ε)
sum(abs.(e .- ε))

gr()
fig = scatter(Us, ε', xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"F/\omega=%$f"*", exact", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, widen=false);
scatter!(Us, ε' .+ ω, markersize=0.5, markerstrokewidth=0, c=1);
scatter!(Us, ε' .- ω, markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native);
ylims!(-ω/2, ω/2)
ylims!(-2, 2)
title!(fig, L"F/\omega=%$f, \omega=%$ω"*", 1x5 lattice, exact")
savefig("f$(f)_w$(ω)_2d3_1x5-exact.png")
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
title!(fig, L"F/\omega=%$f, \omega=%$ω"*", 2x3 lattice, exact")
vline!([40/3], c=:white)
savefig("line.png")
savefig("f$(f)_w$(ω)_2d3_2x3-exact.png")

open("f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_2x3-exact.txt", "w") do io
    writedlm(io, vcat(Us', ε))
end

# degenerate theory

J = 1 # setting to 1 so that `U` is measured in units of `J`
f = 5
ω = 10

r = 2//3
r = 1//6
r = 1//1

U₀ = float(ω) * r

ε = Vector{Float64}(undef, length(bh.E₀)) # energies (including 𝑈 multiplier) reduced to first Floquet zone
for i in eachindex(bh.E₀)
    ε[i] = bh.E₀[i]*1.05U₀ - bh.space_of_state[i][2]*ω
end
scatter(ε ./ U₀, markersize=1, markerstrokewidth=0, legend=false)
scatter!(1:length(bh.E₀), i -> bh.space_of_state[i][1], markersize=1, markerstrokewidth=0, legend=false)

lattice = Lattice(;dims=(1, 5), nbozons=5, isperiodic=true)
lattice = Lattice(;dims=(2, 3), nbozons=6, isperiodic=true)
@time bh = BoseHamiltonian(lattice, J, U₀-1, f, ω, r, type=:dpt_quick, order=3);
scatter(bh.H[1,:], markersize=1, markerstrokewidth=0)
bh.H[2, 2]

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

A = 0
As = findall(s -> s[1] == A, bh.space_of_state) # `As`` stores numbers of state that belong to space `A`
h = zeros(length(As), length(As)) # reduced matrix of the subspace of interest
nU = 300
spectrum = Matrix{Float64}(undef, length(As), nU)
Us = range(1.57, 1.7, nU)
Us = range(6.01, 7.99, nU)
# scatter(bh.H[diagind(bh.H)][As], markersize=0.5, markerstrokewidth=0)

nU = 300
spectrum = Matrix{Float64}(undef, size(bh.H, 1), nU)
Us = range(6.01, 7.99, nU)
Us = range(U₀-1, U₀+1, nU)

function scan_U!(spectrum, lattice, Us, As=Int[]; type, order)
    progbar = Progress(length(Us))

    Threads.@threads for iU in eachindex(Us)
        bh = BoseHamiltonian(lattice, J, Us[iU], f, ω, r; type, order);
        if type == :dpt
            spectrum[:, iU] = eigvals(Symmetric(bh.H))
        elseif type == :dpt_quick
            length(As) == 0 && (@error "`As` not passed, terminating."; return)
            h = zeros(length(As), length(As)) # reduced matrix of the subspace of interest
            for i in eachindex(As), j in i:length(As)
                h[j, i] = bh.H[As[j], As[i]]
            end
            spectrum[:, iU] = eigvals(Symmetric(h, :L))
        end
        next!(progbar)
    end
    finish!(progbar)
end

BLAS.set_num_threads(1)
scan_U!(spectrum, lattice, Us, As; type=:dpt, order=3)

gr()
plotlyjs()
spectrum .%= ω
spectrum[spectrum .< 0] .+= ω
figD2 = scatter(Us, spectrum', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, widen=false);
scatter!(Us, (spectrum .- ω)', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, widen=false);
ylims!(-2, 2)
ylims!(0, 2)
ylims!(figD2, (-0.6, -0.3))
ylims!(figD2, (-ω/2, ω/2))
xlims!(6, 8)
plot(fig, figD2)
title!("order = 2")
plot!(xlims=(U₀-1, U₀+1), ylims=(-2, 2), title="isolated")
vline!([U₀], c=:white)
savefig("f$(f)_w$(ω)_1d6_2x3-dpt2_quick.png")

using DelimitedFiles
Us = range(12, 15, 1000)
ε = readdlm("f2_U12-15.txt")
# Us = range(0, 45, 1000)
spectrum_file = readdlm("f5_w10_U6-8_2x3-dpt3_10min.txt")
figF = scatter(spectrum_file[1, :], spectrum_file[2:end, :]', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native)
scatter!(spectrum_file[1, :], spectrum_file[2:end, :]' .+ ω, markersize=0.5, markerstrokewidth=0, c=1);
scatter!(spectrum_file[1, :], spectrum_file[2:end, :]' .- ω, markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native);
plot(figD2, figF, link=:y)
ylims!(-0.6, -0.4)


fig = scatter(Us, ε', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, title="exact", widen=false)
plot!(fig, ylims=(-2, 2), xlims=(U₀-1, U₀+1))

theme(:dark, size=(1200, 600))
plot(rand(3), rand(3))
ylims!(fig, (-ω/2, ω/2))
xlims!(figD3, (6, 8))
plot(fig, figD3, figD2, layout=(1, 3), link=:y)
savefig("1x5-3rd-order.png")

open("f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_2x3-dpt3_quick.txt", "w") do io
    writedlm(io, vcat(Us', spectrum))
end