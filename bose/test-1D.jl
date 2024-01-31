includet("hamiltonian-1D.jl")

using SparseArrays, LinearAlgebra
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
@time bh = BoseHamiltonian(lattice, J, U, f, ω, type=:smallU, order=1);
# @time vals, vecs, info = eigsolve(bh.H, 100, krylovdim=126, :SR)
vals, vecs = eigen(Symmetric(Matrix(bh.H)))
scatter(vals, xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"f=%$f", markersize=2, markerstrokewidth=0, c=2, legend=false, ticks=:native)
scatter!([-2J * cos(π*i/(ncells+1)) for i in 1:ncells], xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"f=%$f", markersize=2, markerstrokewidth=0, c=2, legend=false, ticks=:native)

M = copy(bh.H)
M[diagind(M)] .= 0
heatmap(M, yaxis=:flip, c=:coolwarm)
heatmap(bh.H, yaxis=:flip, c=:coolwarm)

nU = 100
spectrum = Matrix{Float64}(undef, nstates, nU)
Us = range(0, 16, nU) #.* √1.01
@showprogress for (iU, U) in enumerate(Us)
    bh = BoseHamiltonian(lattice, J, U, f, ω, type=:smallU, order=2)
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
bh = BoseHamiltonian(lattice, J, U, f, ω, type=:smallU)

Us = range(12, 15, 300) # 5 bozons, nU = 300 => 9:12
ε = quasienergy_dense(bh, Us)
minimum(ε[:, 1])

gr()
fig = scatter(Us, ε', xlabel=L"U/J", ylabel=L"\varepsilon/J", title=L"F/\omega=%$f"*", exact", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native);
scatter!(Us, ε' .+ 20, markersize=0.5, markerstrokewidth=0, c=1);
scatter!(Us, ε' .- 20, markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native);
ylims!(-10, 10)
title!("2x3 lattice, exact")
savefig("2x3-lattice-exact.png")
ylims!(-0.5, 5)
vline!([40/3], c=:white)

u = 100
fig1 = scatter(sort(ε[:, u]), xlabel=L"U/J", ylabel=L"\varepsilon/J", title="exact", markersize=1, markerstrokewidth=0, c=1, legend=false, ticks=:native)
scatter!(spectrum[:, u], xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=1, markerstrokewidth=0, c=3, legend=false, ticks=:native)
sp2 = copy(spectrum)

using DelimitedFiles
ε_old = readdlm("f2_U19-21.txt")

# open("U13-2x3-exact.txt", "w") do io
#     writedlm(io, vcat(Us', ε))
# end

# degenerate theory

J = 1 # setting to 1 so that `U` is measured in units of `J`
f = 2
ω = 20

U₀ = ω * 2/3
E_D₀ = [0, U₀, 2U₀]

U₀ = ω
E_D₀ = [0]

# construct the lattice to get basis states, and calculate the zeroth-order spectrum (for U = U₀)
lattice = Lattice(;dims=(1, ncells), nbozons=5, isperiodic=true)
lattice = Lattice(;dims=(2, 3), nbozons=6, isperiodic=true)
nstates = length(lattice.basis_states)
E₀ = zeros(nstates)
for (index, state) in enumerate(lattice.basis_states)
    for n_i in state
        if (n_i > 1)
            E₀[index] += U₀/2 * n_i * (n_i - 1)
        end
    end
end
scatter(E₀, markersize=0.5, markerstrokewidth=0)
range6U = (findfirst(==(6U₀), E₀), findlast(==(6U₀), E₀)) # range of states of energy 6U

# space_of_state[i] stores the subspace number (𝐴, 𝑎) of i'th state,
space_of_state = map(E₀) do E
    for A in eachindex(E_D₀)
        M = (E - E_D₀[A]) / ω
        M_int = round(Int, M)
        if isapprox(M, M_int, atol=0.01)
            return (A, M_int)
        end
    end
    return (-1, -1) # this basically signifies a mistake in user's choice of `E_D₀`
end

scatter!(1:nstates, i -> space_of_state[i][2], markersize=0.5, markerstrokewidth=0)
plot!(legend=false)
bh = BoseHamiltonian(lattice, J, U₀, f, ω, space_of_state, type=:largeU, order=2);

e0 = bh.H[1,1]
e3 = bh.H[c03[1], c03[1]]
c = bh.H[1, c03[1]]
sqrt((e0-e3)^2+4ncells*c^2)

M = copy(bh.H);
N = copy(bh.H);
M[diagind(M)] .= 0;
N[diagind(M)] .= 0;
M[52:81, 102:121] .= 0
M[102:121, 52:81] .= 0
d = M[diagind(M)]
M[52:81, 52:81] .= 0
M[diagind(M)] .= d
f2 = heatmap(abs.(bh.H), yaxis=:flip, c=:viridis)
f2 = heatmap(abs.(M), yaxis=:flip, c=:viridis)
plot(bh.H[diagind(bh.H)])

A = 1
As = findall(s -> s[1] == A, space_of_state) # As store numbers of state that belong to space `A`
h = zeros(length(As), length(As)) # reduced matrix of the subspace of interest
nU = 300 # 2:54 for N=9
spectrum = Matrix{Float64}(undef, length(As), nU)
scatter(bh.H[diagind(bh.H)][As], markersize=0.5, markerstrokewidth=0)

c03 = findall(abs.(bh.H[1, :]) .> 0)[2:end] # numbers of states which ground state is coupled to
c36 = findall(abs.(bh.H[c03[1], range6U[1]:range6U[2]]) .> 0) .+ range6U[1] .- 1 # numbers of states from 6U manifold which states 3U is coupled to
bh.H[c36[1], c36[1]]
bh.H[c03[1], c36[1]]
bh.basis_states[c03]
R = zeros(1+length(c03)+length(c36), 1+length(c03)+length(c36)) # isolated matrix
R[1, 1] = bh.H[1, 1]
R[1, range(2, length=length(c03))] .= bh.H[1, c03]
R[diagind(R)[range(2, length=length(c03))]] .= bh.H[c03[1], c03[1]]
R[1, length(c03)+1:end] .= bh.H[1, c03]
spectrum = Matrix{Float64}(undef, ncells+1, nU)

nU = 300 # 46 s for N=7; eigvals takes the main time
spectrum = Matrix{Float64}(undef, nstates, nU)
Us = range(U₀-1, U₀+1, nU)
Us = range(12.5, 15.5, nU)

n_isol = 5
spectrum = Matrix{Float64}(undef, n_isol, nU)

@showprogress for (iU, U) in enumerate(Us)
    bh = BoseHamiltonian(lattice, J, U, f, ω, space_of_state, type=:largeU, order=2)
    # spectrum[:, iU] = eigvals(Symmetric(Matrix(bh.H)))

    for i in eachindex(As), j in i:length(As)
        h[j, i] = bh.H[As[j], As[i]]
    end
    spectrum[:, iU] = eigvals(Symmetric(h, :L))

    # e, S = eigen(Symmetric(h, :L))
    # sp = sortperm(S[1, :], rev=true)
    # spectrum[:, iU] = e[sp[1:n_isol]]


    # M = Matrix(bh.H)
    # M[52:81, 102:121] .= 0
    # M[102:121, 52:81] .= 0
    # d = M[diagind(M)]
    # M[52:81, 52:81] .= 0
    # M[diagind(M)] .= d
    # spectrum[:, iU] = eigvals(Symmetric(M))

    # R[1, 1] = bh.H[1, 1]
    # R[1, 2:end] .= bh.H[1, c03[1]]
    # R[diagind(R)[2:end]] .= bh.H[c03[1], c03[1]]
    # spectrum[:, iU] = eigvals(Symmetric(R))
end

gr()
plotlyjs()
spectrum .%= ω
spectrum[spectrum .< 0] .+= ω
fig = scatter(Us, spectrum', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, widen=false);
scatter!(Us, (spectrum .- 20)', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, widen=false);
ylims!(-3, 3.5);
ylims!(-1, 5)
ylims!(-10, 10)
title!(L"N=%$ncells"*", isolated")
plot!(xlims=(U₀-1, U₀+1), ylims=(-2, 2), title="isolated")
# plot!(xlims=(U₀-1, U₀+1), ylims=(-2, 2), title=L"\langle 3|W|6\rangle = \langle 3|W|3\rangle = 0")
xlims!(19, 21)
vline!([U₀], c=:white)
plot!(fo2, ylabel="")

plot(full, no6, no36, isol)
savefig("N=$ncells-isolated.png")

sp = Matrix{Float64}(undef, 2(ncells+1), nU)
sp[1:9, :] .= spectrum
sp[10:18, :] .= spectrum .- 20
diffs = zeros(nU)
for (i, col) in enumerate(eachcol(sp))
    ext = extrema(col)
    diffs[i] = ext[2] - ext[1]
end
plot(Us, diffs)
fig = scatter(Us, sp', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, widen=false)
for i in eachindex(sp)
    if sp[i] < -5 || sp[i] > 10
        sp[i] = 1
    end
end

using DelimitedFiles
Us = range(12, 15, 1000)
ε = readdlm("f2_U12-15.txt")
# Us = range(0, 45, 1000)
# ε = readdlm("f2_U45.txt")

fig = scatter(Us, ε', xlabel=L"U/J", ylabel=L"\varepsilon/J", markersize=0.5, markerstrokewidth=0, c=1, legend=false, ticks=:native, title="exact", widen=false)
plot!(fig, ylims=(-2, 2), xlims=(U₀-1, U₀+1))

plot(fig, fo2, fo1, layout=(1, 3), link=:y)
savefig("resonance13_zoom1.png")

# open("U13-3x3-dpt.txt", "w") do io
#     writedlm(io, vcat(Us', spectrum))
# end