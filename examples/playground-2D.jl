includet("../src/GaugeFields.jl")
using .GaugeFields

using LinearAlgebra, Plots, LaTeXStrings, SparseArrays
CMAP = cgrad(:linear_grey_0_100_c0_n256, rev=false);
plotlyjs()
theme(:dark, size=(600, 500)) 

### Scalar potential
x = range(-0.1*2π, 2π*1.1, 500) # in units of 1/kᵣ
ϵ = 0.1f0
ϵc = 1
χ = 0
U = 𝑈(x, x; ϵ=Float32(ϵ), ϵc, χ)
heatmap(x ./ 2π, x ./ 2π, U, c=CMAP, xlabel=L"x / a", ylabel=L"y / a", title=L"U_D(x,y)") # plot x in units of a = 2π/kᵣ
savefig("U_D.png")
plot(x, U[125, :] / 2pi, xlabel=L"x/a", legend=false)
savefig("cut.pdf")

### Averaged potential
L = π # period of the structure
n = 3 # make cell edge `n` times smaller
Δδ = L/n
U = zeros(length(x), length(x))
for j in 0:n-1, i in 0:n-1
    m = j*n+i # number of performed iterations
    U .= (U.*m .+ 𝑈(x.+i*Δδ, x.+j*Δδ; ϵ, ϵc, χ)) ./ (m+1) # on-line average
end
heatmap(x ./ 2π, x ./ 2π, U, c=CMAP, xlabel=L"x / a", ylabel=L"y / a", title="Averaged potential")
savefig("averaged.png")

### Vector potential
x = range(-0.1*2π, 2π*1.1, 50) # in units of 1/kᵣ
χ = π/2
A = 𝐴(x, x; ϵ, ϵc, χ, normalisation=0.01)
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))
X, Y = meshgrid(x, x)
quiver(X ./ 2π, Y ./ 2π, gradient=vec(A), xlabel=L"x / a", ylabel=L"y / a", title=L"\vec{A}(x,y)")

### Lowest band dispersion
@time GaugeField(ϵ, ϵc, χ; n_harmonics=10, fft_threshold=0.01);
@time E = spectrum(gf, n_q=10)
heatmap(E, c=CMAP)
E2 = reverse(E, dims=2)
E3 = reverse(E2, dims=1)
E4 = reverse(E, dims=1)
E_full = [E E2; E4 E3] # not entirely correct because central axes are contained twice
heatmap(E_full, c=CMAP)
surface(range(-1, 1, 20), range(-1, 1, 20), E_full, c=CMAP, xlabel=L"q_x / k_R", ylabel=L"q_x / k_R", zlabel="Energy")
savefig("dispersion.png")

### Cut of the lowest band dispersion
ϵ = 0.1f0
ϵc = 1
χ = 0
gf = GaugeField(ϵ, ϵc, χ; n_harmonics=50, fft_threshold=1e-2)
qxs = range(-1, 1, 25)
qys = [0]
nsaves = 8
@time e = GaugeFields.spectrum(gf, qxs, qys; nsaves)
scatter(qxs, e[:, :, 1]', c=1, markerstrokewidth=0, markersize=2, legend=false)

### q = 0 wavefunctions
gf = GaugeField(ϵ, ϵc, χ; n_harmonics=20, fft_threshold=1e-2)
E, V = GaugeFields.q0_states(gf)
whichstate = 200
xs, wf = GaugeFields.make_wavefunction(V[:, whichstate], 201)
surface(xs ./ 2π, xs ./ 2π, abs2.(wf), xlabel="x / a", ylabel="y / a", title="state no. $whichstate")
xu = range(0, π, 500) # coordinates for potential
U = 𝑈(xu, xu; ϵ=Float32(ϵ), ϵc, χ)
surface!(xu ./ 2π, xu ./ 2π, U ./ (maximum(U)/maximum(abs2, wf)), c=CMAP) # plot x in units of a = 2π/kᵣ

### Floquet spectrum
ω = 1000
n_spatial_harmonics = 40
n_floquet_harmonics = 4
@time fgf = FloquetGaugeField(ϵ, ϵc, χ; subfactor=3, n_floquet_harmonics, n_spatial_harmonics, fft_threshold=1e-2)
Q = sparse(fgf.Q_rows, fgf.Q_cols, fgf.Q_vals)
heatmap(abs.(Q), c=CMAP, yaxis=:flip, title="Q")

E_target = 10
qxs = range(-1, 1, 200)
qys = [0]
@time E = spectrum(fgf, ω, E_target, qxs, qys; nsaves=200);
scatter(qxs, E[:, :, 1]', c=1, markerstrokewidth=0, markersize=1, legend=false, ylims=(6, 20),
        title=L"\omega=%$(ω)", xlabel=L"q_x/k_R", ylabel="quasienergy")
savefig("omega$(ω).png")
writedlm("omega$(ω)_ns$(n_spatial_harmonics)_nf$(n_floquet_harmonics).txt", E[:, :, 1]')