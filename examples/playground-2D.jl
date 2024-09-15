includet("../src/GaugeFields.jl")
using .GaugeFields

using LinearAlgebra, Plots, LaTeXStrings, SparseArrays
CMAP = cgrad(:linear_grey_0_100_c0_n256, rev=true);
plotlyjs()
theme(:dark, size=(600, 400))

x = range(-0.1*2π, 2π*1.1, 500) # in units of 1/kᵣ
ϵ = 0.1f0
ϵc = 1
χ = 0
gf = GaugeField(ϵ, ϵc, χ; n_harmonics=5)

U = 𝑈(x, x; ϵ=Float32(ϵ), ϵc, χ)
u = rfft(U) |> real
uu = u[1, 1]
heatmap(x, x, U, c=CMAP, xlabel=L"x / (1/k_R)", ylabel=L"y / (1/k_R)")
plot(x, U[125, :] / 2pi, xlabel=L"x/a", legend=false)
savefig("cut.pdf")

L = π # period of the structure
n = 3 # make cell edge `n` times smaller
Δδ = L/n
U = zeros(length(x), length(x))
for j in 0:n-1, i in 0:n-1
    m = j*n+i # number of performed iterations
    U .= (U.*m .+ 𝑈(x.+i*Δδ, x.+j*Δδ; ϵ, ϵc, χ)) ./ (m+1) # on-line average
end
heatmap(x, x, U ./ (1/ϵ^2), c=CMAP, xlabel=L"x / (1/k_R)", ylabel=L"y / (1/k_R)")
u = rfft(U) .* (dx/L)^2
sum(abs.(imag.(u)))
u[1,1] = 0
heatmap(real.(u), c=CMAP)
@time GaugeField(ϵ, ϵc, χ; n_harmonics=30, fft_threshold=0.01);
@time E = spectrum(gf, 10)
heatmap(E, c=CMAP)

ϵ = 0.1f0
ϵc = 1
χ = 0
gf = GaugeField(ϵ, ϵc, χ; n_harmonics=10, fft_threshold=1e-2)
H = sparse(gf.H_rows, gf.H_cols, gf.H_vals)
heatmap(abs.(H), c=CMAP, yaxis=:flip, title="H")

qxs = range(-1, 1, 25)
qys = [0]
nsaves = 8
@time e = GaugeFields.spectrum(gf, qxs, qys; nsaves)
scatter(qxs, e[:, :, 1]', c=1, markerstrokewidth=0, markersize=2, legend=false)

n_q = 5; qs = range(0, 1, n_q)
@time E = GaugeFields.spectrum(gf; n_q);
heatmap(qs, qs, E, c=CMAP)

ω = 5000
n_spatial_harmonics = 20
@time fgf = FloquetGaugeField(ϵ, ϵc, χ; subfactor=3, n_floquet_harmonics=0, n_spatial_harmonics, fft_threshold=1e-2)
Q = sparse(fgf.Q_rows, fgf.Q_cols, fgf.Q_vals)
maximum(fgf.Q_rows)
heatmap(abs.(Q), c=CMAP, yaxis=:flip, title="Q")

E_target = 2
qxs = range(-1, 1, 25)
qys = [0]
@time E = spectrum(fgf, ω, E_target, qxs, qys; nsaves=8);
scatter(qxs, E[:, :, 1]', c=1, markerstrokewidth=0, markersize=1, legend=false, title="n_spatial_harmonics=$n_spatial_harmonics")