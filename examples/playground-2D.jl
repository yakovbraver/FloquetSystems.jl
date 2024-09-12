includet("../src/GaugeFields.jl")
using .GaugeFields

using LinearAlgebra, Plots, LaTeXStrings, SparseArrays
CMAP = cgrad(:linear_grey_0_100_c0_n256, rev=true);
plotlyjs()
theme(:dark, size=(800, 600))

x = range(-0.1*2π, 2π*1.1, 500) # in units of 1/kᵣ
ϵ = 0.1
ϵc = 1
χ = 0
gf = GaugeField(ϵ, ϵc, χ; n_harmonics=5)

U = 𝑈(x, x; ϵ=Float32(ϵ), ϵc, χ)
u = rfft(U) |> real
uu = u[1, 1]
heatmap(x, x, U, c=CMAP, xlabel=L"x / (1/k_R)", ylabel=L"y / (1/k_R)")
plot(x, U[125, :] / 2pi, xlabel=L"x/a", legend=false)
savefig("cut.pdf")

# anti-clockwise shaking``
L = π # period of the structure
n = 3 # make cell edge `n` times smaller
Δδ = L/n
U = zeros(length(x), length(x))
for j in 0:n-1, i in 0:n-1
    m = j*n+i # number of performed iterations
    U .= (U.*m .+ 𝑈(gf, x.+i*Δδ, x.+j*Δδ)) ./ (m+1) # on-line average
end
heatmap(x, x, U ./ (1/ϵ^2), c=CMAP, xlabel=L"x / (1/k_R)", ylabel=L"y / (1/k_R)")

@time GaugeField(Float32(ϵ), ϵc, χ; n_harmonics=50, fft_threshold=0.01);
@time E = spectrum(gf, 10)
heatmap(E, c=CMAP)

ϵ = 0.1 # testing for Float64
ϵc = 1
χ = 0
gf = GaugeField(ϵ, ϵc, χ; n_harmonics=3, fft_threshold=0.05)
H = sparse(gf.H_rows, gf.H_cols, gf.H_vals)
heatmap(abs.(H), yaxis=:flip)

ω = 1000
@time fgf = FloquetGaugeField(Float32(ϵ), ϵc, χ; subfactor=2, n_floquet_harmonics=10, n_fourier_harmonics=50)
target = 30
qxs = [0]
qys = [0]
@time E = spectrum(fgf, ω, target, qxs, qys, 20);
scatter(E[:, 1, 1])