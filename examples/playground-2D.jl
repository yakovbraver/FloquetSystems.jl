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

U = 𝑈(gf, x, x)
heatmap(x, x, U ./ (1/ϵ^2), c=CMAP, xlabel=L"x / (1/k_R)", ylabel=L"y / (1/k_R)")
plot(x, U[125, :] / 2pi, xlabel=L"x/a", legend=false)
savefig("cut.pdf")

# anti-clockwise shaking
L = π # period of the structure
n = 3 # make cell edge `n` times smaller
Δδ = L/n
U = zeros(length(x), length(x))
for j in 0:n-1, i in 0:n-1
    m = j*n+i # number of performed iterations
    U .= (U.*m .+ 𝑈(gf, x.+i*Δδ, x.+j*Δδ)) ./ (m+1) # on-line average
end
heatmap(x, x, U ./ (1/ϵ^2), c=CMAP, xlabel=L"x / (1/k_R)", ylabel=L"y / (1/k_R)")

gf = GaugeField(Float32(ϵ), ϵc, χ; n_harmonics=100, fft_threshold=0.01)
@time E = spectrum(gf, 10)
@time E = spectrum(gf2, 10)
heatmap(E, c=CMAP)

ϵ = 0.1 # testing for Float64
ϵc = 1
χ = 0
gf = GaugeField(ϵ, ϵc, χ; n_harmonics=2, fft_threshold=0.05)
heatmap(abs.(sparse(gf.H_rows, gf.H_cols, gf.H_vals)), yaxis=:flip)

ω = 1000
fgf = FloquetGaugeField(ϵ, ϵc, χ, ω, 0, 0; subfactor=2, n_floquet_harmonics=3, n_fourier_harmonics=5)
heatmap(abs.(sparse(fgf.Q_rows, fgf.Q_cols, fgf.Q_vals)), yaxis=:flip)