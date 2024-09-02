includet("../src/GaugeFields.jl")
using .GaugeFields

using LinearAlgebra, Plots, LaTeXStrings
CMAP = cgrad(:linear_grey_0_100_c0_n256, rev=true)
plotlyjs()
theme(:dark, size=(800, 600))

x = range(-0.1*2π, 2π*1.1, 500) # in units of 1/kᵣ
ϵ = 0.1
ϵc = 1
χ = 0
gf = GaugeField(ϵ, ϵc, χ; n_harmonics=5)

U = 𝑈(gf, x, x)
heatmap(x, x, U ./ (1/ϵ^2), c=CMAP, xlabel=L"x / (1/k_R)", ylabel=L"y / (1/k_R)")

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

using SparseArrays, LinearAlgebra
gf = GaugeField(ϵ, ϵc, χ; n_harmonics=30, fft_threshold=0.05)
E = spectrum(gf, 10)
heatmap(E, c=CMAP)
