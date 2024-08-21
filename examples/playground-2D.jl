includet("../src/GaugeFields.jl")
using .GaugeFields

using LinearAlgebra, Plots, LaTeXStrings
CMAP = cgrad(:linear_grey_0_100_c0_n256, rev=true)
plotlyjs()
theme(:dark, size=(800, 600))

x = range(-0.1, 1.1, 500)
ϵ = 0.05
ϵc = 1
χ = 0
lf = LightField(ϵ, ϵc, χ)

# diagonal shaking
n = 3 # make cell edge `n` times smaller
U = 𝑈(lf, x, x)
Δδ = 0.5/n
for i in 1:n-1
    U .= (U.*i .+ 𝑈(lf, x.+i*Δδ, x.+i*Δδ)) ./ (i+1) # on-line average
end
heatmap(x, x, U ./ (1/ϵ^2), c=CMAP)

# anti-clockwise shaking
n = 3 # make cell edge `n` times smaller
Δδ = 0.5/n
U = zeros(length(x), length(x))
for j in 0:n-1, i in 0:n-1
    m = j*n+i # number of performed iterations
    U .= (U.*m .+ 𝑈(lf, x.+i*Δδ, x.+j*Δδ)) ./ (m+1) # on-line average
end
heatmap(x, x, U ./ (1/ϵ^2), c=CMAP)

# anti-clockwise shaking
# n = 3 # make cell edge `n` times smaller
# Δδ = 0.5/n
# U = 𝑈(lf, x, x)
# for i in 1:n-1
#     U .= (U.*i .+ 𝑈(lf, x.+i*Δδ, x)) ./ (i+1) # on-line average
# end
# for i in 1:n
#     U .= (U.*(n-1+i) .+ 𝑈(lf, x.+(n-1)*Δδ, x.+i*Δδ)) ./ (n+i) # on-line average
# end
# heatmap(x, x, U ./ (1/ϵ^2), c=CMAP)