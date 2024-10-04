includet("../src/GaugeFields.jl")
using .GaugeFields

using Plots, LaTeXStrings, JLD2, LinearAlgebra, SparseArrays
CMAP = cgrad(:linear_grey_0_100_c0_n256, rev=false);
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
plotlyjs()
theme(:dark, size=(600, 500)) 

### Scalar potential
x = range(-0.1*2π, 2π*1.1, 500) # in units of 1/kᵣ
ϵ = 0.1f0
ϵc = 1
χ = 0
U = 𝑈(x, x; ϵ, ϵc, χ)
@time 𝑈(x, x; ϵ, ϵc, χ);
heatmap(x ./ 2π, x ./ 2π, U', c=cmap_rainbow, xlabel=L"x / a", ylabel=L"y / a", title=L"U_D(x,y)") # plot x in units of 𝑎 = 2π/kᵣ
savefig("U_D.pdf")
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
heatmap(x ./ 2π, x ./ 2π, U', c=CMAP, xlabel=L"x / a", ylabel=L"y / a", title="Averaged potential")
savefig("averaged.png")

### Vector potential
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))
x = range(-0.1*2π, 2π*1.1, 500) # in units of 1/kᵣ
χ = π/2
A = 𝐴(x, x; ϵ, ϵc, χ)
A_abs2 = map(x -> x[1]^2 + x[2]^2, A)
heatmap(x ./ 2π, x ./ 2π, A_abs2', c=cmap_rainbow, xlabel=L"x / a", ylabel=L"y / a", title=L"\vec{A}^2(x,y)")
savefig("A-abs.pdf")

# components
Aₓ = [A[I][1] for I in CartesianIndices(A)]
heatmap(x ./ 2π, x ./ 2π, Aₓ', c=:coolwarm, xlabel=L"x / a", ylabel=L"y / a", title=L"A_x(x,y)")
savefig("Ax.pdf")
Ay = [A[I][2] for I in CartesianIndices(A)]
heatmap(x ./ 2π, x ./ 2π, Ay', c=:coolwarm, xlabel=L"x / a", ylabel=L"y / a", title=L"A_y(x,y)")
savefig("Ay.pdf")

# quiver
plotlyjs()
theme(:dark, size=(1000, 1000))
X, Y = meshgrid(x, x)
normalisation = 0.02
max_A = √maximum(x -> x[1]^2 + x[2]^2, A)
map!(x -> x ./ max_A .* normalisation, A, A)
quiver(X ./ 2π, Y ./ 2π, gradient=vec(A), xlabel="x / a", ylabel="y / a", title="A(x,y)", lw=2, widen=false)
quiver(X ./ 2π, Y ./ 2π, gradient=vec(A), xlabel=L"x / a", ylabel=L"y / a", title=L"\vec{A}(x,y)")

# divergence
∇A = ∇𝐴(x, x; ϵ, ϵc, χ)
heatmap(x ./ 2π, x ./ 2π, ∇A', c=:coolwarm, xlabel=L"x / a", ylabel=L"y / a", title=L"\nabla \vec{A}(x,y)")
savefig("divA.pdf")

### Magnetic field
B = 𝐵(x, x; ϵ, ϵc, χ)
heatmap(x ./ 2π, x ./ 2π, B', c=:coolwarm, xlabel=L"x / a", ylabel=L"y / a", title=L"B_z(x,y)") # plot x in units of 𝑎 = 2π/kᵣ
savefig("B.pdf")

### Lowest band dispersion
gf = GaugeField(ϵ, ϵc, χ; n_harmonics=20);
n_q = 50
@time E = spectrum(gf; n_q)
heatmap(E, c=CMAP)
E2 = reverse(E, dims=2)
E3 = reverse(E2, dims=1)
E4 = reverse(E, dims=1)
E_full = [E3 E4; E2 E] # not entirely correct because central axes are contained twice
heatmap(E_full, c=CMAP)
surface(range(-1, 1, 2n_q), range(-1, 1, 2n_q), E_full, c=CMAP, xlabel=L"q_x / k_R", ylabel=L"q_y / k_R", zlabel="Energy")
savefig("dispersion.png")
contour(range(-1, 1, 2n_q), range(-1, 1, 2n_q), c=:viridis, E_full, xlabel=L"q_x / k_R", ylabel=L"q_y / k_R", zlabel="Energy", minorticks=6, ticks=-1:1:1, minorgrid=true)
savefig("dispersion-contour.png")

### Cut of the lowest band dispersion
ϵ = 0.1f0
ϵc = 1
χ = 0
gf = GaugeField(ϵ, ϵc, χ; n_harmonics=24)
# H = sparse(gf.H_rows, gf.H_cols, gf.H_vals)
# heatmap(abs.(H), c=:viridis, yaxis=:flip, title="Q")
qxs = range(-1, 1, 25)
qys = [0]
nsaves = 8
@time e = GaugeFields.spectrum(gf, qxs, qys; nsaves)
scatter(qxs, e[:, :, 1]', c=1, markerstrokewidth=0, markersize=2, legend=false)

### q = 0 wavefunctions
gf = GaugeField(ϵ, ϵc, χ; n_harmonics=20)
E, V = GaugeFields.q0_states(gf)
whichstate = 200
xs, wf = GaugeFields.make_wavefunction(V[:, whichstate], 201)
surface(xs ./ 2π, xs ./ 2π, abs2.(wf), xlabel="x / a", ylabel="y / a", title="state no. $whichstate")
xu = range(0, π, 500) # coordinates for potential
U = 𝑈(xu, xu; ϵ=Float32(ϵ), ϵc, χ)
surface!(xu ./ 2π, xu ./ 2π, U ./ (maximum(U)/maximum(abs2, wf)), c=CMAP) # plot x in units of a = 2π/kᵣ

### all eigenvalues for a specific qx, qy pair
gf = GaugeField(ϵ, ϵc, χ; n_harmonics=10);
qx = [0]; qy =[0.5]
e = spectrum(gf, qx, qy)
plotlyjs()
scatter(e[:, 1, 1], markersize=1, markerstrokewidth=0)

### Floquet spectrum
ω = 400
n_spatial_harmonics = 24
n_floquet_harmonics = 2
ϵ = 0.1
ϵc = 1
χ = 0
@time fgf = FloquetGaugeField(ϵ, ϵc, χ; subfactor=3, n_floquet_harmonics, n_spatial_harmonics, fft_threshold=1e-5);

# sparse variant

@time Q = sparse(fgf.Q_rows, fgf.Q_cols, fgf.Q_vals);
plotlyjs()
heatmap(abs.(Q), c=:viridis, yaxis=:flip, title="Q")

E_target = 12
qys = range(-1, 1, 256)
qxs = [0]
@time E = spectrum(fgf, ω, E_target, qxs, qys; nsaves=50);
heatmap(abs.(Matrix(Q)), yaxis=:flip)
scatter(qys, E[:, 1, :]', c=1, markerstrokewidth=0, markersize=1, legend=false, ylims=(6, 18),
        title=L"\omega=%$(ω)", xlabel=L"q_x/k_R", ylabel="quasienergy")
savefig("omega$(ω).png")
jldsave("omega$(ω)_ns$(n_spatial_harmonics)_nf$(n_floquet_harmonics).jld2"; E)

# dense variant

E_target = (6, 18)
qys = range(-1, 1, 256)
qxs = [0]
@time E = GaugeFields.spectrum_dense(fgf, ω, E_target, qxs, qys);
fig = plot();
for i in eachindex(qys)
    scatter!(fill(qys[i], length(E[1, i])), E[1, i], c=1, markerstrokewidth=0, markersize=1, legend=false, xlims=(-1, 1), ylims=E_target,
            title=L"\omega=%$(ω)", xlabel=L"q_y/k_R", ylabel="quasienergy")
end
# for i in eachindex(qxs)
#     scatter!(fill(qxs[i], length(E[i, 1])), E[i, 1], c=1, markerstrokewidth=0, markersize=1, legend=false,
#             title=L"\omega=%$(ω)", xlabel=L"q_y/k_R", ylabel="quasienergy")
# end
fig

subfactor = 3
savefig("omega$(ω)_sf$(subfactor).png")
E = load("omega$(ω)_sf$(subfactor)_ns$(n_spatial_harmonics)_nf$(n_floquet_harmonics).jld2")["E"]