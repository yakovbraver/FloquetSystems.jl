# julia --project --check-bounds=no -t 128 -e 'include("examples/driver-2D.jl")' -- sf &
# julia --project --check-bounds=no -p 128 -e 'include("examples/driver-2D.jl")' -- sf &
@everywhere include("../src/GaugeFields.jl")
using .GaugeFields

ϵ = 0.1f0
ϵc = 1
χ = 0

### Warm-up
ω = 2000
n_spatial_harmonics = 20
n_floquet_harmonics = 0
subfactor = parse(Int, ARGS[1])
fgf = FloquetGaugeField(ϵ, ϵc, χ; subfactor, n_floquet_harmonics, n_spatial_harmonics, fft_threshold=1e-2)
E_target = 12
qys = range(-1, 1, 256)
qxs = [0]
@time E = GaugeFields.spectrum(fgf, ω, E_target, qxs, qys, nsaves=50);

# actual
n_spatial_harmonics = 66
n_floquet_harmonics = 4
fgf = FloquetGaugeField(ϵ, ϵc, χ; subfactor, n_floquet_harmonics, n_spatial_harmonics, fft_threshold=1e-2)
@time E = GaugeFields.spectrum(fgf, ω, E_target, qxs, qys, nsaves=50);

using JLD2
jldsave("omega$(ω)_sf$(subfactor)_ns$(n_spatial_harmonics)_nf$(n_floquet_harmonics).jld2"; E)