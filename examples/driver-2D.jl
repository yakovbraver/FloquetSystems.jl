# julia --project --check-bounds=no -t 128 -e 'include("examples/driver-2D.jl")' -- n_q sf &
@time "Loading package" using FloquetSystems

ϵ = 0.1
ϵc = 1
χ = 0

ω = 2000
n_spatial_harmonics = 20
n_floquet_harmonics = 0
n_q, subfactor = parse.(Int, ARGS)
fgf = FloquetGaugeField(ϵ, ϵc, χ; subfactor, n_floquet_harmonics, n_spatial_harmonics, fft_threshold=1e-3);
E_target = 12
qys = range(-1, 1, Threads.nthreads())
qxs = [0]
@time "Warm-up calculation" E = spectrum(fgf, ω, E_target, qxs, qys, nsaves=10);

n_spatial_harmonics = 66
n_floquet_harmonics = 4
qys = range(-1, 1, n_q)
fgf = FloquetGaugeField(ϵ, ϵc, χ; subfactor, n_floquet_harmonics, n_spatial_harmonics, fft_threshold=1e-3); # 80 procs 442 GB -> out of memory; 64 procs 356 GB (13:15)
GC.gc()
@time "Actual calculation" E = spectrum(fgf, ω, E_target, qxs, qys, nsaves=50);

using JLD2
jldsave("omega$(ω)_sf$(subfactor)_ns$(n_spatial_harmonics)_nf$(n_floquet_harmonics).jld2"; E)